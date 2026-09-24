import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import urllib.request
from urllib.error import HTTPError
from urllib.response import addinfourl
from email.message import Message

from llm_configurator import downloads, runtime
from llm_configurator.catalogue import demo_variants
from llm_configurator.domain import GIB, Cancelled, Variant


def sha(data):
    return hashlib.sha256(data).hexdigest()


def make_variant(files):
    """files: {name: bytes}; one entry makes a single-file variant, more make a sharded one."""
    record = demo_variants()[0].to_dict()
    names = list(files)
    record.update(demo=False, repo="org/model-GGUF", revision="main", filename=names[0], sha256=sha(files[names[0]]),
                  size_bytes=sum(len(v) for v in files.values()),
                  files=[] if len(files) == 1 else [{"filename": n, "size_bytes": len(files[n]), "sha256": sha(files[n])} for n in names])
    return Variant(**record)


class Response:
    def __init__(self, body, status=200, headers=None, fail_after=None, on_read=None):
        self.stream, self.status = io.BytesIO(body), status
        self.headers = {"Content-Length": str(len(body)), **(headers or {})}
        self.fail_after, self.on_read, self.sent = fail_after, on_read, 0

    def read(self, size):
        if self.fail_after is not None and self.sent >= self.fail_after:
            raise ConnectionResetError("connection reset")
        chunk = self.stream.read(size)
        self.sent += len(chunk)
        if self.on_read:
            self.on_read(self.sent)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class Hub:
    """Fake Hugging Face: serves files by name, honours Range unless told not to, can run scripted replies first."""

    def __init__(self, files, ignore_range=False):
        self.files, self.ignore_range, self.requests, self.script = files, ignore_range, [], []

    def reply(self, request, **kwargs):
        name = request.full_url.rsplit("/", 1)[1]
        body = self.files[name]
        rng = request.get_header("Range")
        if rng and not self.ignore_range:
            start = int(rng.split("=")[1].rstrip("-"))
            return Response(body[start:], 206, {"Content-Range": f"bytes {start}-{len(body) - 1}/{len(body)}"}, **kwargs)
        return Response(body, 200, **kwargs)

    def __call__(self, request, timeout=None):
        self.requests.append(request)
        if self.script:
            step = self.script.pop(0)
            return step(self, request)
        return self.reply(request)


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.env = patch.dict(os.environ, {}, clear=False)
        self.env.start()
        os.environ.pop("HF_TOKEN", None)
        os.environ.pop("HF_ENDPOINT", None)
        patches = [patch.object(downloads, "CHUNK", 7), patch.object(downloads, "_sleep", lambda s, c: None)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def run_download(self, variant, hub, **kwargs):
        with patch.object(downloads, "_open", hub):
            return downloads.download_variant(variant, self.dir, **kwargs)

    def test_full_download_verifies_and_uses_endpoint_and_token(self):
        body = b"GGUF" + bytes(range(200))
        variant = make_variant({"m.gguf": body})
        hub = Hub({"m.gguf": body})
        events = []
        path = self.run_download(variant, hub, progress=events.append, token="secret")
        self.assertEqual(path, self.dir / "m.gguf")
        self.assertEqual(path.read_bytes(), body)
        self.assertFalse((self.dir / "m.gguf.part").exists())
        self.assertEqual(hub.requests[0].full_url, "https://huggingface.co/org/model-GGUF/resolve/main/m.gguf")
        self.assertEqual(hub.requests[0].get_header("Authorization"), "Bearer secret")
        self.assertEqual(events[-1]["stage"], "done")
        self.assertEqual(events[-1]["done"], len(body))
        self.assertEqual(events[-1]["total"], len(body))
        for key in ("bytes_per_second", "eta_seconds", "file", "file_index", "file_count", "message"):
            self.assertIn(key, events[-1])
        self.assertEqual({e["unit"] for e in events}, {"bytes"})   # done/total are byte counts in every stage
        # Running again reuses the verified file without touching the network.
        self.assertEqual(self.run_download(variant, Hub({})), path)

    def test_hf_endpoint_and_env_token(self):
        body = b"x" * 30
        variant = make_variant({"dir/m.gguf": body})
        os.environ["HF_ENDPOINT"] = "https://mirror.example/"
        os.environ["HF_TOKEN"] = "envtoken"
        hub = Hub({"m.gguf": body})
        self.run_download(variant, hub)
        self.assertEqual(hub.requests[0].full_url, "https://mirror.example/org/model-GGUF/resolve/main/dir/m.gguf")
        self.assertEqual(hub.requests[0].get_header("Authorization"), "Bearer envtoken")
        self.assertTrue((self.dir / "m.gguf").exists())
        os.environ["HF_ENDPOINT"] = "http://mirror.example"
        with self.assertRaisesRegex(ValueError, "https"):
            self.run_download(make_variant({"n.gguf": body}), Hub({"n.gguf": body}))

    def test_sharded_model_returns_first_shard_and_reports_total(self):
        shards = {"q/m-00001-of-00003.gguf": b"a" * 50, "q/m-00002-of-00003.gguf": b"b" * 40, "q/m-00003-of-00003.gguf": b"c" * 5}
        variant = make_variant(shards)
        hub = Hub({Path(k).name: v for k, v in shards.items()})
        events = []
        with patch.object(downloads, "PROGRESS_INTERVAL", 0):
            path = self.run_download(variant, hub, progress=events.append)
        self.assertEqual(path, self.dir / "m-00001-of-00003.gguf")
        for name, data in shards.items():
            self.assertEqual((self.dir / Path(name).name).read_bytes(), data)
        self.assertEqual({e["total"] for e in events}, {95})
        self.assertEqual(events[-1]["done"], 95)
        self.assertEqual({e["file_index"] for e in events}, {0, 1, 2})
        dones = [e["done"] for e in events]
        self.assertEqual(dones, sorted(dones))

    def test_resume_with_206(self):
        body = bytes(range(256)) * 2
        variant = make_variant({"m.gguf": body})
        (self.dir / "m.gguf.part").write_bytes(body[:300])
        hub = Hub({"m.gguf": body})
        self.run_download(variant, hub)
        self.assertEqual(hub.requests[0].get_header("Range"), "bytes=300-")
        self.assertEqual((self.dir / "m.gguf").read_bytes(), body)

    def test_server_ignoring_range_restarts_from_zero(self):
        body = bytes(range(256))
        variant = make_variant({"m.gguf": body})
        (self.dir / "m.gguf.part").write_bytes(b"garbage-" * 4)  # wrong bytes: a restart must not keep them
        events = []
        self.run_download(variant, Hub({"m.gguf": body}, ignore_range=True), progress=events.append)
        self.assertEqual((self.dir / "m.gguf").read_bytes(), body)
        self.assertEqual(events[-1]["done"], len(body))

    def test_mismatched_content_range_restarts(self):
        body = bytes(range(100))
        variant = make_variant({"m.gguf": body})
        (self.dir / "m.gguf.part").write_bytes(body[:40])
        hub = Hub({"m.gguf": body})
        hub.script = [lambda h, r: Response(body[10:], 206, {"Content-Range": f"bytes 10-99/100"})]
        self.run_download(variant, hub)
        self.assertEqual((self.dir / "m.gguf").read_bytes(), body)
        self.assertIsNone(hub.requests[1].get_header("Range"))

    def test_partial_larger_than_expected_is_discarded(self):
        body = b"z" * 20
        variant = make_variant({"m.gguf": body})
        (self.dir / "m.gguf.part").write_bytes(b"y" * 50)
        hub = Hub({"m.gguf": body})
        self.run_download(variant, hub)
        self.assertIsNone(hub.requests[0].get_header("Range"))
        self.assertEqual((self.dir / "m.gguf").read_bytes(), body)

    def test_corrupted_data_deletes_part(self):
        body = b"good" * 10
        variant = make_variant({"m.gguf": body})
        with self.assertRaisesRegex(ValueError, "SHA256.*deleted"):
            self.run_download(variant, Hub({"m.gguf": b"bad!" * 10}))
        self.assertFalse((self.dir / "m.gguf.part").exists())
        self.assertFalse((self.dir / "m.gguf").exists())

    def test_too_large_response_is_stopped(self):
        body = b"a" * 20
        variant = make_variant({"m.gguf": body})
        hub = Hub({"m.gguf": body})
        hub.script = [lambda h, r: Response(b"a" * 60, 200, {"Content-Length": ""})]
        with self.assertRaisesRegex(ValueError, "more data"):
            self.run_download(variant, hub)
        self.assertFalse((self.dir / "m.gguf.part").exists())
        hub.script = [lambda h, r: Response(b"a" * 60, 200)]  # declared too large: refused before writing
        with self.assertRaisesRegex(ValueError, "more data"):
            self.run_download(variant, hub)

    def test_different_total_size_in_content_range_is_refused(self):
        body = b"a" * 20
        variant = make_variant({"m.gguf": body})
        (self.dir / "m.gguf.part").write_bytes(body[:5])
        hub = Hub({"m.gguf": body})
        hub.script = [lambda h, r: Response(body[5:], 206, {"Content-Range": "bytes 5-19/999"})]
        with self.assertRaisesRegex(ValueError, "different size"):
            self.run_download(variant, hub)

    def test_transient_errors_retry_and_resume(self):
        body = bytes(range(200))
        variant = make_variant({"m.gguf": body})
        hub = Hub({"m.gguf": body})
        hub.script = [lambda h, r: h.reply(r, fail_after=70),
                      lambda h, r: (_ for _ in ()).throw(TimeoutError("timed out")),
                      lambda h, r: (_ for _ in ()).throw(HTTPError(r.full_url, 503, "busy", Message(), None))]
        self.run_download(variant, hub)
        self.assertEqual((self.dir / "m.gguf").read_bytes(), body)
        self.assertEqual(hub.requests[-1].get_header("Range"), "bytes=70-")

    def test_gives_up_after_repeated_failures_and_keeps_part(self):
        body = bytes(range(100))
        variant = make_variant({"m.gguf": body})
        hub = Hub({"m.gguf": body})
        hub.script = [lambda h, r: h.reply(r, fail_after=14)] + [lambda h, r: (_ for _ in ()).throw(ConnectionResetError())] * 10
        with self.assertRaisesRegex(ValueError, "network kept failing"):
            self.run_download(variant, hub)
        self.assertEqual((self.dir / "m.gguf.part").stat().st_size, 14)
        self.assertEqual(len(hub.requests), 1 + downloads.RETRIES)

    def test_server_ignoring_range_and_dropping_does_not_retry_forever(self):
        body = bytes(range(100))
        variant = make_variant({"m.gguf": body})
        hub = Hub({"m.gguf": body}, ignore_range=True)
        hub.script = [lambda h, r: h.reply(r, fail_after=14)] * 50
        with self.assertRaisesRegex(ValueError, "network kept failing"):
            self.run_download(variant, hub)
        # Re-downloading the same first bytes again and again is not progress.
        self.assertEqual(len(hub.requests), 1 + downloads.RETRIES)

    def test_shorter_206_range_keeps_part_and_asks_for_the_rest(self):
        body = bytes(range(100))
        variant = make_variant({"m.gguf": body})
        (self.dir / "m.gguf.part").write_bytes(body[:10])
        hub = Hub({"m.gguf": body})
        hub.script = [lambda h, r: Response(body[10:50], 206, {"Content-Range": "bytes 10-49/100"})]
        self.run_download(variant, hub)
        self.assertEqual((self.dir / "m.gguf").read_bytes(), body)
        self.assertEqual([r.get_header("Range") for r in hub.requests], ["bytes=10-", "bytes=50-"])

    def test_write_failing_at_close_never_installs_a_short_file(self):
        body = bytes(range(100))
        variant = make_variant({"m.gguf": body})

        class DiskFull:
            """Like a buffered file on a full disk: the last bytes are lost when it is closed."""
            def __init__(self, path, mode):
                self.real = open(path, mode)
            def write(self, chunk):
                return self.real.write(chunk)
            def close(self):
                self.real.truncate(max(0, self.real.tell() - 5))
                self.real.close()
                raise OSError(28, "No space left on device")

        with patch.object(downloads, "_open_partial", lambda path, mode, name: DiskFull(path, mode)):
            with self.assertRaisesRegex(ValueError, "Could not save m.gguf"):
                self.run_download(variant, Hub({"m.gguf": body}))
        self.assertFalse((self.dir / "m.gguf").exists())
        # The next run resumes from what really reached the disk.
        self.run_download(variant, Hub({"m.gguf": body}))
        self.assertEqual((self.dir / "m.gguf").read_bytes(), body)

    def test_cancel_is_not_hidden_by_a_failing_close(self):
        body = bytes(range(100))
        variant = make_variant({"m.gguf": body})
        cancel = threading.Event()

        class FailingClose:
            def __init__(self, path, mode):
                self.real = open(path, mode)
            def write(self, chunk):
                return self.real.write(chunk)
            def close(self):
                self.real.close()
                raise OSError(28, "No space left on device")

        hub = Hub({"m.gguf": body})
        hub.script = [lambda h, r: h.reply(r, on_read=lambda sent: sent >= 30 and cancel.set())]
        with patch.object(downloads, "_open_partial", lambda path, mode, name: FailingClose(path, mode)):
            with self.assertRaises(Cancelled):
                self.run_download(variant, hub, cancel=cancel)

    def test_unsafe_or_colliding_local_names_are_refused(self):
        for names in (["m.gguf:stream"], ["a.gguf", "a.gguf.part"], ["C:m.gguf"]):
            variant = make_variant({n: b"x" * (i + 1) for i, n in enumerate(names)})
            with self.assertRaisesRegex(ValueError, "Unsafe or duplicate"):
                downloads.plan(variant, self.dir)

    def test_access_denied_has_plain_message(self):
        body = b"a" * 10
        hub = Hub({})
        hub.script = [lambda h, r: (_ for _ in ()).throw(HTTPError(r.full_url, 401, "no", Message(), None))]
        with self.assertRaisesRegex(ValueError, "HF_TOKEN"):
            self.run_download(make_variant({"m.gguf": body}), hub)

    def test_cancel_mid_file_keeps_part_then_resumes(self):
        body = bytes(range(250))
        variant = make_variant({"m.gguf": body})
        cancel = threading.Event()
        hub = Hub({"m.gguf": body})
        hub.script = [lambda h, r: h.reply(r, on_read=lambda sent: sent >= 70 and cancel.set())]
        with self.assertRaises(Cancelled):
            self.run_download(variant, hub, cancel=cancel)
        kept = (self.dir / "m.gguf.part").stat().st_size
        self.assertTrue(0 < kept < len(body))
        self.assertEqual(downloads.plan(variant, self.dir)["files"][0]["partial_bytes"], kept)
        hub2 = Hub({"m.gguf": body})
        self.run_download(variant, hub2)
        self.assertEqual(hub2.requests[0].get_header("Range"), f"bytes={kept}-")
        self.assertEqual((self.dir / "m.gguf").read_bytes(), body)

    def test_speed_eta_and_throttling(self):
        body = bytes(range(100)) * 2
        variant = make_variant({"m.gguf": body})
        clock = [0.0]

        def tick(sent):
            clock[0] += 0.1  # each 7-byte chunk takes 0.1 s: 70 bytes/s

        events = []
        with patch.object(downloads, "_clock", lambda: clock[0]):
            hub = Hub({"m.gguf": body})
            hub.script = [lambda h, r: h.reply(r, on_read=tick)]
            self.run_download(variant, hub, progress=events.append)
        moving = [e for e in events if e["stage"] == "downloading"]
        self.assertLessEqual(len(moving), 30 * 0.1 / 0.2 + 2)  # about 3 s of chunks, at most 5 updates per second
        late = moving[-1]
        self.assertAlmostEqual(late["bytes_per_second"], 70, delta=3)
        self.assertAlmostEqual(late["eta_seconds"], (len(body) - late["done"]) / 70, delta=1)
        self.assertIsNone(moving[0]["bytes_per_second"])  # unknown until measured, never invented

    def test_insufficient_disk(self):
        body = b"a" * 10
        with patch.object(downloads.shutil, "disk_usage", return_value=type("U", (), {"free": GIB})()):
            with self.assertRaisesRegex(ValueError, "Not enough free disk space"):
                self.run_download(make_variant({"m.gguf": body}), Hub({"m.gguf": body}))

    def test_refuses_to_overwrite_different_file(self):
        body = b"a" * 10
        variant = make_variant({"m.gguf": body})
        (self.dir / "m.gguf").write_bytes(b"b" * 10)
        with self.assertRaisesRegex(ValueError, "different contents"):
            self.run_download(variant, Hub({"m.gguf": body}))
        (self.dir / "m.gguf").write_bytes(b"b" * 3)
        with self.assertRaisesRegex(ValueError, "different size"):
            self.run_download(variant, Hub({"m.gguf": body}))
        self.assertEqual((self.dir / "m.gguf").read_bytes(), b"b" * 3)

    def test_requires_hashes_and_real_model(self):
        with self.assertRaisesRegex(ValueError, "SHA256"):
            downloads.download_variant(demo_variants()[0], self.dir)

    def test_plan_reports_presence_partials_and_space(self):
        shards = {"m-00001-of-00002.gguf": b"a" * 30, "m-00002-of-00002.gguf": b"b" * 20}
        variant = make_variant(shards)
        (self.dir / "m-00001-of-00002.gguf").write_bytes(shards["m-00001-of-00002.gguf"])
        (self.dir / "m-00002-of-00002.gguf.part").write_bytes(b"b" * 5)
        result = downloads.plan(variant, self.dir)
        self.assertEqual(result["files"], [
            {"filename": "m-00001-of-00002.gguf", "size_bytes": 30, "present": True, "partial_bytes": 0},
            {"filename": "m-00002-of-00002.gguf", "size_bytes": 20, "present": False, "partial_bytes": 5}])
        self.assertEqual(result["total_bytes"], 50)
        self.assertEqual(result["remaining_bytes"], 15)
        self.assertTrue(result["enough_space"])
        with patch.object(downloads.shutil, "disk_usage", return_value=type("U", (), {"free": GIB + 14})()):
            self.assertFalse(downloads.plan(variant, self.dir / "not" / "yet")["enough_space"])

    def test_remove_variant_deletes_only_matching_files(self):
        shards = {"m-00001-of-00002.gguf": b"a" * 30, "m-00002-of-00002.gguf": b"b" * 20}
        variant = make_variant(shards)
        (self.dir / "m-00001-of-00002.gguf").write_bytes(b"a" * 30)
        (self.dir / "m-00002-of-00002.gguf").write_bytes(b"x" * 99)  # same name, different file: kept
        (self.dir / "m-00002-of-00002.gguf.part").write_bytes(b"b" * 4)
        (self.dir / "other.gguf").write_bytes(b"o" * 30)
        outside = self.dir.parent / (self.dir.name + "-outside.gguf")
        outside.write_bytes(b"a" * 30)
        self.addCleanup(outside.unlink)
        sub = self.dir / "sub"
        sub.mkdir()
        (sub / "m-00001-of-00002.gguf").symlink_to(outside)
        self.assertEqual(downloads.remove_variant(variant, self.dir), 34)
        self.assertEqual(sorted(p.name for p in self.dir.iterdir()), ["m-00002-of-00002.gguf", "other.gguf", "sub"])
        link = self.dir / "m-00001-of-00002.gguf"
        link.symlink_to(outside)
        self.assertEqual(downloads.remove_variant(variant, self.dir), 0)
        self.assertTrue(outside.exists() and link.is_symlink())

    def test_runtime_download_delegates(self):
        body = b"q" * 33
        variant = make_variant({"m.gguf": body})
        with patch.object(downloads, "_open", Hub({"m.gguf": body})):
            self.assertEqual(runtime.download(variant, self.dir), self.dir / "m.gguf")
        self.assertIs(runtime.DownloadRedirect, downloads.DownloadRedirect)


class FakeHTTPS(urllib.request.HTTPSHandler):
    """Serves https:// requests from a table so the real urllib redirect machinery runs."""

    def __init__(self, routes):
        super().__init__()
        self.routes, self.seen = routes, []

    def https_open(self, req):
        self.seen.append((req.full_url, req.get_header("Authorization"), req.get_header("Range")))
        status, headers, body = self.routes[req.full_url]
        message = Message()
        for key, value in headers.items():
            message[key] = value
        response = addinfourl(io.BytesIO(body), message, req.full_url, status)
        response.msg = "OK"
        return response


class RedirectTests(unittest.TestCase):
    def open_with(self, routes, url, headers):
        fake = FakeHTTPS(routes)
        real = urllib.request.build_opener
        with patch.object(downloads, "build_opener", lambda *handlers: real(*handlers, fake)):
            with downloads._open(urllib.request.Request(url, headers=headers)) as response:
                return fake.seen, response.read(), response.status

    def test_cross_host_redirect_strips_auth_keeps_range(self):
        routes = {"https://huggingface.co/o/r/resolve/main/m.gguf": (302, {"Location": "https://cdn.example/blob"}, b""),
                  "https://cdn.example/blob": (206, {"Content-Range": "bytes 3-5/6"}, b"def")}
        seen, body, status = self.open_with(routes, "https://huggingface.co/o/r/resolve/main/m.gguf",
                                            {"Authorization": "Bearer secret", "Range": "bytes=3-"})
        self.assertEqual(seen, [("https://huggingface.co/o/r/resolve/main/m.gguf", "Bearer secret", "bytes=3-"),
                                ("https://cdn.example/blob", None, "bytes=3-")])
        self.assertEqual((body, status), (b"def", 206))

    def test_same_host_redirect_keeps_auth(self):
        routes = {"https://huggingface.co/a": (307, {"Location": "/b"}, b""), "https://huggingface.co/b": (200, {}, b"ok")}
        seen, body, _ = self.open_with(routes, "https://huggingface.co/a", {"Authorization": "Bearer secret"})
        self.assertEqual(seen[1], ("https://huggingface.co/b", "Bearer secret", None))

    def test_http_redirect_refused(self):
        routes = {"https://huggingface.co/a": (302, {"Location": "http://cdn.example/blob"}, b"")}
        with self.assertRaisesRegex(ValueError, "HTTPS"):
            self.open_with(routes, "https://huggingface.co/a", {})

    def test_http_redirect_during_download_keeps_part(self):
        body = b"a" * 30
        variant = make_variant({"m.gguf": body})
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "m.gguf.part").write_bytes(b"a" * 5)
            url = "https://huggingface.co/org/model-GGUF/resolve/main/m.gguf"
            fake = FakeHTTPS({url: (302, {"Location": "http://evil.example/x"}, b"")})
            real = urllib.request.build_opener
            with patch.object(downloads, "build_opener", lambda *h: real(*h, fake)), patch.dict(os.environ, {"HF_ENDPOINT": ""}):
                with self.assertRaisesRegex(ValueError, "HTTPS"):
                    downloads.download_variant(variant, tmp)
            self.assertEqual(Path(tmp, "m.gguf.part").stat().st_size, 5)


if __name__ == "__main__":
    unittest.main()


class BenchDeviceTests(unittest.TestCase):
    """runtime.bench picks the `-dev` name the build itself lists, per backend, and never guesses one."""

    def setUp(self):
        from dataclasses import replace
        self.model = replace(demo_variants()[0], demo=False, sha256="correct")
        self.calls = []

    def hw(self, *gpus):
        return {"fingerprint": "machine", "ram_available": 64 * GIB, "cores": 8, "threads": 16, "gpus": list(gpus)}

    def run_bench(self, hardware, devices, layers=4, executable="llama-bench", build_number=11158):
        row = {"n_prompt": 0, "n_gen": 128, "n_depth": 4096 - 128, "avg_ts": 30.0, "n_gpu_layers": layers, "n_threads": 8,
               "type_k": "f16", "type_v": "f16", "build_commit": "d2e5458", "build_number": build_number, "n_batch": 2048,
               "n_ubatch": 512}

        def fake_run(args, **kwargs):
            self.calls.append((args, kwargs.get("env") or {}))
            return type("Done", (), {"returncode": 0, "stdout": json.dumps([row]), "stderr": ""})()

        listed = []

        def fake_list(command, env=None):
            listed.append(dict(env or {}))
            return devices

        with patch("llm_configurator.runtime.digest", return_value="correct"), \
                patch("llm_configurator.runtime.shutil.which", return_value="/bin/llama-bench"), \
                patch("llm_configurator.runtime.scan", return_value=hardware), \
                patch("llm_configurator.runtime.list_devices", side_effect=fake_list), \
                patch("llm_configurator.runtime.subprocess.run", side_effect=fake_run):
            result = runtime.bench(self.model, "file.gguf", executable, 4096, layers)
        self.listed = listed
        return result

    def args(self):
        return self.calls[-1][0]

    def dev(self):
        args = self.args()
        return args[args.index("-dev") + 1] if "-dev" in args else None

    @staticmethod
    def device(name, description, backend):
        return {"name": name, "description": description, "backend": backend}

    def test_nvidia_is_pinned_by_uuid_and_uses_its_cuda_name(self):
        gpu = {"index": 0, "uuid": "GPU-abc", "name": "NVIDIA GeForce RTX 4090", "backend": "cuda", "available": 24 * GIB}
        result = self.run_bench(self.hw(gpu), [self.device("CUDA0", "NVIDIA GeForce RTX 4090", "cuda")])
        self.assertEqual(self.dev(), "CUDA0")
        self.assertEqual(self.calls[-1][1]["CUDA_VISIBLE_DEVICES"], "GPU-abc")
        self.assertEqual(self.listed[0]["CUDA_VISIBLE_DEVICES"], "GPU-abc")   # the list is taken with the pin applied
        self.assertEqual(result["runtime"], {"version": "b11158", "backend": "cuda"})
        self.assertEqual((result["kind"], result["depth"], result["runtime_build"]), ("bench", 4096 - 128, "b11158"))
        self.assertRegex(result["id"], r"^[0-9a-f]{12}$")

    def test_amd_on_vulkan_is_found_by_name_among_several(self):
        amd = {"index": 0, "uuid": "amdgpu-0000:03:00.0", "name": "AMD Radeon RX 7900 XTX", "backend": "vulkan",
               "vendor": "amd", "available": 20 * GIB}
        nvidia = {"index": 1, "uuid": "GPU-x", "name": "NVIDIA GeForce RTX 3060", "backend": "cuda", "available": 12 * GIB}
        devices = [self.device("Vulkan0", "NVIDIA GeForce RTX 3060", "vulkan"),
                   self.device("Vulkan1", "AMD Radeon RX 7900 XTX (RADV NAVI31)", "vulkan")]
        self.run_bench(self.hw(amd, nvidia), devices)
        self.assertEqual(self.dev(), "Vulkan1")
        self.assertEqual(self.calls[-1][1].get("CUDA_VISIBLE_DEVICES"), os.environ.get("CUDA_VISIBLE_DEVICES"))

    def test_rocm_and_metal_names(self):
        rocm = {"index": 0, "uuid": "amdgpu-1", "name": "AMD Radeon PRO W7900", "backend": "rocm", "available": 40 * GIB}
        self.run_bench(self.hw(rocm), [self.device("ROCm0", "AMD Radeon PRO W7900", "rocm")])
        self.assertEqual(self.dev(), "ROCm0")
        apple = {"index": 0, "uuid": "apple-m3-max", "name": "Apple M3 Max", "backend": "metal", "unified": True,
                 "available": 40 * GIB}
        self.run_bench(self.hw(apple), [self.device("MTL0", "Apple M3 Max", "metal")])
        self.assertEqual(self.dev(), "MTL0")
        self.assertEqual(self.calls[-1][1].get("CUDA_VISIBLE_DEVICES"), os.environ.get("CUDA_VISIBLE_DEVICES"))

    def test_other_backend_card_is_never_taken_without_a_name_match(self):
        # Intel iGPU picked on a laptop whose CUDA build only lists the NVIDIA card.
        igpu = {"index": 0, "uuid": "intel-1", "name": "Intel(R) Iris(R) Xe Graphics", "backend": "vulkan", "available": 2 * GIB}
        with self.assertRaisesRegex(ValueError, "cannot tell which"):
            self.run_bench(self.hw(igpu), [self.device("CUDA0", "NVIDIA GeForce RTX 4070 Laptop GPU", "cuda")], layers=1)
        # Whole words only: an A100 is not an A10.
        a100 = {"index": 0, "uuid": "GPU-a", "name": "NVIDIA A100-PCIE-40GB", "backend": "cuda", "available": 40 * GIB}
        with self.assertRaisesRegex(ValueError, "cannot tell which"):
            self.run_bench(self.hw(a100), [self.device("Vulkan0", "NVIDIA A10", "vulkan"),
                                           self.device("Vulkan1", "-", "vulkan")], layers=1)

    def test_nvidia_card_with_vulkan_build(self):
        gpu = {"index": 0, "uuid": "GPU-abc", "name": "NVIDIA GeForce RTX 4090", "backend": "cuda", "available": 24 * GIB}
        self.run_bench(self.hw(gpu), [self.device("Vulkan0", "NVIDIA GeForce RTX 4090", "vulkan")])
        self.assertEqual(self.dev(), "Vulkan0")

    def test_cpu_only_build_is_refused_before_running(self):
        gpu = {"index": 0, "uuid": "GPU-abc", "name": "NVIDIA GeForce RTX 4090", "backend": "cuda", "available": 24 * GIB}
        with self.assertRaisesRegex(ValueError, "sees no graphics card"):
            self.run_bench(self.hw(gpu), [])
        self.assertEqual(self.calls, [])

    def test_ambiguous_devices_are_refused_not_guessed(self):
        gpu = {"index": 0, "uuid": "amdgpu-1", "name": "AMD Radeon RX 7900 XTX", "backend": "vulkan", "available": 20 * GIB}
        devices = [self.device("Vulkan0", "AMD Radeon RX 7900 XTX (RADV NAVI31)", "vulkan"),
                   self.device("Vulkan1", "AMD Radeon RX 7900 XTX (RADV NAVI31)", "vulkan")]
        with self.assertRaisesRegex(ValueError, "cannot tell which"):
            self.run_bench(self.hw(gpu), devices)
        self.assertEqual(self.calls, [])

    def test_build_without_device_list_leaves_dev_out(self):
        gpu = {"index": 0, "uuid": "GPU-abc", "name": "NVIDIA GeForce RTX 4090", "available": 24 * GIB}  # pre-v0.4 scan
        self.run_bench(self.hw(gpu), None)
        self.assertIsNone(self.dev())
        self.assertEqual(self.calls[-1][1]["CUDA_VISIBLE_DEVICES"], "GPU-abc")

    def test_cpu_run_uses_none_and_accepts_an_argv_prefix(self):
        result = self.run_bench(self.hw(), None, layers=0, executable=["python3", "fake_llama.py", "--as", "bench"])
        self.assertEqual(self.args()[:4], ["python3", "fake_llama.py", "--as", "bench"])
        self.assertEqual(self.dev(), "none")
        self.assertEqual(result["runtime"]["backend"], "cpu")
        self.assertEqual(self.listed, [])

    def test_missing_gpu_says_gpu_not_nvidia(self):
        with self.assertRaisesRegex(ValueError, "^The selected GPU is unavailable$"):
            self.run_bench(self.hw(), None)
