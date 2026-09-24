"""v0.4 HTTP API: auth, validation, demo refusals, job lifecycle and path privacy.

The modules other workstreams build (downloads, testing, tuner, ...) are replaced by small
fakes injected into sys.modules, so these tests pin down only the api wiring.
"""
from contextlib import ExitStack
import json
from pathlib import Path, PureWindowsPath
import re
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import patch

import llm_configurator
from llm_configurator import app
from llm_configurator.catalogue import demo_variants
from llm_configurator.domain import Cancelled, Variant, check_cancel
from llm_configurator.server import STATIC, make_server, scrub
from llm_configurator.storage import Store

SECRET = "/home/secret-user/private"


def real_variant(index=0, quant=None):
    record = demo_variants()[index].to_dict()
    record.update(demo=False, sha256="ab" * 32, id=f"org/repo@rev:{record['filename']}", repo="org/repo")
    if quant:
        record.update(quant=quant, filename=f"model-{quant}.gguf", id=f"org/repo@rev:model-{quant}.gguf")
    return Variant(**record)


def candidate(variant, gpu_layers=0, context=4096):
    return {"id": f"{variant.id}|now|{context}|{gpu_layers}", "variant_id": variant.id, "name": variant.name,
            "quant": variant.quant, "context": context, "users": 1, "gpu_layers": gpu_layers,
            "total_layers": variant.layers, "gpu_index": None, "threads": 4, "mode": "cpu", "scenario": "now",
            "demo": False}


class Fakes:
    """Stand-ins for the parallel workstreams' modules, recording their calls."""

    def __init__(self, store):
        self.store, self.calls, self.local = store, [], {}
        self.release = threading.Event()
        self.block_download = False
        self.modules = {name: types.ModuleType(f"llm_configurator.{name}") for name in
                        ["runtime_install", "downloads", "discover", "testing", "tuner", "evals", "quantcheck",
                         "export", "community", "llama_server", "gguf"]}
        m = self.modules
        m["runtime_install"].detect = lambda store: {"installed": True, "source": "managed", "directory": f"{SECRET}/runtime/b1-cpu",
                                                     "version": "b1", "build": 1, "backend": "cpu",
                                                     "binaries": {"llama-server": [f"{SECRET}/runtime/llama-server"]}, "warnings": []}
        m["runtime_install"].binary = lambda store, name: [f"{SECRET}/runtime/{name}"]
        m["runtime_install"].install = self.install
        m["runtime_install"].install_archive = lambda store, path, progress=None, cancel=None: m["runtime_install"].detect(store)
        m["runtime_install"].use_directory = lambda store, directory: m["runtime_install"].detect(store)
        m["downloads"].plan = lambda variant, directory: {
            "files": [{"filename": f["filename"], "size_bytes": f["size_bytes"], "present": variant.id in self.local, "partial_bytes": 0}
                      for f in variant.all_files()],
            "total_bytes": variant.size_bytes, "remaining_bytes": 0 if variant.id in self.local else variant.size_bytes,
            "disk_free": 10**12, "enough_space": True}
        m["downloads"].download_variant = self.download
        m["downloads"].remove_variant = lambda variant, directory: self.calls.append(("remove", variant.id, directory)) or 123
        m["discover"].find_for_variant = lambda store, variant, verify=True: self.local.get(variant.id)
        m["discover"].locations = lambda store=None, extra_dirs=(): [{"source": "hf_cache", "path": str(Path.home() / ".cache/huggingface/hub"), "exists": True},
                                           {"source": "custom", "path": f"{SECRET}/elsewhere", "exists": False}]
        m["discover"].scan = self.scan
        m["discover"].hash_cached = lambda store, path, progress=None, cancel=None: "cd" * 32
        m["gguf"].variant_from_file = lambda path, sha256=None: Variant(**{**real_variant().to_dict(), "id": f"local:{sha256[:12]}",
                                                                            "source": "local", "sha256": sha256})
        m["testing"].run_tests = self.run_tests
        m["tuner"].tune = self.tune
        m["llama_server"].LlamaServer = self.server_class()
        m["llama_server"].ServerRegistry = self.registry_class()
        m["evals"].run_quiz = lambda chat, workload, limit=None, progress=None, cancel=None: {
            "workload": workload, "correct": 3, "total": 4, "score": 0.75, "ci_low": 0.3, "ci_high": 0.95,
            "items": [{"id": "g1", "ok": True, "expected": "4", "got": chat([{"role": "user", "content": "2+2"}], 16)["text"]}],
            "seconds": 1.0, "note": "Small sample"}
        m["evals"].needle_test = lambda chat, tokenize, context_tokens, progress=None, cancel=None: {"context_tokens": context_tokens, "found": 3, "total": 3}
        m["evals"].run_comparison = self.run_comparison
        m["evals"].blinded = lambda store, cid: {"items": [{"prompt": "hi", "outputs": [{"slot": "A", "text": "x"}, {"slot": "B", "text": "y"}]}]}
        m["evals"].vote = lambda store, cid, item, slot: {"votes": {slot: 1}, "item": item}
        m["evals"].reveal = lambda store, cid: {"mapping": {"A": "one"}, "tallies": {"one": 1}}
        m["quantcheck"].kl_check = self.kl_check
        m["export"].formats = lambda: [{"id": "llama-server", "label": "Script", "description": "d"}, {"id": "ollama", "label": "Ollama", "description": "d"}]
        m["export"].export = lambda config, variant, fmt, platform="posix", server_command=None: {
            "format": fmt, "filename": "run.sh", "content": f"{server_command} -m {config['model_path']} -t {config['threads']}", "instructions": [], "notes": []}
        m["community"].import_records = lambda store, source=None, text=None, progress=None, cancel=None: (
            store.put("community", {"source": source or "default", "fetched_at": "2026-09", "records": [{"tps": 1}]})
            or {"source": source or "default", "fetched_at": "2026-09", "records": [{"tps": 1}], "rejected": 2})
        m["community"].share_payload = lambda store, ids: {"json": json.dumps(ids), "issue_url": "https://github.com/Eyad-3D/LLM-Configurator/issues/new"}

    def install(self, store, hardware, progress=None, cancel=None, release=None, allow_unverified=False):
        self.calls.append(("install", allow_unverified))
        progress({"stage": "download", "done": 5, "total": 10, "message": f"Saving to {SECRET}/runtime"})
        return self.modules["runtime_install"].detect(store)

    def download(self, variant, directory, progress=None, cancel=None, token=None):
        self.calls.append(("download", variant.id, str(directory)))
        for done in range(0, 101, 25):
            check_cancel(cancel)
            if progress:
                progress({"stage": "download", "done": done * variant.size_bytes // 100, "total": variant.size_bytes,
                          "message": "", "bytes_per_second": 50 * 1024**2, "eta_seconds": 3})
            if self.block_download:
                while not self.release.wait(0.02):
                    check_cancel(cancel)
        path = Path(directory) / Path(variant.filename).name
        self.local[variant.id] = path
        return path

    def scan(self, store, extra_dirs=(), progress=None, cancel=None):
        files = [{"path": f"{SECRET}/models/found.gguf", "size_bytes": 5, "sha256": None, "source": "models_dir",
                  "variant_id": None, "gguf": {"layers": 2}, "mtime": 1.0, "verified": False}]
        store.put("local_files", files)
        return files

    def run_tests(self, store, variant, config, commands, kind="full", hardware=None, progress=None, cancel=None):
        self.calls.append(("test", variant.id, config, commands, kind))
        progress({"stage": "smoke", "done": 1, "total": 2, "message": f"Loading {config['model_path']}"})
        store.append("measurements", {"id": "a1b2c3", "variant_id": variant.id, "kind": "speed_test", "tps": 12.0})
        return {"smoke": {"ok": True, "log_tail": f"llama_model_load: loading {config['model_path']}"},
                "speed": None, "verdict": "works", "verdict_text": "It works."}

    def tune(self, bench_command, variant, base_config, hardware, budget_seconds=300, goal="generation",
             memory_check=None, progress=None, cancel=None):
        self.calls.append(("tune", bench_command, base_config, budget_seconds, goal))
        return {"best": {**base_config, "threads": 8, "flash_attn": "on"}, "baseline": {"tps": 10.0, "pp_tps": 100.0},
                "best_result": {"tps": 12.0, "pp_tps": 110.0}, "improvement": 1.2,
                "trials": [{"changes": {"threads": 8}, "tps": 12.0, "pp_tps": 110.0, "seconds": 1, "status": "ok"}],
                "stopped": "converged", "notes": ["Eight threads were fastest."]}

    def server_class(self):
        fakes = self

        class LlamaServer:
            live = 0
            max_live = 0

            def __init__(self, command, config, log_path=None):
                self.command, self.config, self.base_url, self.pid = command, config, "http://127.0.0.1:9999", 42
                fakes.calls.append(("server", command, config))

            def start(self, timeout=300, progress=None, cancel=None):
                LlamaServer.live += 1
                LlamaServer.max_live = max(LlamaServer.max_live, LlamaServer.live)
                if progress:
                    progress({"stage": "load", "done": 1, "total": 1, "message": "ready"})

            def stop(self, timeout=10):
                if self.pid:
                    LlamaServer.live -= 1
                    self.pid = None

            def chat(self, messages, max_tokens=256, **options):
                return {"text": "4", "finish_reason": "stop", "timings": {}, "usage": {}}

            def tokenize(self, text):
                return list(range(len(text.split())))

            def ready(self):
                return fakes.release.is_set() is False

            def failure_reason(self):
                return "The model ran out of memory."
        return LlamaServer

    def registry_class(self):
        fakes = self

        class ServerRegistry:
            def __init__(self, log_dir=None):
                self.state = {"running": False, "base_url": None, "openai_base_url": None, "pid": None, "config": None,
                              "started_at": None, "model": None, "log_tail": ""}

            def start(self, command, config, progress=None, cancel=None):
                fakes.calls.append(("serve", command, config))
                self.state = {**self.state, "running": True, "base_url": "http://127.0.0.1:8081",
                              "openai_base_url": "http://127.0.0.1:8081/v1", "pid": 7, "config": config,
                              "model": Path(config["model_path"]).name, "log_tail": f"main: loaded {config['model_path']}"}
                return self.status()

            def stop(self):
                fakes.calls.append(("serve_stop",))
                self.state = {**self.state, "running": False, "pid": None}
                return self.status()

            def status(self):
                return dict(self.state)
        return ServerRegistry

    def run_comparison(self, store, prompts, runners, max_tokens=512, progress=None, cancel=None):
        self.calls.append(("compare", sorted(runners), prompts))
        for label, factory in runners.items():
            with factory() as chat:
                chat([{"role": "user", "content": prompts[0]}], max_tokens)
        return "cmp_1"

    def kl_check(self, perplexity_command, reference_path, candidates, corpus_path=None, context=512, chunks=None,
                 config=None, progress=None, cancel=None):
        self.calls.append(("kl", perplexity_command, reference_path, candidates))
        return {"reference": reference_path, "results": {label: {"mean_kld": 0.01, "plain": "Nearly identical."} for label in candidates},
                "corpus": "built-in", "notes": [f"Temporary files in {SECRET}/tmp were deleted."]}

    def installed(self):
        stack = ExitStack()
        stack.enter_context(patch.dict(sys.modules, {f"llm_configurator.{k}": v for k, v in self.modules.items()}))
        for name, module in self.modules.items():
            stack.enter_context(patch.object(llm_configurator, name, module, create=True))
        return stack


class ApiCase(unittest.TestCase):
    demo = False

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(self.temp.name)
        self.variant = real_variant()
        self.other = real_variant(0, "Q8_0")
        self.store.put("variants", [self.variant.to_dict(), self.other.to_dict()])
        self.fakes = Fakes(self.store)
        self.stack = self.fakes.installed()
        self.stack.enter_context(patch("llm_configurator.app.scan", return_value={"fingerprint": "fp", "gpus": [], "cores": 4}))
        self.server = make_server(self.store, port=0, demo=self.demo)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        with urlopen(self.url) as response:
            self.token = re.search(r'name="session-token" content="([^"]+)"', response.read().decode())[1]

    def tearDown(self):
        self.fakes.release.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.stack.close()
        self.temp.cleanup()

    def call(self, path, body=None, headers=None, token=True):
        request_headers = {"X-Session-Token": self.token} if token else {}
        request_headers.update(headers or {})
        data = json.dumps(body).encode() if body is not None else None
        try:
            with urlopen(Request(self.url + path, data=data, headers=request_headers)) as response:
                raw = response.read().decode()
                return response.status, json.loads(raw) if raw[:1] in "{[" else raw, raw
        except HTTPError as error:
            raw = error.read().decode()
            return error.code, json.loads(raw) if raw.startswith("{") else raw, raw

    def remember(self, *candidates):
        report = {"candidates": list(candidates), "hardware": {"fingerprint": "fp", "gpus": [], "cores": 4},
                  "requirements": {"workload": "coding"}, "shortlist": [], "notes": [], "demo": self.demo}
        with patch("llm_configurator.server.evaluate", return_value=report):
            status, result, _ = self.call("/api/recommend", {"context": 4096})
        self.assertEqual(status, 200)
        return result

    def wait(self, job_id, states=("done", "failed", "cancelled")):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            status, job, raw = self.call(f"/api/jobs/{job_id}")
            self.assertEqual(status, 200)
            self.assertNotIn(SECRET, raw)
            if job["state"] in states:
                return job
            time.sleep(0.02)
        self.fail(f"Job {job_id} did not finish")

    def downloaded(self, variant=None):
        variant = variant or self.variant
        self.fakes.local[variant.id] = Path(self.temp.name) / "models" / Path(variant.filename).name


class ScrubTests(unittest.TestCase):
    def test_paths_become_file_names_and_other_text_stays(self):
        self.assertEqual(scrub("loading /home/me/models/x.gguf now"), "loading x.gguf now")
        self.assertEqual(scrub(r"C:\Users\me\m.gguf"), "m.gguf")
        self.assertEqual(scrub({"a": [f"{SECRET}/llama-server"]}), {"a": ["llama-server"]})
        self.assertEqual(scrub(r"\\server\share\m.gguf"), "m.gguf")
        self.assertEqual(scrub(r"C:\Users\Eyad Abu\m\x.gguf", roots=(r"C:\Users\Eyad Abu",)), "x.gguf")
        self.assertEqual(scrub("/home/A B/models/x.gguf and ~/y/z.gguf", roots=("/home/A B",)), "x.gguf and z.gguf")
        self.assertEqual(scrub({"got": "cd /usr/local/bin", "log": "cd /usr/local/bin"}, keep={"got"}),
                         {"got": "cd /usr/local/bin", "log": "cd bin"})
        self.assertEqual(scrub({"n": float("nan"), "p": Path("/a/b/c.gguf"), "s": {1}}), {"n": None, "p": "c.gguf", "s": [1]})
        for kept in ["https://huggingface.co/Qwen/x", "Q4_K_M/model.gguf", "org/repo@rev:file.gguf", "12 MB/s",
                     "/api/jobs/job-1", "http://127.0.0.1:8081/v1", "2026/09/24", "and/or"]:
            self.assertEqual(scrub(kept), kept)


class StaticAndAuthTests(ApiCase):
    def test_static_allowlist(self):
        for path in ["/run.js", "/jobs.js", "/quality.js", "/quality.css", "/app.js", "/style.css", "/"]:
            self.assertIn(path, STATIC)
        self.assertTrue(STATIC["/quality.css"][1].startswith("text/css"))
        self.assertTrue(STATIC["/run.js"][1].startswith("text/javascript"))
        for path in ["/cache.sqlite3", "/../server.py", "/static/app.js", "/app.py"]:
            status, _, _ = self.call(path, token=False)
            self.assertIn(status, {403, 404})
        for name in ["run.js", "jobs.js", "quality.js", "quality.css"]:
            exists = (Path(llm_configurator.__file__).with_name("static") / name).exists()
            status, _, raw = self.call("/" + name, token=False)
            self.assertEqual(status, 200 if exists else 404)
            if exists:
                self.assertNotIn("__SESSION_TOKEN__", raw)

    def test_every_new_endpoint_needs_the_session(self):
        for path in ["/api/jobs", "/api/jobs/job-1", "/api/runtime", "/api/local-models", "/api/quality/results",
                     "/api/quality/compare/cmp_1", "/api/serve", "/api/community"]:
            self.assertEqual(self.call(path, token=False)[0], 403, path)
        for path in ["/api/runtime/install", "/api/downloads", "/api/test", "/api/serve/start", "/api/serve/stop",
                     "/api/jobs/job-1/cancel", "/api/community/share"]:
            self.assertEqual(self.call(path, {}, token=False)[0], 403, path)
            self.assertEqual(self.call(path, {}, headers={"Origin": "https://evil.example"})[0], 403, path)
        self.assertEqual(self.call("/api/jobs", headers={"Host": "evil.example"})[0], 403)

    def test_body_limits_and_unknown_routes(self):
        big = {"variant_id": "x" * 70000}
        self.assertEqual(self.call("/api/downloads", big)[0], 400)
        self.assertEqual(self.call("/api/nothing", {})[0], 404)
        self.assertEqual(self.call("/api/nothing")[0], 404)
        status, _, raw = self.call("/api/jobs", headers={})
        self.assertEqual(status, 200)


class ValidationTests(ApiCase):
    def test_invalid_bodies_are_rejected(self):
        c = candidate(self.variant)
        self.remember(c)
        bad = [("/api/downloads", {"variant_id": self.variant.id, "path": "/etc"}),
               ("/api/downloads", {}),
               ("/api/downloads", {"variant_id": 5}),
               ("/api/downloads/plan", {"variant_id": "a\nb"}),
               ("/api/test", {"candidate_id": c["id"], "kind": "everything"}),
               ("/api/test", {"candidate_id": c["id"], "tuned": "yes"}),
               ("/api/test", {"candidate_id": c["id"], "executable": "/bin/sh"}),
               ("/api/tune", {"candidate_id": c["id"], "budget_seconds": 59}),
               ("/api/tune", {"candidate_id": c["id"], "budget_seconds": 1801}),
               ("/api/tune", {"candidate_id": c["id"], "budget_seconds": 60.5}),
               ("/api/tune", {"candidate_id": c["id"], "budget_seconds": 120, "goal": "max"}),
               ("/api/quality/quiz", {"candidate_id": c["id"], "workload": "poetry"}),
               ("/api/quality/quiz", {"candidate_id": c["id"], "include_needle": 1}),
               ("/api/quality/compare", {"candidate_ids": [c["id"]], "prompts": ["hi"]}),
               ("/api/quality/compare", {"candidate_ids": [c["id"], c["id"]], "prompts": ["hi"]}),
               ("/api/quality/compare", {"candidate_ids": ["a", "b"], "prompts": []}),
               ("/api/quality/compare", {"candidate_ids": ["a", "b"], "prompts": ["x" * 4001]}),
               ("/api/quality/compare", {"candidate_ids": ["a", "b"], "prompts": ["a"] * 6}),
               ("/api/quality/vote", {"comparison_id": "cmp_1", "item": 0, "slot": "D"}),
               ("/api/quality/vote", {"comparison_id": "../x", "item": 0, "slot": "A"}),
               ("/api/quality/vote", {"comparison_id": "cmp_1", "item": -1, "slot": "A"}),
               ("/api/quality/reveal", {"comparison_id": "a b"}),
               ("/api/quality/quant-check", {"reference_variant_id": self.variant.id, "variant_ids": []}),
               ("/api/quality/quant-check", {"reference_variant_id": self.variant.id, "variant_ids": [self.variant.id]}),
               ("/api/export", {"candidate_id": c["id"], "format": "bash; rm -rf"}),
               ("/api/export", {"candidate_id": c["id"], "format": "ollama", "platform": "amiga"}),
               ("/api/serve/start", {"candidate_id": c["id"], "port": 80}),
               ("/api/serve/stop", {"force": True}),
               ("/api/runtime/install", {"allow_unverified": True}),
               ("/api/community/share", {"measurement_ids": ["not hex!"]}),
               ("/api/community/share", {"measurement_ids": []}),
               ("/api/community/import", {"source": "https://evil.example/x.json"}),
               ("/api/local-models/scan", {"dirs": ["/"]}),
               ("/api/jobs/job-1/cancel", {"x": 1})]
        for path, body in bad:
            status, result, _ = self.call(path, body)
            self.assertIn(status, {400, 404}, (path, body, result))
            self.assertIn("error", result)
        self.assertEqual(self.fakes.calls, [])
        self.assertEqual(self.server.jobs.list(), [])

    def test_unknown_ids(self):
        self.assertEqual(self.call("/api/jobs/job-999")[0], 404)
        self.assertEqual(self.call("/api/jobs/../../etc")[0], 404)
        self.assertEqual(self.call("/api/jobs/job-x/cancel", {})[0], 404)
        self.assertEqual(self.call("/api/downloads", {"variant_id": "nope"})[0], 404)
        self.assertEqual(self.call("/api/quality/compare/..%2Fsecret")[0], 404)

    def test_candidate_must_come_from_latest_comparison(self):
        status, result, _ = self.call("/api/test", {"candidate_id": "anything"})
        self.assertEqual((status, result["error"].startswith("Compare again first")), (409, True))
        old = candidate(self.variant, context=4096)
        self.remember(old)
        self.remember(candidate(self.variant, context=8192))
        self.assertEqual(self.call("/api/test", {"candidate_id": old["id"]})[0], 409)
        # A candidate whose model disappeared from the catalogue is stale too.
        ghost = {**candidate(self.variant), "id": "ghost", "variant_id": "gone"}
        self.remember(ghost)
        self.assertEqual(self.call("/api/export", {"candidate_id": "ghost", "format": "ollama"})[0], 409)


class DemoTests(ApiCase):
    demo = True

    def test_demo_mode_never_downloads_tests_or_serves(self):
        variant = demo_variants()[0]
        c = {**candidate(variant), "demo": True}
        self.remember(c)
        for path, body in [("/api/runtime/install", {}), ("/api/downloads/plan", {"variant_id": variant.id}),
                           ("/api/downloads", {"variant_id": variant.id}), ("/api/downloads/remove", {"variant_id": variant.id}),
                           ("/api/test", {"candidate_id": c["id"]}), ("/api/tune", {"candidate_id": c["id"], "budget_seconds": 60}),
                           ("/api/quality/quiz", {"candidate_id": c["id"]}),
                           ("/api/quality/compare", {"candidate_ids": [c["id"], "b"], "prompts": ["hi"]}),
                           ("/api/quality/quant-check", {"reference_variant_id": variant.id, "variant_ids": [demo_variants()[1].id]}),
                           ("/api/export", {"candidate_id": c["id"], "format": "ollama"}),
                           ("/api/serve/start", {"candidate_id": c["id"]})]:
            status, result, _ = self.call(path, body)
            self.assertEqual(status, 409, path)
            self.assertIn("Demo mode", result["error"])
        self.assertEqual(self.fakes.calls, [])
        self.assertEqual(self.server.jobs.list(), [])
        # Read-only views still work.
        for path in ["/api/jobs", "/api/serve", "/api/community", "/api/quality/results"]:
            self.assertEqual(self.call(path)[0], 200, path)


class JobTests(ApiCase):
    def test_runtime_status_and_install(self):
        status, info, raw = self.call("/api/runtime")
        self.assertEqual(status, 200)
        self.assertTrue(info["installed"])
        self.assertEqual(info["directory"], "b1-cpu")
        self.assertNotIn(SECRET, raw)
        status, job, _ = self.call("/api/runtime/install", {})
        self.assertEqual((status, job["kind"], job["exclusive"]), (202, "runtime_install", "download:runtime"))
        done = self.wait(job["id"])
        self.assertEqual(done["state"], "done")
        self.assertEqual(done["progress"]["message"], "Saving to runtime")
        self.assertIn(("install", False), self.fakes.calls)

    def test_download_lifecycle_and_plan(self):
        status, plan, _ = self.call("/api/downloads/plan", {"variant_id": self.variant.id})
        self.assertEqual(status, 200)
        self.assertFalse(plan["local_copy"])
        self.assertTrue(plan["enough_space"])
        status, job, _ = self.call("/api/downloads", {"variant_id": self.variant.id})
        self.assertEqual((status, job["exclusive"], job["subject"]["variant_id"]), (202, f"download:{self.variant.id}", self.variant.id))
        done = self.wait(job["id"])
        self.assertEqual(done["state"], "done")
        self.assertEqual(done["result"], {"reused": False, "bytes": self.variant.size_bytes,
                                          "filename": Path(self.variant.filename).name, "variant_id": self.variant.id})
        self.assertEqual(done["progress"]["bytes_per_second"], 50 * 1024**2)
        self.assertTrue(self.fakes.calls[0][2].startswith(self.temp.name))  # the app's own models folder
        self.assertTrue(self.call("/api/downloads/plan", {"variant_id": self.variant.id})[1]["local_copy"])
        # A second download reuses the local copy.
        again = self.wait(self.call("/api/downloads", {"variant_id": self.variant.id})[1]["id"])
        self.assertTrue(again["result"]["reused"])
        status, freed, _ = self.call("/api/downloads/remove", {"variant_id": self.variant.id})
        self.assertEqual((status, freed), (200, {"freed_bytes": 123}))

    def test_download_cancel_and_duplicate_and_busy_remove(self):
        self.fakes.block_download = True
        job = self.call("/api/downloads", {"variant_id": self.variant.id})[1]
        self.wait(job["id"], states=("running",))
        self.assertEqual(self.call("/api/downloads", {"variant_id": self.variant.id})[1]["id"], job["id"])
        self.assertEqual(self.call("/api/downloads/remove", {"variant_id": self.variant.id})[0], 409)
        status, cancelled, _ = self.call(f"/api/jobs/{job['id']}/cancel", {})
        self.assertEqual(status, 200)
        self.assertEqual(self.wait(job["id"])["state"], "cancelled")
        status, listing, _ = self.call("/api/jobs")
        self.assertEqual([j["id"] for j in listing["jobs"]], [job["id"]])

    def test_test_job_uses_app_built_settings_and_hides_paths(self):
        c = candidate(self.variant)
        self.remember(c)
        failed = self.wait(self.call("/api/test", {"candidate_id": c["id"], "kind": "smoke"})[1]["id"])
        self.assertEqual(failed["state"], "failed")
        self.assertIn("Download it first", failed["error"])
        self.downloaded()
        status, job, _ = self.call("/api/test", {"candidate_id": c["id"], "kind": "smoke"})
        self.assertEqual((status, job["exclusive"]), (202, "compute"))
        done = self.wait(job["id"])
        self.assertEqual(done["state"], "done", done["error"])
        self.assertEqual(done["result"]["smoke"]["log_tail"], f"llama_model_load: loading {Path(self.variant.filename).name}")
        _, variant_id, config, commands, kind = self.fakes.calls[-1]
        self.assertEqual((variant_id, kind, config["context"], config["host"]), (self.variant.id, "smoke", 4096, "127.0.0.1"))
        self.assertTrue(config["model_path"].startswith(self.temp.name))
        self.assertEqual(commands["server"], [f"{SECRET}/runtime/llama-server"])
        self.assertIsNone(commands["bench"])
        self.assertEqual(self.store.get("measurements")[-1]["id"], "a1b2c3")

    def test_tune_saves_result_and_tuned_settings_apply(self):
        c = candidate(self.variant)
        self.remember(c)
        self.downloaded()
        status, job, _ = self.call("/api/tune", {"candidate_id": c["id"], "budget_seconds": 120, "goal": "balanced"})
        self.assertEqual(status, 202)
        done = self.wait(job["id"])
        self.assertEqual(done["state"], "done", done["error"])
        self.assertNotIn("model_path", done["result"]["best"])
        record = self.store.get("tuned")[-1]
        for key in ["variant_id", "context", "placement", "fingerprint", "timestamp"]:
            self.assertIn(key, record)
        self.assertEqual((record["variant_id"], record["context"], record["placement"], record["fingerprint"]),
                         (self.variant.id, 4096, "cpu", "fp"))
        self.assertNotIn("model_path", record["best"])
        self.assertEqual(self.fakes.calls[-1][3:], (120, "balanced"))
        # --tuned / "tuned": true reuses the saved threads and flash attention.
        self.wait(self.call("/api/test", {"candidate_id": c["id"], "kind": "speed", "tuned": True})[1]["id"])
        config, commands = self.fakes.calls[-1][2], self.fakes.calls[-1][3]
        self.assertEqual((config["threads"], config["flash_attn"]), (8, "on"))
        self.assertEqual(commands["bench"], [f"{SECRET}/runtime/llama-bench"])

    def test_quiz_saves_quality_results_and_stops_server(self):
        c = candidate(self.variant)
        self.remember(c)
        self.downloaded()
        status, job, _ = self.call("/api/quality/quiz", {"candidate_id": c["id"], "include_needle": True})
        self.assertEqual(status, 202)
        self.assertEqual(job["subject"]["workload"], "coding")  # defaults to the compared workload
        done = self.wait(job["id"])
        self.assertEqual(done["state"], "done", done["error"])
        self.assertEqual(done["result"]["quiz"]["score"], 0.75)
        self.assertEqual(done["result"]["needle"]["context_tokens"], 4096)
        kinds = [r["kind"] for r in self.call("/api/quality/results")[1]["results"]]
        self.assertEqual(kinds, ["quiz", "needle"])
        self.assertEqual(self.fakes.modules["llama_server"].LlamaServer.live, 0)

    def test_blind_comparison_runs_models_one_at_a_time(self):
        first, second = candidate(self.variant), candidate(self.other)
        self.remember(first, second)
        self.downloaded(self.variant)
        self.downloaded(self.other)
        status, job, _ = self.call("/api/quality/compare", {"candidate_ids": [first["id"], second["id"]], "prompts": ["Say hi"]})
        self.assertEqual(status, 202)
        done = self.wait(job["id"])
        self.assertEqual(done["result"], {"comparison_id": "cmp_1"}, done["error"])
        LlamaServer = self.fakes.modules["llama_server"].LlamaServer
        self.assertEqual((LlamaServer.live, LlamaServer.max_live), (0, 1))
        self.assertEqual(self.call("/api/quality/compare/cmp_1")[1]["items"][0]["outputs"][0]["slot"], "A")
        self.assertEqual(self.call("/api/quality/vote", {"comparison_id": "cmp_1", "item": 0, "slot": "B"})[1]["votes"], {"B": 1})
        self.assertIn("mapping", self.call("/api/quality/reveal", {"comparison_id": "cmp_1"})[1])

    def test_quant_check_needs_local_files_and_saves_results(self):
        body = {"reference_variant_id": self.other.id, "variant_ids": [self.variant.id]}
        failed = self.wait(self.call("/api/quality/quant-check", body)[1]["id"])
        self.assertEqual(failed["state"], "failed")
        self.downloaded(self.variant)
        self.downloaded(self.other)
        done = self.wait(self.call("/api/quality/quant-check", body)[1]["id"])
        self.assertEqual(done["state"], "done", done["error"])
        self.assertEqual(done["result"]["reference"], "Q8_0")
        self.assertEqual(done["result"]["notes"], ["Temporary files in tmp were deleted."])
        saved = self.store.get("quality_results")[-1]
        self.assertEqual((saved["kind"], saved["reference_variant_id"], list(saved["results"])), ("quant_check", self.other.id, [self.variant.id]))

    def test_export_keeps_real_path_for_the_script_and_notes_missing_download(self):
        c = candidate(self.variant)
        self.remember(c)
        status, result, _ = self.call("/api/export", {"candidate_id": c["id"], "format": "llama-server"})
        self.assertEqual(status, 200)
        self.assertIn(self.temp.name, result["content"])
        self.assertIn("not downloaded yet", result["notes"][0])
        self.downloaded()
        status, result, _ = self.call("/api/export", {"candidate_id": c["id"], "format": "ollama", "platform": "windows"})
        self.assertEqual((status, result["notes"]), (200, []))

    def test_serve_start_status_stop_and_shutdown(self):
        c = candidate(self.variant)
        self.remember(c)
        self.assertFalse(self.call("/api/serve")[1]["running"])
        self.downloaded()
        status, job, _ = self.call("/api/serve/start", {"candidate_id": c["id"]})
        self.assertEqual((status, job["exclusive"]), (202, "compute"))
        done = self.wait(job["id"])
        self.assertTrue(done["result"]["running"], done["error"])
        status, state, raw = self.call("/api/serve")
        self.assertEqual((state["openai_base_url"], state["config"]["model_path"]), ("http://127.0.0.1:8081/v1", Path(self.variant.filename).name))
        self.assertNotIn(self.temp.name, raw)
        # No second model load while the server holds memory.
        self.assertEqual(self.call("/api/test", {"candidate_id": c["id"]})[0], 409)
        self.assertFalse(self.call("/api/serve/stop", {})[1]["running"])
        self.call("/api/serve/start", {"candidate_id": c["id"]})
        self.wait(self.server.jobs.list()[0]["id"])
        self.server.server_close()
        self.assertEqual(self.fakes.calls[-1], ("serve_stop",))

    def test_queued_compute_job_rechecks_the_server_and_remove_respects_it(self):
        c = candidate(self.variant)
        self.remember(c)
        self.downloaded()
        from llm_configurator import server as server_module
        with patch.object(server_module, "MAX_ACTIVE_JOBS", 2):
            gate = threading.Event()
            self.server.jobs.submit("hold", "hold", lambda p, cancel: gate.wait(5), exclusive="compute")
            test = self.call("/api/test", {"candidate_id": c["id"]})[1]
            self.assertEqual(self.call("/api/tune", {"candidate_id": c["id"], "budget_seconds": 60})[0], 409)  # job cap
            registry = self.fakes.modules["llama_server"].ServerRegistry()
            registry.start(["x"], {"model_path": "m.gguf"})
            self.server.registry = registry  # a server came up while the test waited for the compute slot
            gate.set()
            failed = self.wait(test["id"])
        self.assertEqual(failed["state"], "failed")
        self.assertIn("Stop the running model server first", failed["error"])
        self.server.registry = None
        self.wait(self.call("/api/serve/start", {"candidate_id": c["id"]})[1]["id"])
        self.assertEqual(self.call("/api/downloads/remove", {"variant_id": self.variant.id})[0], 409)
        self.call("/api/serve/stop", {})
        self.assertEqual(self.call("/api/downloads/remove", {"variant_id": self.variant.id})[0], 200)

    def test_odd_job_results_never_break_the_job_list(self):
        job = self.server.jobs.submit("x", "x", lambda p, c: {"nan": float("nan"), "path": Path(SECRET) / "m.gguf"})
        self.server.jobs.wait(job["id"], 2)
        status, listing, raw = self.call("/api/jobs")
        self.assertEqual((status, listing["jobs"][0]["result"]), (200, {"nan": None, "path": "m.gguf"}))

    def test_refresh_during_comparison_discards_the_old_report(self):
        c = candidate(self.variant)
        report = {"candidates": [c], "hardware": {}, "requirements": {}, "shortlist": [], "notes": []}
        def slow_evaluate(*args):
            self.server.latest["generation"] += 1  # as if a refresh finished meanwhile
            return report
        with patch("llm_configurator.server.evaluate", side_effect=slow_evaluate):
            self.assertEqual(self.call("/api/recommend", {})[0], 200)
        self.assertEqual(self.call("/api/test", {"candidate_id": c["id"]})[0], 409)

    def test_local_models_hide_folders(self):
        status, result, raw = self.call("/api/local-models")
        self.assertEqual(status, 200)
        self.assertEqual(result["locations"][0], {"source": "hf_cache", "exists": True, "label": "~/.cache/huggingface/hub"})
        self.assertEqual(result["locations"][1]["label"], "elsewhere")
        job = self.call("/api/local-models/scan", {})[1]
        done = self.wait(job["id"])
        self.assertEqual(done["result"][0]["filename"], "found.gguf")
        self.assertNotIn("path", done["result"][0])
        status, result, raw = self.call("/api/local-models")
        self.assertEqual(result["files"][0]["filename"], "found.gguf")
        self.assertNotIn(SECRET, raw)

    def test_community_import_status_and_share(self):
        self.assertEqual(self.call("/api/community")[1], {"records": [], "source": None, "fetched_at": None})
        done = self.wait(self.call("/api/community/import", {})[1]["id"])
        self.assertEqual((done["result"]["count"], done["result"]["rejected"]), (1, 2))
        self.assertEqual(self.call("/api/community")[1]["records"], [{"tps": 1}])
        status, shared, _ = self.call("/api/community/share", {"measurement_ids": ["a1b2c3"]})
        self.assertEqual(status, 200)
        self.assertTrue(shared["issue_url"].startswith("https://github.com/"))

    def test_missing_module_is_a_plain_conflict(self):
        with patch.dict(sys.modules, {"llm_configurator.community": None}), patch.object(llm_configurator, "community", None, create=True):
            delattr(llm_configurator, "community")
            status, result, _ = self.call("/api/community/share", {"measurement_ids": ["a1b2c3"]})
        self.assertEqual(status, 409)
        self.assertIn("not installed", result["error"])


class FixupTests(ApiCase):
    """Seams found when the modules met each other and real llama.cpp (docs/v0.4/fixups/api-http.md)."""

    def test_scrub_handles_spaces_bare_paths_whole_folders_and_cut_log_tails(self):
        for value in [r"C:\Program Files\llama.cpp\llama-server.exe", r"E:\LLM stuff\x.gguf",
                      "/mnt/My Models/secret project/x.gguf", r"C:\Users\j\OneDrive - Contoso Ltd\Models\x.gguf"]:
            self.assertEqual(scrub({"directory": value})["directory"].rsplit(" ", 1)[-1], PureWindowsPath(value).name)
        self.assertEqual(scrub("Model file not found: /Volumes/Samsung T7/m.gguf. Download it first."),
                         "Model file not found: m.gguf. Download it first.")
        # A root only replaces whole folders: "/tmp" must not eat the end of another folder's name.
        self.assertEqual(scrub("in /home/u/private/tmp now", roots=("/tmp",)), "in tmp now")
        self.assertEqual(scrub("/tmp/My Dir/x.gguf and more", roots=("/tmp",)), "x.gguf and more")
        tail = "e/secret user/models/m.gguf'\n" + "llama_model_load: ok\n" * 100
        self.assertNotIn("secret", scrub({"log_tail": tail})["log_tail"])
        for kept in ["tokens/s", "see ~ for more", "/v1/chat/completions"]:
            self.assertEqual(scrub(kept), kept)

    def test_job_errors_and_progress_are_path_free_in_memory(self):
        c = candidate(self.variant)
        self.remember(c)
        self.downloaded()
        def failing(store, variant, config, commands, **kwargs):
            kwargs["progress"]({"stage": "loading", "done": 12.5, "total": 300, "message": "Loading the model into memory…"})
            raise OSError(f"[Errno 2] No such file: '{SECRET}/My Models/m.gguf'")
        self.fakes.modules["testing"].run_tests = failing
        job = self.wait(self.call("/api/test", {"candidate_id": c["id"], "kind": "smoke"})[1]["id"])
        stored = self.server.jobs.get(job["id"])
        self.assertEqual(stored["state"], "failed")
        self.assertNotIn(SECRET, json.dumps(stored))
        self.assertIn("m.gguf", stored["error"])
        self.assertEqual((stored["progress"]["done"], stored["progress"]["total"], stored["progress"]["seconds"]), (None, None, 12.5))

    def test_serve_status_names_the_candidate_and_uses_logs_folder(self):
        c = candidate(self.variant)
        self.remember(c)
        self.downloaded()
        idle = self.call("/api/serve")[1]
        self.assertEqual((idle["candidate_id"], idle["variant_id"], idle["starting"], idle["error"]), (None, None, False, None))
        done = self.wait(self.call("/api/serve/start", {"candidate_id": c["id"]})[1]["id"])
        self.assertEqual((done["result"]["candidate_id"], done["result"]["variant_id"]), (c["id"], self.variant.id))
        state = self.call("/api/serve")[1]
        self.assertEqual((state["candidate_id"], state["variant_id"]), (c["id"], self.variant.id))
        self.assertTrue((Path(self.temp.name) / "logs").is_dir())
        stopped = self.call("/api/serve/stop", {})[1]
        self.assertEqual((stopped["running"], stopped["candidate_id"]), (False, None))

    def test_serve_prefers_the_port_the_exports_use(self):
        from llm_configurator import server as server_module
        c = candidate(self.variant)
        self.remember(c)
        self.downloaded()
        with socket.socket() as busy:
            busy.bind(("127.0.0.1", 0))
            busy.listen()
            for port, expected in [(busy.getsockname()[1], None), (0, None)]:
                with patch.object(server_module, "PREFERRED_PORT", port):
                    self.wait(self.call("/api/serve/start", {"candidate_id": c["id"]})[1]["id"])
                self.assertEqual(self.fakes.calls[-1][2]["port"], expected)
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            free = probe.getsockname()[1]
        with patch.object(server_module, "PREFERRED_PORT", free):
            self.wait(self.call("/api/serve/start", {"candidate_id": c["id"]})[1]["id"])
        self.assertEqual(self.fakes.calls[-1][2]["port"], free)

    def test_saved_tune_is_used_unless_the_page_says_otherwise(self):
        c = candidate(self.variant)
        self.remember(c)
        self.downloaded()
        self.wait(self.call("/api/serve/start", {"candidate_id": c["id"]})[1]["id"])
        self.assertEqual(self.fakes.calls[-1][2]["threads"], 4)
        self.call("/api/serve/stop", {})
        tune = self.wait(self.call("/api/tune", {"candidate_id": c["id"], "budget_seconds": 60})[1]["id"])
        self.assertTrue(tune["result"]["saved"])
        self.wait(self.call("/api/serve/start", {"candidate_id": c["id"]})[1]["id"])
        self.assertEqual(self.fakes.calls[-1][2]["threads"], 8)
        self.wait(self.call("/api/serve/start", {"candidate_id": c["id"], "tuned": False})[1]["id"])
        self.assertEqual(self.fakes.calls[-1][2]["threads"], 4)
        self.assertIn("-t 8", self.call("/api/export", {"candidate_id": c["id"], "format": "llama-server"})[1]["content"])

    def test_cancelled_tune_says_it_was_not_saved(self):
        c = candidate(self.variant)
        self.remember(c)
        self.downloaded()
        original = self.fakes.tune
        self.fakes.modules["tuner"].tune = lambda *a, **k: {**original(*a, **k), "stopped": "cancelled"}
        done = self.wait(self.call("/api/tune", {"candidate_id": c["id"], "budget_seconds": 60})[1]["id"])
        self.assertFalse(done["result"]["saved"])
        self.assertEqual(self.store.get("tuned", []), [])

    def test_speed_target_reaches_the_test_when_the_app_accepts_it(self):
        c = candidate(self.variant)
        report = {"candidates": [c], "hardware": {"fingerprint": "fp"}, "requirements": {"min_tps": 25.0}, "notes": []}
        with patch("llm_configurator.server.evaluate", return_value=report):
            self.call("/api/recommend", {})
        seen = {}
        def test_job(store, variant, candidate, hardware, kind="full", tuned=False, min_tps=None):
            seen["min_tps"] = min_tps
            return lambda progress, cancel: {"verdict": "works_slowly"}
        with patch.object(app, "test_job", test_job):
            self.wait(self.call("/api/test", {"candidate_id": c["id"]})[1]["id"])
        self.assertEqual(seen, {"min_tps": 25.0})

    def test_local_models_are_reused_never_downloaded_or_deleted(self):
        local = Variant(**{**self.variant.to_dict(), "id": "local:abc:m.gguf", "source": "local", "repo": "local"})
        self.store.put("local_variants", [{"variant": local.to_dict(), "path": f"{SECRET}/gone.gguf"}])
        plan = self.call("/api/downloads/plan", {"variant_id": local.id})
        self.assertEqual((plan[0], plan[1]["local"], plan[1]["local_copy"]), (200, True, False))
        status, error, raw = self.call("/api/downloads", {"variant_id": local.id})
        self.assertEqual(status, 409)
        self.assertIn("no longer there", error["error"])
        self.assertEqual(self.call("/api/downloads/remove", {"variant_id": local.id})[0], 409)
        self.fakes.local[local.id] = Path(self.temp.name) / "m.gguf"
        done = self.wait(self.call("/api/downloads", {"variant_id": local.id})[1]["id"])
        self.assertTrue(done["result"]["reused"])
        self.assertNotIn("download", [call[0] for call in self.fakes.calls])

    def test_reveal_names_candidates_and_waits_for_answers(self):
        first, second = candidate(self.variant), candidate(self.other)
        self.remember(first, second)
        self.downloaded(self.variant)
        self.downloaded(self.other)
        evals = self.fakes.modules["evals"]
        labels = [f"{self.variant.name} {self.variant.quant}", f"{self.other.name} {self.other.quant}"]
        state = {"state": "running"}
        def blinded(store, cid):
            if cid != "cmp_1":
                raise ValueError("That comparison was not found. It may be old; start a new one.")
            return {"comparison_id": cid, "state": state["state"], "items": []}
        evals.blinded = blinded
        evals.reveal = lambda store, cid: {"comparison_id": cid, "mapping": [{"item": 0, "slots": {"A": labels[1], "B": labels[0]},
                                                                             "vote": "A", "winner": labels[1]}],
                                           "tallies": {labels[0]: 0, labels[1]: 1}, "overall": labels[1],
                                           "speed": {labels[0]: {"answers": 1}, labels[1]: {"answers": 1}}, "note": "n"}
        evals.vote = lambda store, cid, item, slot: {"slot": slot}
        self.wait(self.call("/api/quality/compare", {"candidate_ids": [first["id"], second["id"]], "prompts": ["Hi"]})[1]["id"])
        self.assertEqual(self.call("/api/quality/reveal", {"comparison_id": "cmp_1"})[0], 409)
        self.assertEqual(self.call("/api/quality/compare/cmp_9")[0], 404)
        self.assertEqual(self.call("/api/quality/vote", {"comparison_id": "cmp_1", "item": 0, "slot": "tie"})[1], {"slot": "tie"})
        state["state"] = "ready"
        out = self.call("/api/quality/reveal", {"comparison_id": "cmp_1"})[1]
        self.assertEqual({k: out["mapping"][0][k] for k in "AB"}, {"A": second["id"], "B": first["id"]})
        self.assertEqual((out["tallies"], out["overall"]), ({first["id"]: 0, second["id"]: 1}, second["id"]))
        self.assertEqual(out["labels"], {first["id"]: labels[0], second["id"]: labels[1]})
        self.assertEqual(out["mapping"][0]["winner"], second["id"])

    def test_catalogue_endpoints_validate_and_refresh_in_a_job(self):
        calls = []
        catalogue = types.ModuleType("llm_configurator.catalogue_fake")
        with patch("llm_configurator.catalogue.add_entry", lambda store, b, g: calls.append(("add", b, g))), \
             patch("llm_configurator.catalogue.refresh_entry", lambda store, b: calls.append(("refresh", b)) or {"base_repo": b, "variants": 3}), \
             patch("llm_configurator.catalogue.remove_entry", lambda store, b: calls.append(("remove", b)) or {"base_repo": b, "removed": True}):
            for body in [{"base_repo": "../x", "gguf_repo": "a/b"}, {"base_repo": "a/b", "gguf_repo": "https://x/y"},
                         {"base_repo": "a/b"}, {"base_repo": "a/b", "gguf_repo": "c/d", "url": "x"}]:
                self.assertEqual(self.call("/api/catalogue", body)[0], 400, body)
            status, job, _ = self.call("/api/catalogue", {"base_repo": "Org/Model-1", "gguf_repo": "Org/Model-1-GGUF"})
            self.assertEqual((status, job["kind"]), (202, "catalogue_refresh"))
            self.assertEqual(self.wait(job["id"])["result"], {"base_repo": "Org/Model-1", "variants": 3})
            self.assertEqual(self.call("/api/catalogue/remove", {"base_repo": "Org/Model-1"})[1]["removed"], True)
        self.assertEqual(calls, [("add", "Org/Model-1", "Org/Model-1-GGUF"), ("refresh", "Org/Model-1"), ("remove", "Org/Model-1")])

    def test_community_share_accepts_up_to_fifty_ids(self):
        seen = []
        self.fakes.modules["community"].share_payload = lambda store, ids: seen.append(ids) or {"json": "{}", "issue_url": "https://github.com/o/r/issues/new?body=%2Fhome"}
        ids = [f"id_{i}" for i in range(50)]
        status, out, _ = self.call("/api/community/share", {"measurement_ids": ids})
        self.assertEqual((status, seen[-1], out["issue_url"]), (200, ids, "https://github.com/o/r/issues/new?body=%2Fhome"))
        self.assertEqual(self.call("/api/community/share", {"measurement_ids": ids + ["x"]})[0], 400)
        self.assertEqual(self.call("/api/community/share", {"measurement_ids": ["../x"]})[0], 400)

    def test_scan_job_kind_and_export_formats(self):
        self.assertEqual(self.call("/api/local-models/scan", {})[1]["kind"], "local_scan")
        self.assertEqual([f["id"] for f in self.call("/api/export")[1]["formats"]], ["llama-server", "ollama"])

    def test_shutdown_waits_for_cancelled_jobs(self):
        stopped = threading.Event()
        def slow(progress, cancel):
            while not cancel.wait(0.01):
                pass
            time.sleep(0.2)  # the job's own clean-up, like killing llama-bench
            stopped.set()
        self.server.jobs.submit("tune", "t", slow)
        time.sleep(0.05)
        self.server.server_close()
        self.assertTrue(stopped.is_set())


class ServeProcessTests(unittest.TestCase):
    @unittest.skipIf(sys.platform == "win32", "POSIX signals")
    def test_terminate_takes_the_clean_shutdown_path(self):
        with tempfile.TemporaryDirectory() as directory:
            process = subprocess.Popen([sys.executable, "-m", "llm_configurator", "--data-dir", directory, "serve",
                                        "--no-browser", "--port", "0"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            try:
                self.assertIn("LLM Configurator", process.stdout.readline())
                process.send_signal(signal.SIGTERM)
                self.assertEqual(process.wait(timeout=20), 0)
            finally:
                process.kill() if process.poll() is None else None
                process.stdout.close()


REAL_RUNTIME = __import__("os").environ.get("LLM_CONFIG_REAL_RUNTIME")
TINY_MODELS = __import__("os").environ.get("LLM_CONFIG_TINY_MODELS")


@unittest.skipUnless(REAL_RUNTIME and TINY_MODELS, "set LLM_CONFIG_REAL_RUNTIME and LLM_CONFIG_TINY_MODELS (scripts/build_llama_cpp.sh)")
class RealRuntimeHttpTests(unittest.TestCase):
    """The HTTP API end to end against real llama.cpp, from a data folder whose name has a space in it."""

    def test_smoke_test_serve_and_no_folder_reaches_the_page(self):
        import shutil
        from llm_configurator import runtime_install
        with tempfile.TemporaryDirectory() as temp:
            data, models = Path(temp) / "My Data", Path(temp) / "My Models"
            models.mkdir()
            model = models / "tiny-llama-Q4_K_M.gguf"
            shutil.copy(Path(TINY_MODELS) / model.name, model)
            store = Store(data)
            runtime_install.use_directory(store, REAL_RUNTIME)
            app.add_local_job(store, model)(lambda value: None, threading.Event())
            server = make_server(store, port=0)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            url = f"http://127.0.0.1:{server.server_port}"
            try:
                with urlopen(url) as response:
                    token = re.search(r'name="session-token" content="([^"]+)"', response.read().decode())[1]
                def call(path, body=None):
                    request = Request(url + path, data=None if body is None else json.dumps(body).encode(),
                                      headers={"X-Session-Token": token})
                    try:
                        with urlopen(request, timeout=300) as response:
                            raw = response.read().decode()
                    except HTTPError as error:
                        raw = error.read().decode()
                    for secret in (temp, REAL_RUNTIME, "My Models", "My Data"):
                        self.assertNotIn(secret, raw, path)
                    return json.loads(raw)
                def job(path, body):
                    found = call(path, body)
                    self.assertIn("id", found, found)
                    deadline = time.monotonic() + 300
                    while found["state"] not in {"done", "failed", "cancelled"} and time.monotonic() < deadline:
                        time.sleep(0.2)
                        found = call(f"/api/jobs/{found['id']}")
                    return found
                self.assertTrue(call("/api/runtime")["installed"])
                report = call("/api/recommend", {"context": 4096, "include_rankings": False})
                chosen = next(c for c in report["candidates"] if c["quant"] == "Q4_K_M")
                smoke = job("/api/test", {"candidate_id": chosen["id"], "kind": "smoke"})
                self.assertEqual(smoke["state"], "done", smoke["error"])
                self.assertTrue(smoke["result"]["smoke"]["ok"], smoke["result"])
                started = job("/api/serve/start", {"candidate_id": chosen["id"]})
                self.assertTrue(started["result"]["running"], started["error"])
                status = call("/api/serve")
                self.assertEqual((status["candidate_id"], status["config"]["model_path"]), (chosen["id"], model.name))
                with urlopen(status["base_url"] + "/health", timeout=10) as response:
                    self.assertEqual(response.status, 200)
                self.assertFalse(call("/api/serve/stop", {})["running"])
            finally:
                server.shutdown()
                server.server_close()
                thread.join()


class DemoCatalogueTests(ApiCase):
    demo = True

    def test_demo_cannot_change_the_catalogue(self):
        self.assertEqual(self.call("/api/catalogue", {"base_repo": "a/b", "gguf_repo": "c/d"})[0], 409)
        self.assertEqual(self.call("/api/catalogue/remove", {"base_repo": "a/b"})[0], 409)


class RecommendEvidenceTests(unittest.TestCase):
    def test_engine_receives_community_and_tuned_only_when_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory)
            store.put("community", {"records": [{"tps": 5}]})
            store.put("tuned", [{"variant_id": "x"}])
            def new_engine(variants, hardware, requirements, measurements=(), calibration=None, community=(), tuned=()):
                pass
            def old_engine(variants, hardware, requirements, measurements=(), calibration=None):
                pass
            self.assertEqual(app.engine_extras(store, new_engine), {"community": [{"tps": 5}], "tuned": [{"variant_id": "x"}]})
            self.assertEqual(app.engine_extras(store, old_engine), {})
            report = {"candidates": [], "notes": [], "shortlist": []}
            with patch("llm_configurator.app.recommend", side_effect=lambda *a, **k: dict(report, kwargs=k)) as engine, \
                 patch("llm_configurator.app.engine_extras", return_value={"community": [1], "tuned": [2]}), \
                 patch("llm_configurator.app.scan", return_value={"fingerprint": "fp", "gpus": []}):
                result = app.evaluate(store, {"context": 2048})
                self.assertEqual(result["kwargs"], {"community": [1], "tuned": [2]})
                self.assertEqual(app.evaluate(store, {"context": 2048}, demo=True)["kwargs"], {})

    def test_local_variants_join_the_model_list(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory)
            variant = real_variant()
            store.put("variants", [variant.to_dict()])
            local = Variant(**{**variant.to_dict(), "id": "local:1", "source": "local"})
            store.put("local_variants", [{"variant": local.to_dict(), "path": str(Path(directory) / "m.gguf")},
                                         {"variant": {"broken": True}, "path": "x"}])
            self.assertEqual([v.id for v in app.variants(store)], [variant.id, "local:1"])
            self.assertEqual(len(app.variants(store, demo=True)), len(demo_variants()))


if __name__ == "__main__":
    unittest.main()
