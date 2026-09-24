import copy
import io
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock
from urllib.parse import parse_qs, urlsplit

from llm_configurator import community
from llm_configurator.domain import GIB, Cancelled
from llm_configurator.storage import Store

SHA = "ab" * 32
HOST, USER = "eyad-secret-laptop", "eyadsecretuser"
GPU_UUID = "GPU-8f3a2c1e-1234-5678-9abc-def012345678"
FINGERPRINT = "f00dfacecafe12345678"

VARIANT = {"id": "qwen|q4", "repo": "Qwen/Qwen3-8B-GGUF", "filename": "Qwen3-8B-Q4_K_M.gguf", "sha256": SHA,
           "quant": "Q4_K_M", "layers": 36, "source": "catalogue"}


def hardware(**extra):
    base = {"fingerprint": FINGERPRINT, "cpu": "x86_64", "cpu_name": "Intel(R) Core(TM) i7-9700K CPU @ 3.60GHz",
            "os": "Linux-6.18.44-fc-v37-x86_64-with-glibc2.36", "ram_total": int(31.3 * GIB),
            "platform": {"system": "Linux", "machine": "x86_64", "release": "6.18.44-fc-v37"},
            "gpus": [{"index": 0, "uuid": GPU_UUID, "name": "NVIDIA GeForce RTX 4090", "total": 24 * GIB,
                      "available": 20 * GIB, "backend": "cuda", "driver": "555.1"}], "processes": [], "warnings": []}
    base.update(extra)
    return base


def measurement(**extra):
    base = {"id": "m1", "variant_id": "qwen|q4", "sha256": SHA, "fingerprint": FINGERPRINT,
            "timestamp": "2026-09-14T13:37:42.123456+00:00", "context": 8192, "users": 1, "gpu_layers": 36,
            "gpu_uuid": GPU_UUID, "threads": 8, "tps": 101.25, "pp_tps": 3000.0, "ttft_s": 0.21, "depth": 0,
            "kind": "speed_test", "runtime_build": "b4500", "runtime": {"version": "b4500", "backend": "cuda"},
            "settings": {"flash_attn": "on", "cache_type_k": "q8_0", "cache_type_v": "q8_0", "batch": 2048,
                         "ubatch": 512, "n_cpu_moe": 0},
            "raw": "llama-server output", "note": "from speed test"}
    base.update(extra)
    return base


def record(tps=50.0, **changes):
    row = community.anonymize(measurement(tps=tps), hardware(), VARIANT)
    for dotted, value in changes.items():
        target = row
        *path, last = dotted.split("__")
        for key in path:
            target = target[key]
        target[last] = value
    return row


class AnonymizeTests(unittest.TestCase):
    def setUp(self):
        patches = [mock.patch("socket.gethostname", return_value=HOST), mock.patch("getpass.getuser", return_value=USER)]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def test_allowlisted_shape(self):
        row = community.anonymize(measurement(), hardware(), VARIANT)
        self.assertEqual(set(row), {"schema", "id", "month", "kind", "tps", "pp_tps", "ttft_s", "context", "depth",
                                    "variant", "hardware", "settings", "runtime"})
        self.assertEqual(row["schema"], 1)
        self.assertEqual(row["month"], "2026-09")
        self.assertRegex(row["id"], r"^[0-9a-f]{16}$")
        self.assertEqual(row["variant"], {"repo": "Qwen/Qwen3-8B-GGUF", "filename": "Qwen3-8B-Q4_K_M.gguf",
                                          "sha256": SHA, "quant": "Q4_K_M", "layers": 36})
        self.assertEqual(row["hardware"], {"cpu": "Intel Core i7-9700K", "gpu": "NVIDIA GeForce RTX 4090", "backend": "cuda",
                                           "vram_gib": 24, "ram_gib": 32, "unified": False, "os": "Linux",
                                           "os_version": None, "arch": "x86_64"})
        self.assertEqual(row["settings"]["placement"], "gpu")
        self.assertEqual(row["runtime"], {"version": "b4500", "backend": "cuda"})

    def test_ids_are_random_not_machine_derived(self):
        ids = {community.anonymize(measurement(), hardware(), VARIANT)["id"] for _ in range(20)}
        self.assertEqual(len(ids), 20)

    def test_cpu_only_run_drops_gpu(self):
        row = community.anonymize(measurement(gpu_layers=0), hardware(), VARIANT)
        self.assertEqual(row["settings"]["placement"], "cpu")
        self.assertIsNone(row["hardware"]["gpu"])
        self.assertIsNone(row["hardware"]["vram_gib"])
        self.assertEqual(row["hardware"]["backend"], "cpu")
        self.assertEqual(community.anonymize(measurement(gpu_layers=10), hardware(), VARIANT)["settings"]["placement"], "split")

    def test_local_and_custom_models_share_only_the_hash(self):
        for source in ("local", "custom"):
            variant = {**VARIANT, "source": source, "repo": f"{USER}/private-finetune", "filename": f"{USER}-diary.gguf"}
            row = community.anonymize(measurement(), hardware(), variant)
            self.assertIsNone(row["variant"]["repo"])
            self.assertIsNone(row["variant"]["filename"])
            self.assertEqual(row["variant"]["sha256"], SHA)

    def test_rejects_results_without_speed(self):
        for bad in (None, 0, -1, float("nan"), float("inf"), True, "12"):
            with self.assertRaises(ValueError):
                community.anonymize(measurement(tps=bad), hardware(), VARIANT)

    def test_memory_buckets_and_names(self):
        self.assertEqual(community.bucket_gib(15.6 * GIB), 16)
        self.assertEqual(community.bucket_gib(1 * GIB), 4)
        self.assertEqual(community.bucket_gib(7.9 * GIB), 8)
        self.assertIsNone(community.bucket_gib(None))
        self.assertIsNone(community.bucket_gib(float("nan")))
        self.assertEqual(community.clean_name("AMD Ryzen 9 7950X 16-Core Processor", set()), "AMD Ryzen 9 7950X 16-Core")
        self.assertIsNone(community.clean_name("x86_64", set()))
        self.assertIsNone(community.clean_name(None, set()))

    def test_os_family(self):
        cases = [({"os": "Windows-10-10.0.22631-SP0", "platform": {"system": "Windows", "release": "10"}}, ("Windows", "11")),
                 ({"os": "Windows-10-10.0.19045-SP0", "platform": {"system": "Windows", "release": "10"}}, ("Windows", "10")),
                 ({"os": "macOS-14.2.1-arm64-arm-64bit", "platform": {"system": "Darwin", "release": "23.2.0"}}, ("macOS", "14")),
                 ({"os": "Linux-6.1.0-custom-myhost", "platform": {"system": "Linux"}}, ("Linux", None)),
                 ({"os": "FreeBSD-14"}, ("other", None))]
        for hw, expected in cases:
            self.assertEqual(community._os_family(hw), expected)

    def test_apple_unified(self):
        hw = hardware(cpu_name="Apple M2 Max", os="macOS-14.2.1-arm64-arm-64bit", unified_memory=True,
                      platform={"system": "Darwin", "machine": "arm64", "release": "23.2.0"},
                      gpus=[{"index": 0, "uuid": "apple-m2-max", "name": "Apple M2 Max", "backend": "metal", "unified": True,
                             "total": int(48 * GIB)}])
        row = community.anonymize(measurement(gpu_uuid="apple-m2-max"), hw, VARIANT)
        self.assertEqual((row["hardware"]["gpu"], row["hardware"]["backend"], row["hardware"]["unified"], row["hardware"]["arch"]),
                         ("Apple M2 Max", "metal", True, "arm64"))


class PrivacyLeakTests(unittest.TestCase):
    SECRETS = [HOST, "secret-laptop", USER, GPU_UUID, "8f3a2c1e", FINGERPRINT, "/home/", "C:\\Users", "13:37", "12345",
               "SN98765432", "deadbeefcafe", "/models/", "diary", "hostx.local", "4242"]

    def planted(self):
        hw = hardware(
            hostname=HOST, user=USER, node=HOST, fingerprint=FINGERPRINT, disk_free=123456789,
            cpu_name=f"Intel(R) Core(TM) i9-13900K ({HOST}) SN98765432 deadbeefcafe0000 {USER} CPU @ 5.8GHz",
            os=f"Linux-6.1.0-{HOST}-x86_64-with-glibc2.36",
            platform={"system": "Linux", "machine": "x86_64", "release": f"6.1.0-{HOST}", "node": HOST},
            processes=[{"pid": 4242, "name": f"/home/{USER}/secret-app", "rss": 1}],
            gpus=[{"index": 0, "uuid": GPU_UUID, "name": f"NVIDIA GeForce RTX 4090 [{GPU_UUID}] {HOST}",
                   "total": 24 * GIB, "backend": "cuda", "driver": "555.1", "pci": "0000:01:00.0", "serial": "SN98765432"}])
        m = measurement(
            model_path=f"/home/{USER}/models/{USER}-diary.gguf", pid=4242, hostname=HOST, host="hostx.local",
            runtime_build=f"/home/{USER}/llama.cpp/build", raw=f"loaded C:\\Users\\{USER}\\x", note=USER,
            runtime={"version": f"C:\\Users\\{USER}", "backend": "cuda", "path": f"/home/{USER}/bin"},
            settings={"flash_attn": "on", "cache_type_k": "f16", "cache_type_v": "f16", "model_path": f"/home/{USER}/models/x.gguf"})
        variant = {**VARIANT, "filename": f"/home/{USER}/models/Qwen3-8B-Q4_K_M.gguf", "path": f"/home/{USER}/models"}
        return m, hw, variant

    def test_no_planted_secret_appears_anywhere(self):
        m, hw, variant = self.planted()
        with mock.patch("socket.gethostname", return_value=HOST), mock.patch("getpass.getuser", return_value=USER):
            row = community.anonymize(m, hw, variant)
        text = json.dumps(row)
        for secret in self.SECRETS:
            self.assertNotIn(secret.lower(), text.lower(), secret)
        self.assertEqual(row["variant"]["filename"], "Qwen3-8B-Q4_K_M.gguf")  # basename only
        self.assertEqual(row["hardware"]["cpu"], "Intel Core i9-13900K")
        self.assertEqual(row["hardware"]["gpu"], "NVIDIA GeForce RTX 4090")
        self.assertIsNone(row["runtime"]["version"])

    def test_secrets_scrubbed_even_without_personal_word_list(self):
        m, hw, variant = self.planted()
        with mock.patch("socket.gethostname", return_value="x"), mock.patch("getpass.getuser", return_value="y"), \
                mock.patch.dict("os.environ", {}, clear=True):
            text = json.dumps(community.anonymize(m, hw, variant))
        for secret in [GPU_UUID, "8f3a2c1e", FINGERPRINT, "/home/", "C:\\Users", "SN98765432", "deadbeef", "4242", "13:37"]:
            self.assertNotIn(secret.lower(), text.lower(), secret)

    def test_share_payload_has_no_secrets(self):
        m, hw, variant = self.planted()
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch("socket.gethostname", return_value=HOST), mock.patch("getpass.getuser", return_value=USER):
            store = Store(directory)
            store.put("measurements", [m])
            result = community.share_payload(store, ["m1"], hardware=hw, variants=[{**variant, "id": "qwen|q4"}])
        blob = (result["json"] + result["issue_url"]).lower()
        for secret in self.SECRETS:
            self.assertNotIn(secret.lower(), blob.replace("%2f", "/").replace("%5c", "\\"), secret)


    def test_reviewer_found_cases(self):
        # A variant without an explicit catalogue source is treated as private.
        row = community.anonymize(measurement(), hardware(), {"repo": "alice/private-finetune", "filename": "secret-client.gguf",
                                                              "quant": "Q4_K_M", "sha256": SHA, "layers": 36})
        self.assertIsNone(row["variant"]["repo"])
        self.assertIsNone(row["variant"]["filename"])
        # Parts of multi-word host and user names are removed, not just exact tokens.
        with mock.patch("socket.gethostname", return_value="johns-macbook-pro.local"), \
                mock.patch("getpass.getuser", return_value="john.smith"):
            self.assertEqual(community.clean_name("Apple M2 Johns MacBook Pro"), "Apple M2")
            self.assertEqual(community.clean_name("Intel Core i7 john smith"), "Intel Core i7")
        # Serial-like tokens and domain names are dropped.
        self.assertEqual(community.clean_name("NVIDIA RTX A6000 SN12345 ZX9K2LQ7WP build01.corp.acme.com", set()), "NVIDIA RTX A6000")
        self.assertEqual(community.clean_name("AMD Ryzen 7 7950X3D", set()), "AMD Ryzen 7 7950X3D")
        # Runtime versions must look like build tags.
        for build in ("b4567-alice_laptop-deadbee", "1.0+alice.hostname"):
            self.assertIsNone(community.anonymize(measurement(runtime_build=build, runtime={}), hardware(), VARIANT)["runtime"]["version"])
        self.assertEqual(community.anonymize(measurement(runtime={"version": "1.2.3"}), hardware(), VARIANT)["runtime"]["version"], "1.2.3")
        # Huge integers are refused, not crashed on.
        with self.assertRaises(ValueError):
            community.anonymize(measurement(tps=10**400), hardware(), VARIANT)


class ValidationTests(unittest.TestCase):
    def test_good_row_round_trips(self):
        row = record()
        self.assertEqual(community.validate_record(row), row)

    def test_extra_keys_are_dropped(self):
        row = record()
        row["hostname"] = HOST
        row["hardware"]["uuid"] = GPU_UUID
        clean = community.validate_record(row)
        self.assertNotIn("hostname", clean)
        self.assertNotIn("uuid", clean["hardware"])

    def test_bad_rows(self):
        bad = [
            {"tps": float("nan")}, {"tps": float("inf")}, {"tps": -1}, {"tps": 1e308}, {"tps": "fast"}, {"tps": True},
            {"context": 10**400}, {"context": 8192.0}, {"context": True}, {"context": 0},
            {"schema": 2}, {"schema": "1"}, {"id": "not-hex"}, {"id": HOST}, {"month": "2026-13"},
            {"month": "2026-09-14T13:37"}, {"kind": "hack"},
            {"variant__sha256": "zz"}, {"variant__repo": "../../etc/passwd"}, {"variant__filename": "/home/u/x.gguf"},
            {"variant__filename": "a.exe"}, {"hardware__cpu": "<script>alert(1)</script>"},
            {"hardware__gpu": "x" * 200}, {"hardware__vram_gib": 23}, {"hardware__ram_gib": -4},
            {"hardware__backend": "opencl"}, {"hardware__os": "TempleOS"}, {"hardware__unified": 1},
            {"hardware__cpu": "Ｉｎｔｅｌ"}, {"month": "20\u0662\u0666-01"}, {"hardware__os_version": "\u0661\u0661"},
            {"tps": 10**400}, {"runtime__version": "b1-host"}, {"settings__placement": "cpu"}, {"settings__users": 0},
            {"settings__flash_attn": "yes"}, {"runtime__version": "a b"}, {"variant": "x"}, {"hardware": None},
        ]
        for change in bad:
            with self.subTest(change=change), self.assertRaises(ValueError):
                community.validate_record(record(**change))
        no_model = record()
        no_model["variant"].update(sha256=None, repo=None)
        with self.assertRaises(ValueError):
            community.validate_record(no_model)
        for junk in (None, [], "x", 5):
            with self.assertRaises(ValueError):
                community.validate_record(junk)

    def test_parse_counts_rejected_and_deduplicates(self):
        good = record()
        text = json.dumps({"schema": 1, "records": [good, good, {"schema": 1}, "junk", None]})
        records, rejected = community.parse(text)
        self.assertEqual((len(records), rejected), (1, 4))  # a duplicate plus three junk rows
        huge = json.dumps([record()]).replace("50.0", "1" + "0" * 400)
        self.assertEqual(community.parse(huge), ([], 1))

    def test_parse_rejects_nan_literals_and_bad_files(self):
        text = json.dumps({"schema": 1, "records": [record()]}).replace("50.0", "NaN")
        with self.assertRaises(ValueError):
            community.parse(text)
        for text in ("not json", "[" * 100000, json.dumps({"schema": 9, "records": []}),
                     json.dumps({"schema": 1, "records": {"a": 1}})):
            with self.subTest(text=text[:20]), self.assertRaises(ValueError):
                community.parse(text)
        with mock.patch.object(community, "MAX_ROWS", 2), self.assertRaises(ValueError):
            community.parse(json.dumps([record(), record(), record()]))
        with mock.patch.object(community, "MAX_BYTES", 100), self.assertRaises(ValueError):
            community.parse(json.dumps([record()]))

    def test_parse_accepts_list_or_single(self):
        self.assertEqual(len(community.parse(json.dumps([record(), record()]))[0]), 2)
        self.assertEqual(len(community.parse(json.dumps(record()))[0]), 1)

    def test_repo_results_file_is_valid(self):
        path = Path(__file__).resolve().parents[1] / "community" / "results.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(data["schema"], 1)
        records, rejected = community.parse(path.read_text(encoding="utf-8"))
        self.assertEqual(rejected, 0)
        self.assertEqual(len(records), len(data["records"]))


class FakeResponse(io.BytesIO):
    def __init__(self, data, url="https://raw.githubusercontent.com/x", length=None):
        super().__init__(data)
        self.url = url
        self.headers = {} if length is None else {"Content-Length": str(length)}

    def geturl(self):
        return self.url


class FakeOpener:
    def __init__(self, response):
        self.response, self.requests = response, []

    def open(self, request, timeout=None):
        self.requests.append((request.full_url, timeout))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class ImportTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = Store(self.directory.name)
        self.body = json.dumps({"schema": 1, "records": [record(), record(), {"bad": True}]}).encode()

    def test_import_from_text(self):
        result = community.import_records(self.store, text=self.body.decode())
        self.assertEqual((result["source"], len(result["records"]), result["rejected"]), ("pasted", 2, 1))
        saved = self.store.get("community")
        self.assertEqual(set(saved), {"source", "fetched_at", "records"})
        self.assertEqual(len(saved["records"]), 2)

    def test_import_from_https_with_progress(self):
        opener = FakeOpener(FakeResponse(self.body, length=len(self.body)))
        events = []
        with mock.patch.object(community, "_opener", return_value=opener):
            result = community.import_records(self.store, progress=events.append)
        self.assertEqual(opener.requests[0][0], community.DEFAULT_SOURCE)
        self.assertTrue(opener.requests[0][1])
        self.assertEqual(len(result["records"]), 2)
        self.assertEqual(events[-1]["stage"], "done")
        self.assertTrue(all({"stage", "done", "total", "message"} <= set(e) for e in events))

    def test_rejects_http_and_odd_schemes(self):
        opener = FakeOpener(FakeResponse(self.body))
        with mock.patch.object(community, "_opener", return_value=opener):
            for url in ("http://example.com/r.json", "file:///etc/passwd", "ftp://x/y", "javascript:x"):
                with self.subTest(url=url), self.assertRaises(ValueError):
                    community.import_records(self.store, source=url)
        self.assertEqual(opener.requests, [])
        self.assertIsNone(self.store.get("community"))

    def test_rejects_redirect_to_http(self):
        handler = community._HttpsOnlyRedirect()
        from urllib.request import Request
        with self.assertRaises(ValueError):
            handler.redirect_request(Request("https://a.example/x"), None, 302, "Found", {}, "http://a.example/x")
        self.assertIsNotNone(handler.redirect_request(Request("https://a.example/x"), None, 302, "Found", {}, "https://b.example/x"))
        # Defence in depth: a response that ended on http is refused too.
        opener = FakeOpener(FakeResponse(self.body, url="http://evil.example/x"))
        with mock.patch.object(community, "_opener", return_value=opener), self.assertRaises(ValueError):
            community.import_records(self.store)

    def test_redirect_handler_is_installed(self):
        from urllib.request import HTTPRedirectHandler
        handlers = community._opener().handlers
        self.assertTrue(any(isinstance(h, community._HttpsOnlyRedirect) for h in handlers))
        self.assertFalse(any(type(h) is HTTPRedirectHandler for h in handlers))

    def test_rejects_oversized(self):
        with mock.patch.object(community, "MAX_BYTES", 1000):
            big = FakeOpener(FakeResponse(b" " * 5000))
            with mock.patch.object(community, "_opener", return_value=big), self.assertRaises(ValueError):
                community.import_records(self.store)
            declared = FakeOpener(FakeResponse(b"{}", length=10**9))
            with mock.patch.object(community, "_opener", return_value=declared), self.assertRaises(ValueError):
                community.import_records(self.store)

    def test_network_errors_are_plain(self):
        from urllib.error import HTTPError, URLError
        for error in (URLError("down"), TimeoutError(), HTTPError("u", 404, "nf", {}, None)):
            with mock.patch.object(community, "_opener", return_value=FakeOpener(error)), \
                    self.assertRaisesRegex(ValueError, "Community results"):
                community.import_records(self.store)

    def test_cancel(self):
        cancel = threading.Event()
        cancel.set()
        with mock.patch.object(community, "_opener", return_value=FakeOpener(FakeResponse(self.body))), \
                self.assertRaises(Cancelled):
            community.import_records(self.store, cancel=cancel)


class ShareTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = Store(self.directory.name)
        self.store.put("measurements", [measurement(), measurement(id="m2", tps=90.0), measurement(id="m3", fingerprint="other")])
        self.store.put("variants", [VARIANT])

    def test_issue_url(self):
        result = community.share_payload(self.store, ["m1", "m2"], hardware=hardware())
        self.assertTrue(result["fits_in_url"])
        self.assertEqual(result["records"], 2)
        parts = urlsplit(result["issue_url"])
        self.assertEqual((parts.scheme, parts.netloc, parts.path), ("https", "github.com", "/Eyad-3D/LLM-Configurator/issues/new"))
        query = parse_qs(parts.query)
        self.assertEqual(query["labels"], ["community-results"])
        self.assertIn("2 speed results", query["title"][0])
        self.assertIn(result["json"], query["body"][0])
        self.assertNotIn(" ", result["issue_url"])
        self.assertNotIn("+", parts.query)  # spaces encoded as %20, not +
        payload = json.loads(result["json"])
        self.assertEqual(len(community.parse(result["json"])[0]), 2)
        self.assertEqual(payload["schema"], 1)

    def test_issue_url_falls_back_when_too_long(self):
        ids = []
        for i in range(40):
            self.store.append("measurements", measurement(id=f"x{i}"))
            ids.append(f"x{i}")
        result = community.share_payload(self.store, ids, hardware=hardware())
        self.assertFalse(result["fits_in_url"])
        self.assertLessEqual(len(result["issue_url"]), community.MAX_URL)
        self.assertIn("Paste", parse_qs(urlsplit(result["issue_url"]).query)["body"][0])
        self.assertEqual(len(json.loads(result["json"])["records"]), 40)

    def test_refuses_bad_requests(self):
        for ids in ([], ["missing"], "m1", [5], ["../x"], ["m1"] * 51):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                community.share_payload(self.store, ids, hardware=hardware())
        with self.assertRaisesRegex(ValueError, "different hardware"):
            community.share_payload(self.store, ["m3"], hardware=hardware())

    def test_unknown_variant_shares_hash_only(self):
        self.store.put("measurements", [measurement(variant_id="gone")])
        row = json.loads(community.share_payload(self.store, ["m1"], hardware=hardware())["json"])["records"][0]
        self.assertEqual(row["variant"]["sha256"], SHA)
        self.assertIsNone(row["variant"]["repo"])

    def test_scans_hardware_when_not_given(self):
        with mock.patch("llm_configurator.hardware.scan", return_value=hardware()) as scan:
            community.share_payload(self.store, ["m1"])
        scan.assert_called_once_with(include_processes=False)


class EvidenceTests(unittest.TestCase):
    CONFIG = {"context": 8192, "gpu_layers": 36, "total_layers": 36, "gpu_uuid": GPU_UUID, "parallel": 1}

    def rows(self, speeds, **changes):
        return [record(tps=s, **changes) for s in speeds]

    def test_same_gpu_median_and_iqr(self):
        rows = self.rows([80, 90, 100, 110, 120])
        result = community.evidence(rows, VARIANT, hardware(), self.CONFIG)
        self.assertEqual(result["similar"], "same_gpu")
        self.assertEqual((result["median_tps"], result["n"], result["range"]), (100.0, 5, [90.0, 110.0]))
        self.assertIn("not on yours", result["note"])

    def test_single_match_is_not_enough(self):
        self.assertIsNone(community.evidence(self.rows([100]), VARIANT, hardware(), self.CONFIG))
        self.assertIsNone(community.evidence([], VARIANT, hardware(), self.CONFIG))

    def test_falls_back_to_same_class(self):
        rows = self.rows([60, 70], hardware__gpu="NVIDIA GeForce RTX 3090") + self.rows([999])
        result = community.evidence(rows, VARIANT, hardware(), self.CONFIG)
        self.assertEqual((result["similar"], result["n"], result["median_tps"]), ("same_class", 3, 70.0))
        other_class = self.rows([60, 70], hardware__gpu="NVIDIA GeForce RTX 4060", hardware__vram_gib=8)
        self.assertIsNone(community.evidence(other_class, VARIANT, hardware(), self.CONFIG))

    def test_filters_model_placement_and_context(self):
        base = self.rows([100])
        noise = (self.rows([1], variant__sha256="cd" * 32) + self.rows([2], context=65536)
                 + self.rows([3], settings__placement="split", settings__gpu_layers=10))
        self.assertIsNone(community.evidence(base + noise, VARIANT, hardware(), self.CONFIG))
        self.assertEqual(community.evidence(base + self.rows([110], context=12000), VARIANT, hardware(), self.CONFIG)["n"], 2)

    def test_cpu_tiers(self):
        config = {**self.CONFIG, "gpu_layers": 0}
        rows = [community.anonymize(measurement(gpu_layers=0, tps=t), hardware(), VARIANT) for t in (10, 12)]
        self.assertEqual(community.evidence(rows, VARIANT, hardware(), config)["similar"], "same_cpu")
        for row in rows:
            row["hardware"]["cpu"] = "AMD Ryzen 5 5600X"
        self.assertEqual(community.evidence(rows, VARIANT, hardware(), config)["similar"], "same_class")
        # GPU results never stand in for a CPU-only run.
        self.assertIsNone(community.evidence(self.rows([100, 100]), VARIANT, hardware(), config))

    def test_matches_by_repo_and_quant_without_hash(self):
        rows = self.rows([50, 60], variant__sha256=None)
        self.assertEqual(community.evidence(rows, {**VARIANT, "sha256": None}, hardware(), self.CONFIG)["n"], 2)
        self.assertIsNone(community.evidence(rows, {**VARIANT, "sha256": None, "quant": "Q8_0"}, hardware(), self.CONFIG))

    def test_ignores_malformed_rows(self):
        rows = self.rows([100, 120]) + [{}, None, {"settings": {}}, "x"]
        rows = [r for r in rows if r is not None and not isinstance(r, str)] + [copy.deepcopy(rows[0])]
        self.assertEqual(community.evidence(rows, VARIANT, hardware(), self.CONFIG)["n"], 3)


if __name__ == "__main__":
    unittest.main()
