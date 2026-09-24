"""Checks the fake llama.cpp (tests/fixtures/fake_llama.py) against output captured from a REAL build.

The samples in tests/integration/samples come from scripts/capture_llama_facts.py run
against llama.cpp 4df29be (see docs/v0.4/llama-cpp-facts.md). These tests need no real
runtime, so they run in the normal suite; they skip until the fake is merged.
"""
import json
from pathlib import Path
import re
import socket
import subprocess
import tempfile
import time
import unittest
import urllib.error
import urllib.request

SAMPLES = Path(__file__).with_name("samples")
VERSION = re.compile(r"(?m)^version: (\d+\.\d+\.\d+(-dev)? \(build \d+, commit [0-9a-f]+\)|\d+ \([0-9a-f]+\))$")


def fixtures():
    for name in ("fixtures", "tests.fixtures"):
        try:
            return __import__(name, fromlist=["fake_command"])
        except ImportError:
            continue
    raise unittest.SkipTest("tests/fixtures (the fake llama.cpp) is not merged yet")


def sample_json(name):
    """JSON from a capture transcript or a plain JSON sample."""
    text = (SAMPLES / name).read_text()
    if text.startswith("$ "):
        text = text.split("# ---- stdout ----\n", 1)[1].split("# ---- stderr ----", 1)[0]
    return json.loads(text)


def http(method, url, body=None):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read() or b"null")


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FakeMatchesRealTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fx = fixtures()
        cls.tmp = tempfile.TemporaryDirectory()
        cls.model = str(Path(cls.tmp.name) / "tiny.gguf")
        cls.fx.write_fake_gguf(cls.model, layers=4)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_fake(self, mode, args, timeout=60):
        return subprocess.run(self.fx.fake_command(mode) + args, capture_output=True, text=True, timeout=timeout)

    def test_version_line_format(self):
        for mode in ["server", "perplexity", "cli"]:
            done = self.run_fake(mode, ["--version"])
            self.assertEqual(done.returncode, 0, mode)
            self.assertRegex(done.stderr, VERSION, mode)
        real = (SAMPLES / "version.txt").read_text()
        self.assertRegex(real, VERSION)

    def test_bench_has_no_version_flag_like_real(self):
        # Real llama-bench prints usage and exits 1 on --version; version comes from build_number/build_commit rows.
        self.assertNotEqual(self.run_fake("bench", ["--version"]).returncode, 0)

    def test_bench_json_rows(self):
        real = sample_json("bench-sweep.json")
        done = self.run_fake("bench", ["-m", self.model, "-r", "1", "-o", "json", "-p", "32", "-n", "8", "-t", "1,2",
                                       "-fa", "0,1", "-b", "64,128", "-ub", "32"])
        self.assertEqual(done.returncode, 0, done.stderr[-2000:])
        fake = json.loads(done.stdout)
        self.assertEqual(len(fake), len(real))
        missing = set(real[0]) - set(fake[0])
        self.assertFalse(missing, f"fields real llama-bench prints but the fake does not: {sorted(missing)}")
        for key, value in real[0].items():
            if key in fake[0]:
                self.assertIs(type(fake[0][key]), type(value), key)
        order = lambda rows: [(r["n_batch"], r["flash_attn"], r["n_threads"], r["n_prompt"], r["n_gen"]) for r in rows]
        self.assertEqual(order(fake), order(real), "sweep row order differs from real llama-bench")

    def test_bench_flash_attention_is_an_int(self):
        done = self.run_fake("bench", ["-m", self.model, "-r", "1", "-o", "json", "-p", "8", "-n", "0", "-fa", "on,off,auto"])
        self.assertEqual(done.returncode, 0, done.stderr[-2000:])
        self.assertEqual([r["flash_attn"] for r in json.loads(done.stdout)], [1, 0, -1])

    def test_perplexity_kl_summary_labels(self):
        real = (SAMPLES / "perplexity-kld.txt").read_text()
        labels = [line.split(":")[0] for line in real.splitlines() if re.match(r"^(Mean|Median|Maximum|Minimum|RMS|Same|\s*[\d.]+%)\s", line)]
        self.assertIn("Same top p", labels)
        with tempfile.TemporaryDirectory() as tmp:
            base = str(Path(tmp) / "base.kld")
            corpus = Path(tmp) / "corpus.txt"
            corpus.write_text("The quick brown fox jumps over the lazy dog. " * 400)
            first = self.run_fake("perplexity", ["-m", self.model, "-f", str(corpus), "-c", "128", "--kl-divergence-base", base])
            self.assertEqual(first.returncode, 0, first.stderr[-2000:])
            done = self.run_fake("perplexity", ["-m", self.model, "--kl-divergence-base", base, "--kl-divergence"])
            self.assertEqual(done.returncode, 0, done.stderr[-2000:])
        fake = [line.split(":")[0] for line in done.stdout.splitlines() if re.match(r"^(Mean|Median|Maximum|Minimum|RMS|Same|\s*[\d.]+%)\s", line)]
        self.assertEqual(set(labels) - set(fake), set(), "summary lines real llama-perplexity prints on stdout")

    def test_server_rejects_removed_draft_max_like_real(self):
        process = subprocess.Popen(self.fx.fake_command("server") + ["-m", self.model, "--port", str(free_port()), "-md",
                                                                     self.model, "--draft-max", "4"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            code = process.wait(15)
        except subprocess.TimeoutExpired:
            code = None
        finally:
            process.kill()
            process.wait()
        self.assertTrue(code, "real llama-server exits 1: --draft-max has been removed (use --spec-draft-n-max)")

    def test_server_error_texts(self):
        real = {name: (SAMPLES / f"error-server-{name}.txt").read_text() for name in ["missing-file", "unknown-arch", "corrupt"]}
        self.assertIn("failed to open GGUF file", real["missing-file"])
        self.assertIn("unknown model architecture: 'notarealarch'", real["unknown-arch"])
        self.assertIn("is not within the file bounds, model is corrupted or incomplete", real["corrupt"])
        missing = self.run_fake("server", ["-m", str(Path(self.tmp.name) / "nope.gguf"), "--port", str(free_port())], 30)
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn("failed to open GGUF file", missing.stdout + missing.stderr)

    def test_server_http_shapes(self):
        port = free_port()
        process = subprocess.Popen(self.fx.fake_command("server") + ["-m", self.model, "--port", str(port), "-c", "2048"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            base = f"http://127.0.0.1:{port}"
            for _ in range(200):
                try:
                    status, body = http("GET", base + "/health")
                except OSError:
                    time.sleep(0.05)
                    continue
                if status == 200:
                    break
                self.assertEqual(body, json.loads(json.loads((SAMPLES / "server-health.json").read_text())[1]["body"]))
                time.sleep(0.05)
            self.assertEqual(body, {"status": "ok"})
            real = json.loads((SAMPLES / "server-chat-completions.json").read_text())
            status, fake = http("POST", base + "/v1/chat/completions", real["request"])
            self.assertEqual(status, 200)
            self.assertEqual(set(real["response"]) - set(fake), set())
            self.assertEqual(set(real["response"]["timings"]) - set(fake["timings"]), set())
            self.assertEqual(set(real["response"]["usage"]) - set(fake["usage"]), set())
            real = json.loads((SAMPLES / "server-completion.json").read_text())
            status, fake = http("POST", base + "/completion", real["request"])
            self.assertEqual(status, 200)
            self.assertEqual({"content", "stop", "timings", "tokens_predicted", "tokens_evaluated"} - set(fake), set())
            status, fake = http("POST", base + "/tokenize", {"content": "Hello world"})
            self.assertTrue(all(isinstance(t, int) for t in fake["tokens"]))
            real = json.loads((SAMPLES / "server-error-context-exceeded.json").read_text())
            status, fake = http("POST", base + "/v1/chat/completions",
                                {"messages": [{"role": "user", "content": "hello " * 3000}], "max_tokens": 4})
            self.assertEqual(status, real["status"])
            self.assertEqual(fake["error"]["type"], real["response"]["error"]["type"])
        finally:
            process.terminate()
            process.wait(20)


if __name__ == "__main__":
    unittest.main()
