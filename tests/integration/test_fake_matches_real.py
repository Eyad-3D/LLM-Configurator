"""Checks the fake llama.cpp (tests/fixtures/fake_llama.py) against output captured from a REAL build.

The samples in tests/integration/samples come from scripts/capture_llama_facts.py run
against llama.cpp 4df29be (see docs/v0.4/llama-cpp-facts.md). These tests need no real
runtime, so they run in the normal suite; they skip until the fake is merged.
"""
import json
import os
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

    def run_fake(self, mode, args, timeout=60, **knobs):
        env = {k: v for k, v in os.environ.items() if not k.startswith("FAKE_LLAMA_")}
        env.update({f"FAKE_LLAMA_{k}": str(v) for k, v in knobs.items()})
        return subprocess.run(self.fx.fake_command(mode) + args, capture_output=True, text=True, timeout=timeout, env=env)

    def start_server(self, args, **knobs):
        """(process, base_url) of a fake llama-server that answers /health 200; the caller stops it."""
        port = free_port()
        env = {k: v for k, v in os.environ.items() if not k.startswith("FAKE_LLAMA_")}
        env.update({f"FAKE_LLAMA_{k}": str(v) for k, v in knobs.items()})
        process = subprocess.Popen(self.fx.fake_command("server") + ["-m", self.model, "--port", str(port)] + args,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, env=env)
        self.addCleanup(lambda: (process.kill(), process.wait(), process.stderr.close()))
        base = f"http://127.0.0.1:{port}"
        for _ in range(400):
            try:
                if http("GET", base + "/health")[0] == 200:
                    return process, base
            except OSError:
                pass
            time.sleep(0.05)
        self.fail("fake llama-server did not start")

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
        self.assertTrue(done.stdout.startswith(real.split("# ---- stdout ----\n", 1)[1][:len("0.00 minutes\n\nchunk")]))
        self.assertRegex(done.stderr, r"I kl_divergence: computing over \d+ chunks, n_ctx=\d+")

    def test_perplexity_normal_layout(self):
        real = (SAMPLES / "perplexity-normal.txt").read_text()
        real_out, real_err = real.split("# ---- stdout ----\n", 1)[1].split("# ---- stderr ----\n")
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp) / "corpus.txt"
            corpus.write_text("The quick brown fox jumps over the lazy dog. " * 400)
            done = self.run_fake("perplexity", ["-m", self.model, "-f", str(corpus), "-c", "128"])
        chunks = re.compile(r"^0\.00 minutes\n(\[\d+\]\d+\.\d{4},)+\n")
        self.assertRegex(real_out, chunks)
        self.assertRegex(done.stdout, chunks)
        final = re.compile(r"ETA \d+\.\d\d\.\d{3}\.\d{3} I Final estimate: PPL = \d+\.\d{4} \+/- \d+\.\d{5}")
        self.assertRegex(real_err, final)
        self.assertRegex(done.stderr, final)

    def test_bench_load_errors_are_one_line_like_real(self):
        # Without -v real llama-bench hides llama.cpp's log: no cause (missing, unknown arch, out of memory) on stderr.
        line = re.compile(r"^llama_bench: error: failed to load model '[^']+'\n$")
        for name in ["error-bench-missing-file.txt", "error-bench-unknown-arch.txt"]:
            real = (SAMPLES / name).read_text()
            self.assertRegex(real.split("# ---- stderr ----\n", 1)[1], line)
            self.assertIn("# ---- stdout ----\n[\n", real)
        missing = str(Path(self.tmp.name) / "does-not-exist.gguf")
        for args, knobs in [(["-m", missing], {}), (["-m", self.model], {"FAIL": "arch"}),
                            (["-m", self.model, "-ngl", "99"], {"MAX_GPU_LAYERS": "2"})]:
            done = self.run_fake("bench", args + ["-o", "json", "-p", "8", "-n", "4", "-r", "1"], **knobs)
            self.assertEqual(done.returncode, 1)
            self.assertRegex(done.stderr, line)
            self.assertTrue(done.stdout.startswith("["))

    def test_bench_flag_values_like_real(self):
        # Checked on the real llama-bench 4df29be: these -fa words work, "yes" fails; -mmp maps onto load_mode.
        words = {"on": 1, "off": 0, "auto": -1, "1": 1, "0": 0, "-1": -1, "true": 1, "false": 0, "enabled": 1,
                 "disabled": 0}
        done = self.run_fake("bench", ["-m", self.model, "-r", "1", "-o", "jsonl", "-p", "8", "-n", "0",
                                       "-fa", ",".join(words)])
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual([json.loads(r)["flash_attn"] for r in done.stdout.splitlines()], list(words.values()))
        bad = self.run_fake("bench", ["-m", self.model, "-fa", "yes"])
        self.assertEqual((bad.returncode, bad.stderr), (1, "error: invalid parameter for argument: -fa\n"))
        self.assertTrue(bad.stdout.startswith("usage: "))
        self.assertEqual(sample_json("bench-basic.json")[0]["load_mode"], "auto")
        for args, mode in [([], "auto"), (["-mmp", "0"], "none"), (["-mmp", "1"], "mmap"), (["-lm", "none"], "none")]:
            done = self.run_fake("bench", ["-m", self.model, "-r", "1", "-o", "jsonl", "-p", "8", "-n", "0"] + args)
            self.assertEqual(json.loads(done.stdout.splitlines()[0])["load_mode"], mode, args)

    def test_bench_version_output_like_real(self):
        real = (SAMPLES / "version.txt").read_text().split("$ llama-bench --version\n", 1)[1].split("\n$ ", 1)[0]
        real_err = real.split("# ---- stderr ----\n", 1)[1].strip()
        done = self.run_fake("bench", ["--version"])
        self.assertEqual(done.returncode, 1)
        self.assertEqual(done.stderr.strip(), real_err)
        self.assertTrue(done.stdout.startswith("usage: "))

    def test_list_devices_like_real(self):
        # Real CPU-only llama-server/llama-bench/llama-perplexity --list-devices: stdout, exit 0.
        for mode in ["server", "bench", "perplexity"]:
            done = self.run_fake(mode, ["--list-devices"], GPU="none")
            self.assertEqual((done.returncode, done.stdout), (0, "Available devices:\n  (none)\n"), mode)
        gpu = self.run_fake("server", ["--list-devices"])
        self.assertRegex(gpu.stdout, r"^Available devices:\n  CUDA0: .+ \(\d+ MiB, \d+ MiB free\)\n$")

    def test_removed_draft_flags_fail_with_the_real_text(self):
        help_text = (SAMPLES / "help-llama-server.txt").read_text()
        self.assertIn("--draft, --draft-n, --draft-max N       the argument has been removed. use --spec-draft-n-max or", help_text)
        for flag, hint in [("--draft-max", "use --spec-draft-n-max or --spec-ngram-mod-n-max"),
                           ("--draft-min", "use --spec-draft-n-min or --spec-ngram-mod-n-min")]:
            done = self.run_fake("server", ["-m", self.model, flag, "4"])
            self.assertEqual(done.returncode, 1)
            self.assertIn(f'error while handling argument "{flag}": the argument has been removed. {hint}', done.stderr)
        process, base = self.start_server(["-md", self.model, "--spec-type", "draft-simple", "--spec-draft-n-max", "4"])
        status, body = http("POST", base + "/completion", {"prompt": "The quick brown fox", "n_predict": 8})
        self.assertEqual(status, 200)
        self.assertTrue({"draft_n", "draft_n_accepted"} <= set(body["timings"]))

    def test_cpu_only_warnings_and_argument_errors_like_real(self):
        real = (SAMPLES / "error-server-bad-flag.txt").read_text().split("# ---- stderr ----\n", 1)[1]
        done = self.run_fake("server", ["-c", "512", "-ngl", "0", "-m", self.model, "--no-such-flag"], GPU="none")
        self.assertEqual(done.returncode, 1)
        self.assertEqual(done.stderr, real)

    def test_server_log_lines_like_real(self):
        real = (SAMPLES / "server-log.txt").read_text()
        process, base = self.start_server([])
        process.kill()
        process.wait()
        fake = process.stderr.read()
        for pattern in [r"(?m)^\d+\.\d\d\.\d{3}\.\d{3} I srv  llama_server: model loaded$",
                        r"(?m)^\d+\.\d\d\.\d{3}\.\d{3} I srv  llama_server: listening on http://127\.0\.0\.1:\d+$",
                        r"(?m)^\d+\.\d\d\.\d{3}\.\d{3} I srv    load_model: loading model '"]:
            self.assertRegex(real, pattern)
            self.assertRegex(fake, pattern)

    def test_server_prompt_cache_like_real(self):
        real = json.loads((SAMPLES / "server-chat-no-cache.json").read_text())
        process, base = self.start_server(["-c", "2048"])
        first = http("POST", base + "/v1/chat/completions", real["request"])[1]["timings"]
        self.assertEqual((first["cache_n"], real["timings"]["cache_n"]), (0, 0))
        again = http("POST", base + "/v1/chat/completions", real["request"])[1]["timings"]
        self.assertEqual(again["cache_n"], 0, "cache_prompt: false re-reads the whole prompt")
        self.assertEqual(again["prompt_n"], first["prompt_n"])
        cached = dict(real["request"], cache_prompt=True)
        http("POST", base + "/v1/chat/completions", cached)
        hit = http("POST", base + "/v1/chat/completions", cached)[1]
        # Real: a repeated prompt re-reads only one token ("cache_n": 66, "prompt_n": 1 for 67 prompt tokens).
        self.assertEqual((hit["timings"]["prompt_n"], hit["timings"]["cache_n"]), (1, hit["usage"]["prompt_tokens"] - 1))
        self.assertEqual(hit["usage"]["prompt_tokens_details"]["cached_tokens"], hit["timings"]["cache_n"])

    def test_server_stream_models_and_props_like_real(self):
        real_chunks = [json.loads(line[6:]) for line in (SAMPLES / "server-chat-completions-stream.txt").read_text().splitlines()
                       if line.startswith("data: {")]
        process, base = self.start_server([])
        request = urllib.request.Request(base + "/v1/chat/completions", method="POST",
                                         data=json.dumps({"messages": [{"role": "user", "content": "What is 40+2?"}],
                                                          "stream": True}).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=30) as response:
            lines = response.read().decode().splitlines()
        fake_chunks = [json.loads(line[6:]) for line in lines if line.startswith("data: {")]
        self.assertEqual([line for line in lines if line.startswith("data: ")][-1], "data: [DONE]")
        for chunks in (real_chunks, fake_chunks):
            self.assertEqual(chunks[-1]["choices"], [])
            self.assertTrue({"usage", "timings"} <= set(chunks[-1]))
            self.assertEqual(chunks[-2]["choices"][0]["finish_reason"], "stop")
            self.assertNotIn("usage", chunks[-2])
        real = json.loads((SAMPLES / "server-models.json").read_text())["response"]
        fake = http("GET", base + "/v1/models")[1]
        self.assertEqual(set(real) - set(fake), set())
        self.assertEqual(set(real["data"][0]) - set(fake["data"][0]), set())
        self.assertEqual(set(real["data"][0]["meta"]) - set(fake["data"][0]["meta"]), set())
        real = json.loads((SAMPLES / "server-props.json").read_text())["response"]
        fake = http("GET", base + "/props")[1]
        for key in ["default_generation_settings", "total_slots", "model_alias", "model_path", "modalities",
                    "chat_template_caps", "build_info"]:
            self.assertIn(key, real)
            self.assertIn(key, fake)
        self.assertIn("n_ctx", fake["default_generation_settings"])
        self.assertRegex(fake["build_info"], r"^b\d+-[0-9a-f]+$")
        self.assertRegex(real["build_info"], r"^b\d+-[0-9a-f]+$")

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
        real = {name: (SAMPLES / f"error-server-{name}.txt").read_text()
                for name in ["missing-file", "unknown-arch", "corrupt", "port-in-use"]}
        lines = {"missing-file": "E gguf_init_from_file: failed to open GGUF file '",
                 "unknown-arch": "E llama_model_load: error loading model: unknown model architecture: '",
                 "corrupt": "data is not within the file bounds, model is corrupted or incomplete",
                 "port-in-use": "E srv         start: couldn't bind HTTP server socket, hostname: 127.0.0.1, port: "}
        last = {"port-in-use": "E srv  llama_server: exiting due to HTTP server error"}
        odd = str(Path(self.tmp.name) / "odd.gguf")
        self.fx.write_fake_gguf(odd, architecture="notarealarch", layers=4)
        busy = socket.socket()
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        self.addCleanup(busy.close)
        runs = {"missing-file": (["-m", str(Path(self.tmp.name) / "nope.gguf")], {}),
                "unknown-arch": (["-m", odd], {}),
                "corrupt": (["-m", self.model], {"FAIL": "corrupt"}),
                "port-in-use": (["-m", self.model, "--port", str(busy.getsockname()[1])], {})}
        for name, (args, knobs) in runs.items():
            with self.subTest(name=name):
                self.assertIn(lines[name], real[name])
                if "--port" not in args:
                    args = args + ["--port", str(free_port())]
                done = self.run_fake("server", ["-c", "512", "-ngl", "0"] + args, 30, LOAD_SECONDS="0", **knobs)
                self.assertEqual(done.returncode, 1)
                self.assertIn(lines[name], done.stderr)
                ending = last.get(name, "E srv  llama_server: exiting due to model loading error")
                self.assertTrue(real[name].rstrip().endswith(ending))
                self.assertTrue(done.stderr.rstrip().endswith(ending), done.stderr[-300:])

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
