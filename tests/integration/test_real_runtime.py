"""Integration tests against a REAL llama.cpp build and tiny generated GGUF models.

Skipped unless both are set:
  LLM_CONFIG_REAL_RUNTIME=<directory holding llama-server, llama-bench, llama-perplexity, ...>
  LLM_CONFIG_TINY_MODELS=<directory written by scripts/make_tiny_models.py --bin ...>
Each module test also skips while its module has not been merged yet. The tiny models
have random weights, so tests check shapes and plumbing, never answer quality.
Reproduce: see docs/v0.4/handoff/harness.md.
"""
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request

RUNTIME = os.environ.get("LLM_CONFIG_REAL_RUNTIME")
MODELS = os.environ.get("LLM_CONFIG_TINY_MODELS")
SAMPLES = Path(__file__).with_name("samples")
CORPUS = Path(__file__).resolve().parents[2] / "scripts" / "tiny_corpus.txt"
THREADS = 2


def tool(name):
    return [str(Path(RUNTIME) / name)]


def model(name):
    path = Path(MODELS) / name
    if not path.exists():
        raise unittest.SkipTest(f"{path} missing; run scripts/make_tiny_models.py with --bin")
    return str(path)


def module(name):
    try:
        return importlib.import_module(f"llm_configurator.{name}")
    except ModuleNotFoundError as error:
        if error.name == f"llm_configurator.{name}":
            raise unittest.SkipTest(f"llm_configurator.{name} is not merged yet")
        raise


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def http(method, url, body=None, timeout=60):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read() or b"null")
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read() or b"null")
    except OSError:
        return None, None


def wait_healthy(port, process, timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        if http("GET", f"http://127.0.0.1:{port}/health", timeout=2)[0] == 200:
            return True
        time.sleep(0.05)
    return False


def tiny_variant(path, quant="F16", moe=False):
    """A local Variant for a tiny model, built from the discover module when merged, else by hand."""
    from llm_configurator.domain import Variant
    from llm_configurator.runtime import digest
    try:
        gguf = importlib.import_module("llm_configurator.gguf")
        return gguf.variant_from_file(path, sha256=digest(path))
    except ModuleNotFoundError:
        pass
    size = Path(path).stat().st_size
    return Variant(id=f"local/{Path(path).name}", name=Path(path).name, base_repo="local/tiny", repo="local/tiny",
                   revision="local", base_revision="local", filename=Path(path).name, sha256=digest(path), quant=quant,
                   size_bytes=size, layers=4, kv_heads=4, head_dim=64, max_context=16384,
                   architecture="qwen3_moe" if moe else "llama", experts=8 if moe else 0, active_experts=2 if moe else 0,
                   expert_fraction=0.5 if moe else 0.0, source="local")


def config(path, **overrides):
    base = {"model_path": path, "context": 1024, "parallel": 1, "gpu_layers": 0, "total_layers": 4, "threads": THREADS}
    base.update(overrides)
    return base


@unittest.skipUnless(RUNTIME and MODELS, "set LLM_CONFIG_REAL_RUNTIME and LLM_CONFIG_TINY_MODELS")
class RawToolTests(unittest.TestCase):
    """What the binaries really do; these facts back docs/v0.4/llama-cpp-facts.md."""

    def run_tool(self, argv, timeout=120):
        return subprocess.run(argv, capture_output=True, text=True, timeout=timeout, errors="replace")

    def test_version_format(self):
        for name in ["llama-server", "llama-perplexity", "llama-cli", "llama-gguf-split"]:
            done = self.run_tool(tool(name) + ["--version"], 30)
            self.assertEqual(done.returncode, 0, name)
            # Printed on stderr, not stdout. Since mid-2026: "version: 0.1.0-dev (build 1, commit 4df29be)";
            # older builds print "version: 5678 (4df29be)". Parsers must accept both.
            self.assertRegex(done.stderr, r"(?m)^version: (\d+\.\d+\.\d+(-dev)? \(build \d+, commit [0-9a-f]+\)|\d+ \([0-9a-f]+\))$", name)
            self.assertRegex(done.stderr, r"(?m)^built with .+ for .+$", name)

    def test_llama_bench_has_no_version_flag(self):
        done = self.run_tool(tool("llama-bench") + ["--version"], 30)
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("usage:", done.stdout)

    def test_bench_json_rows_have_contract_fields(self):
        done = self.run_tool(tool("llama-bench") + ["-m", model("tiny-llama-F16.gguf"), "-r", "1", "-o", "json",
                                                  "-p", "16", "-n", "4", "-d", "0,32", "-t", "1,2", "-fa", "0,1"])
        self.assertEqual(done.returncode, 0, done.stderr[-2000:])
        rows = json.loads(done.stdout)
        # 2 depths x 2 thread counts x 2 flash settings x (prompt test + generation test)
        self.assertEqual(len(rows), 16)
        for key in ["n_prompt", "n_gen", "n_depth", "avg_ts", "stddev_ts", "n_gpu_layers", "n_threads", "type_k", "type_v",
                    "flash_attn", "n_batch", "n_ubatch", "build_commit", "build_number", "model_filename", "n_cpu_moe",
                    "devices", "samples_ts"]:
            self.assertIn(key, rows[0], key)
        self.assertEqual({r["n_depth"] for r in rows}, {0, 32})
        # flash_attn is an int in JSON: -1 auto, 0 off, 1 on (not a bool).
        self.assertEqual({r["flash_attn"] for r in rows}, {0, 1})
        self.assertTrue(all((r["n_prompt"] == 0) != (r["n_gen"] == 0) for r in rows))

    def test_bench_flash_attention_words(self):
        done = self.run_tool(tool("llama-bench") + ["-m", model("tiny-llama-F16.gguf"), "-r", "1", "-o", "json",
                                                  "-p", "8", "-n", "0", "-t", "2", "-fa", "on,off,auto"])
        self.assertEqual(done.returncode, 0, done.stderr[-2000:])
        self.assertEqual([r["flash_attn"] for r in json.loads(done.stdout)], [1, 0, -1])

    def test_bench_dev_none_keeps_ngl(self):
        done = self.run_tool(tool("llama-bench") + ["-m", model("tiny-llama-F16.gguf"), "-r", "1", "-o", "json",
                                                  "-p", "8", "-n", "0", "-t", "2", "-ngl", "99", "-dev", "none"])
        self.assertEqual(done.returncode, 0, done.stderr[-2000:])
        row = json.loads(done.stdout)[0]
        self.assertEqual(row["n_gpu_layers"], 99)
        self.assertEqual(row["devices"], "none")

    def test_bench_ncmoe_on_moe_model(self):
        done = self.run_tool(tool("llama-bench") + ["-m", model("tiny-qwen3moe-Q4_K_M.gguf"), "-r", "1", "-o", "json",
                                                  "-p", "8", "-n", "4", "-t", "2", "-ncmoe", "0,2"])
        self.assertEqual(done.returncode, 0, done.stderr[-2000:])
        self.assertEqual({r["n_cpu_moe"] for r in json.loads(done.stdout)}, {0, 2})

    def test_server_endpoints_and_timings(self):
        port = free_port()
        process = subprocess.Popen(tool("llama-server") + ["-m", model("tiny-llama-F16.gguf"), "-c", "1024", "-np", "1",
                                                           "-ngl", "0", "--host", "127.0.0.1", "--port", str(port),
                                                           "--jinja", "-t", str(THREADS)],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            self.assertTrue(wait_healthy(port, process))
            base = f"http://127.0.0.1:{port}"
            status, body = http("POST", base + "/v1/chat/completions",
                                {"messages": [{"role": "user", "content": "What is 2+2?"}], "max_tokens": 8,
                                 "temperature": 0.0, "seed": 1})
            self.assertEqual(status, 200)
            for key in ["prompt_n", "prompt_ms", "prompt_per_second", "predicted_n", "predicted_ms", "predicted_per_second"]:
                self.assertIn(key, body["timings"])
            self.assertEqual(body["choices"][0]["finish_reason"], "stop")
            self.assertEqual(body["choices"][0]["message"]["content"], "42")  # the tiny model's wired answer
            # Repeating a prompt reuses the cached prefix; "cache_prompt": false forces a full prompt read.
            again = http("POST", base + "/v1/chat/completions", {"messages": [{"role": "user", "content": "What is 2+2?"}],
                                                                 "max_tokens": 8, "cache_prompt": False})[1]
            self.assertEqual(again["timings"]["cache_n"], 0)
            self.assertEqual(again["timings"]["prompt_n"], body["timings"]["prompt_n"])
            status, error = http("POST", base + "/v1/chat/completions",
                                 {"messages": [{"role": "user", "content": "hello " * 1500}], "max_tokens": 4})
            self.assertEqual(status, 400)
            self.assertEqual(error["error"]["type"], "exceed_context_size_error")
            self.assertEqual(set(body["usage"]) >= {"prompt_tokens", "completion_tokens", "total_tokens"}, True)
            status, body = http("POST", base + "/completion", {"prompt": "Hello", "n_predict": 8, "temperature": 0.0})
            self.assertEqual(status, 200)
            self.assertIn("predicted_per_second", body["timings"])
            self.assertEqual(body["content"], " the quick brown fox.")
            # A second server on the same port fails fast.
            busy = self.run_tool(tool("llama-server") + ["-m", model("tiny-llama-F16.gguf"), "--host", "127.0.0.1",
                                                        "--port", str(port)], 60)
            self.assertNotEqual(busy.returncode, 0)
            self.assertIn("couldn't bind HTTP server socket", busy.stderr)
            status, body = http("POST", base + "/tokenize", {"content": "Hello world"})
            self.assertEqual(status, 200)
            self.assertTrue(all(isinstance(t, int) for t in body["tokens"]))
        finally:
            process.terminate()
            process.wait(20)

    def test_server_failures_exit_nonzero_with_recognisable_text(self):
        port = free_port()
        cases = {"does-not-exist.gguf": r"failed to (open|load)|No such file", "corrupt.gguf": r"failed to (read|load)",
                 "unknown-arch.gguf": r"unknown model architecture: 'notarealarch'"}
        for name, pattern in cases.items():
            path = str(Path(MODELS) / name)
            done = self.run_tool(tool("llama-server") + ["-m", path, "-c", "256", "--host", "127.0.0.1", "--port", str(port)], 60)
            self.assertNotEqual(done.returncode, 0, name)
            self.assertRegex(done.stdout + done.stderr, pattern, name)

    def test_perplexity_kl_divergence_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            logits = str(Path(tmp) / "base.kld")
            base = tool("llama-perplexity") + ["-f", str(CORPUS), "-c", "128", "-b", "128", "-t", str(THREADS), "--chunks", "3"]
            first = self.run_tool(base + ["-m", model("tiny-llama-Q8_0.gguf"), "--kl-divergence-base", logits], 300)
            self.assertEqual(first.returncode, 0, first.stderr[-2000:])
            # The summary is on stdout. Under CPU load llama-perplexity sometimes exits 0 with the summary cut off
            # (its log thread is not flushed at exit), so callers must check for it and retry.
            for _ in range(3):
                second = self.run_tool(tool("llama-perplexity") + ["-m", model("tiny-llama-Q4_K_M.gguf"), "-t", str(THREADS),
                                                                   "--kl-divergence-base", logits, "--kl-divergence"], 300)
                self.assertEqual(second.returncode, 0, second.stderr[-2000:])
                if "Same top p:" in second.stdout:
                    break
        text = second.stdout
        for label in ["Mean    KLD:", "Median  KLD:", "99.0%   KLD:", "Same top p:", "Mean PPL(Q)", "Mean PPL(base)"]:
            self.assertIn(label, text)


@unittest.skipUnless(RUNTIME and MODELS, "set LLM_CONFIG_REAL_RUNTIME and LLM_CONFIG_TINY_MODELS")
class LaunchTests(unittest.TestCase):
    """launch.server_args must produce a command line the real llama-server accepts."""

    CASES = {
        "defaults": {},
        "flash_on_q8_cache": {"flash_attn": "on", "cache_type_k": "q8_0", "cache_type_v": "q8_0"},
        "flash_off_q4_k_only": {"flash_attn": "off", "cache_type_k": "q4_0"},
        "batches_and_parallel": {"batch": 256, "ubatch": 64, "parallel": 2, "context": 512},
        "no_mmap_alias": {"mmap": False, "alias": "tiny model"},
        "mlock": {"mlock": True},
        "device_none_with_gpu_backend": {"gpu_backend": "vulkan"},
        "full_offload_on_cpu_build": {"gpu_layers": 4},
    }

    def serve(self, cfg, request=None):
        """Start llama-server with launch.server_args; returns (healthy, status or /completion body, argv, log)."""
        from llm_configurator.launch import server_args
        port = free_port()
        argv = tool("llama-server") + server_args({**cfg, "port": port})
        process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
        try:
            healthy = wait_healthy(port, process)
            status = None
            if healthy and request:
                status = http("POST", f"http://127.0.0.1:{port}/completion", request)[1]
            elif healthy:
                status = http("POST", f"http://127.0.0.1:{port}/v1/chat/completions",
                              {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 4})[0]
        finally:
            process.terminate()
            try:
                output = process.communicate(timeout=20)[0]
            except subprocess.TimeoutExpired:
                process.kill()
                output = process.communicate()[0]
        return healthy, status, argv, output

    def test_server_args_start_real_server(self):
        path = model("tiny-llama-F16.gguf")
        for name, overrides in self.CASES.items():
            with self.subTest(name):
                healthy, status, argv, output = self.serve(config(path, **overrides))
                self.assertTrue(healthy, f"{argv}\n{output[-3000:]}")
                self.assertEqual(status, 200)

    def test_moe_offload(self):
        healthy, status, argv, output = self.serve(config(model("tiny-qwen3moe-Q4_K_M.gguf"), n_cpu_moe=2))
        self.assertTrue(healthy, f"{argv}\n{output[-3000:]}")
        self.assertEqual(status, 200)

    def test_draft_model_is_accepted_and_used(self):
        # Known mismatch at 4df29be: --draft-max was removed (use --spec-draft-n-max), and -md alone loads the
        # draft model without using it unless --spec-type draft-simple is also given.
        cfg = config(model("tiny-llama-Q8_0.gguf"), draft_model_path=model("tiny-llama-Q4_K_M.gguf"), draft_max=4)
        healthy, status, argv, output = self.serve(cfg, request={"prompt": "Hello", "n_predict": 8})
        self.assertTrue(healthy, f"{argv}\n{output[-3000:]}")
        self.assertIn("draft_n", status["timings"], "draft model loaded but speculative decoding is off")

    def test_split_gguf_loads_from_first_shard(self):
        shards = sorted((Path(MODELS) / "split").glob("*-00001-of-*.gguf"))
        if not shards:
            self.skipTest("no split model")
        healthy, status, argv, output = self.serve(config(str(shards[0])))
        self.assertTrue(healthy, output[-3000:])


@unittest.skipUnless(RUNTIME and MODELS, "set LLM_CONFIG_REAL_RUNTIME and LLM_CONFIG_TINY_MODELS")
class ModuleTests(unittest.TestCase):
    """v0.4 modules driving the real binaries (each skips until its workstream is merged)."""

    def test_llama_server_start_chat_timings_stop(self):
        llama_server = module("llama_server")
        server = llama_server.LlamaServer(tool("llama-server"), config(model("tiny-llama-F16.gguf")))
        server.start(timeout=120)
        try:
            self.assertTrue(server.ready())
            reply = server.chat([{"role": "user", "content": "What is 2+2?"}], max_tokens=8)
            self.assertEqual(set(reply) >= {"text", "finish_reason", "timings", "usage"}, True)
            self.assertGreater(reply["timings"]["predicted_n"], 0)
            self.assertGreater(reply["timings"]["predicted_per_second"], 0)
            raw = server.complete("Hello", max_tokens=4)
            self.assertIn("timings", raw)
            self.assertTrue(server.tokenize("Hello world"))
        finally:
            server.stop()
        self.assertFalse(server.ready())

    def test_llama_server_failure_reasons(self):
        llama_server = module("llama_server")
        for name in ["corrupt.gguf", "unknown-arch.gguf"]:
            with self.subTest(name):
                server = llama_server.LlamaServer(tool("llama-server"), config(str(Path(MODELS) / name)))
                with self.assertRaises(ValueError):
                    server.start(timeout=60)
                self.assertTrue(server.failure_reason())

    def test_smoke_test(self):
        testing = module("testing")
        result = testing.smoke_test(tool("llama-server"), config(model("tiny-llama-F16.gguf")))
        # Random weights cannot answer the question, but the server must start and reply.
        self.assertIn(result["stage_failed"], {None, "reply"})
        self.assertGreater(result["load_seconds"], 0)
        self.assertTrue(result["checks"])

    def test_speed_test(self):
        testing = module("testing")
        hardware = module("hardware").scan(False)
        path = model("tiny-llama-F16.gguf")
        # The tiny vocabulary spends ~1.3 characters per token (real models ~4), so give the filler prompt room.
        result = testing.speed_test(tool("llama-bench"), tool("llama-server"), tiny_variant(path), config(path, context=8192),
                                    hardware)
        summary = result["summary"]
        self.assertGreater(summary["tps"], 0)
        self.assertGreater(summary["pp_tps"], 0)
        self.assertGreater(summary["ttft_s"], 0)
        self.assertEqual(result["measurement"]["kind"], "speed_test")

    def test_tuner_small_budget(self):
        tuner = module("tuner")
        hardware = module("hardware").scan(False)
        path = model("tiny-llama-F16.gguf")
        started = time.monotonic()
        result = tuner.tune(tool("llama-bench"), tiny_variant(path), config(path), hardware, budget_seconds=30,
                            memory_check=lambda cfg: True)
        self.assertLess(time.monotonic() - started, 30 + 60)
        self.assertGreater(result["baseline"]["tps"], 0)
        self.assertTrue(result["trials"])
        self.assertIn(result["stopped"], {"budget", "converged"})
        from llm_configurator.launch import normalize
        normalize(result["best"])

    def test_tuner_bench_args_accepted_by_real_bench(self):
        tuner = module("tuner")
        args = tuner.bench_args(config(model("tiny-llama-F16.gguf")), n_prompt=16, n_gen=4, repetitions=1,
                                sweep={"threads": [1, 2], "flash_attn": ["on", "off"]})
        done = subprocess.run(tool("llama-bench") + args, capture_output=True, text=True, timeout=300)
        self.assertEqual(done.returncode, 0, done.stderr[-2000:])

    def test_quantcheck_kl_check(self):
        quantcheck = module("quantcheck")
        result = quantcheck.kl_check(tool("llama-perplexity"), model("tiny-llama-Q8_0.gguf"),
                                     {"Q4_K_M": model("tiny-llama-Q4_K_M.gguf")}, corpus_path=str(CORPUS), context=128,
                                     chunks=3)
        parsed = result["results"]["Q4_K_M"]
        self.assertIsNotNone(parsed["mean_kld"])
        self.assertIsNotNone(parsed["same_top_p"])
        self.assertTrue(parsed["plain"])

    def test_quantcheck_parser_on_real_sample(self):
        quantcheck = module("quantcheck")
        text = (SAMPLES / "perplexity-kld.txt").read_text()
        parsed = quantcheck.parse_kld_output(text)
        for key in ["mean_kld", "median_kld", "kld_99", "same_top_p", "ppl_base", "ppl"]:
            self.assertIsNotNone(parsed[key], key)

    def test_export_llama_server_script_runs(self):
        export = module("export")
        if os.name == "nt":
            self.skipTest("posix script")
        path = model("tiny-llama-F16.gguf")
        port = free_port()
        result = export.export(config(path, port=port), tiny_variant(path), "llama-server", platform="posix",
                               server_command=tool("llama-server"))
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / result["filename"]
            script.write_text(result["content"])
            process = subprocess.Popen(["bash", str(script)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                       start_new_session=True)
            try:
                self.assertTrue(wait_healthy(port, process), result["content"])
            finally:
                os.killpg(process.pid, 15)
                process.wait(20)

    def test_runtime_install_detects_configured_directory(self):
        runtime_install = module("runtime_install")
        from llm_configurator.storage import Store
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(tmp)
            store.put("settings", {"runtime_dir": RUNTIME})
            found = runtime_install.detect(store)
        self.assertTrue(found["installed"])
        self.assertEqual(found["source"], "configured")
        self.assertIsNotNone(found["version"])
        self.assertIsInstance(found["build"], int)
        self.assertEqual(found["binaries"]["llama-server"][0], tool("llama-server")[0])
        self.assertEqual(found["backend"], "cpu")

    def test_runtime_bench_with_real_llama_bench(self):
        runtime = module("runtime")
        path = model("tiny-llama-F16.gguf")
        record = runtime.bench(tiny_variant(path), path, tool("llama-bench")[0], context=512, layers=0)
        self.assertGreater(record["tps"], 0)
        self.assertEqual(record["raw"]["n_depth"], 512 - 128)


if __name__ == "__main__":
    unittest.main()
