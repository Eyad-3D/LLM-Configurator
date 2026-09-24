import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest

from llm_configurator import testing
from llm_configurator.catalogue import demo_variants
from llm_configurator.domain import GIB, Cancelled
from llm_configurator.engine import allocations, matching_speed
from llm_configurator.storage import Store

# A tiny stand-in for llama-bench: echoes the requested settings back as llama-bench JSON rows.
FAKE_BENCH = r'''
import json, os, sys, time
args = sys.argv[1:]
def get(flag, default=None):
    return args[args.index(flag) + 1] if flag in args else default
if os.environ.get("FAKE_BENCH_SLEEP"):
    open(os.environ["FAKE_BENCH_STARTED"], "w").close()
    time.sleep(float(os.environ["FAKE_BENCH_SLEEP"]))
if os.environ.get("FAKE_BENCH_FAIL") == "oom":
    sys.stderr.write("ggml_backend_cuda_buffer_type_alloc_buffer: allocating 9000 MiB on device 0: cudaMalloc failed: out of memory\n")
    sys.exit(1)
ngl = int(get("-ngl", "99")) + int(os.environ.get("FAKE_BENCH_NGL_SHIFT", "0"))
base = {"build_commit": "abc1234", "build_number": 6500, "backends": "CUDA", "model_filename": get("-m"),
        "n_batch": int(get("-b", "2048")), "n_ubatch": int(get("-ub", "512")), "n_threads": int(get("-t", "8")),
        "type_k": get("-ctk", "f16"), "type_v": get("-ctv", "f16"), "n_gpu_layers": ngl,
        "flash_attn": {"on": 1, "off": 0}.get(get("-fa", "auto"), -1), "n_depth": int(get("-d", "0"))}
rows = [dict(base, n_prompt=int(get("-p")), n_gen=0, avg_ts=812.5, stddev_ts=3.1),
        dict(base, n_prompt=0, n_gen=int(get("-n")), avg_ts=float(os.environ.get("FAKE_BENCH_TPS", "21.5")), stddev_ts=0.4)]
print("main: loading model", file=sys.stderr)
print(json.dumps(rows, indent=2))
'''


class FakeServer:
    instances = []
    reply = "42"
    fail = None
    pid = 4242

    def __init__(self, command, config, log_path=None):
        self.command, self.config, self.stopped, self.calls = command, config, False, []
        FakeServer.instances.append(self)

    def start(self, timeout=300, progress=None, cancel=None):
        if self.fail:
            raise ValueError("llama-server exited while loading")

    def failure_reason(self):
        return "Out of memory: the model does not fit." if self.fail == "oom" else None

    def chat(self, messages, max_tokens=256, temperature=0.0, seed=1, stop=None, timeout=300, chat_template_kwargs=None):
        self.calls.append({"messages": messages, "max_tokens": max_tokens, "chat_template_kwargs": chat_template_kwargs})
        return {"text": self.reply, "finish_reason": "stop",
                "timings": {"prompt_n": 1500, "prompt_ms": 900.0, "prompt_per_second": 1666.0,
                            "predicted_n": 1, "predicted_ms": 40.0, "predicted_per_second": 25.0}, "usage": {}}

    def log_tail(self, chars=4000):
        return "log tail"

    def stop(self, timeout=10):
        self.stopped = True


class FakePeak:
    result = {"peak_ram_bytes": 1 * GIB, "peak_vram_bytes": None, "samples": 5}

    def __init__(self, pid, gpu_backend=None, interval=0.25):
        self.pid, self.gpu_backend, self.running = pid, gpu_backend, False

    def start(self):
        self.running = True

    def stop(self):
        self.running = False
        return dict(FakePeak.result)


def hardware():
    return {"fingerprint": "test-machine", "ram_available": 24 * GIB, "cores": 8, "threads": 16, "gpus": []}


class TestingBase(unittest.TestCase):
    def setUp(self):
        FakeServer.instances, FakeServer.reply, FakeServer.fail = [], "42", None
        FakePeak.result = {"peak_ram_bytes": 1 * GIB, "peak_vram_bytes": None, "samples": 5}
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        script = Path(self.tmp.name) / "fake_bench.py"
        script.write_text(FAKE_BENCH)
        self.bench = [sys.executable, str(script)]
        self.variant = demo_variants()[0]
        self.variant.sha256 = "a" * 64
        self.config = {"model_path": str(Path(self.tmp.name) / "model.gguf"), "context": 4096, "gpu_layers": 0,
                       "total_layers": self.variant.layers, "threads": 8}

    def env(self, **values):
        for key, value in values.items():
            os.environ[key] = value
            self.addCleanup(os.environ.pop, key, None)


class SmokeTests(TestingBase):
    def test_success_records_load_time_and_disables_thinking(self):
        FakeServer.reply = "<think>add them</think>The answer is 42."
        result = testing.smoke_test(["llama-server"], self.config, server_factory=FakeServer)
        self.assertTrue(result["ok"], result)
        self.assertIsNone(result["stage_failed"])
        self.assertEqual(result["reply"], "The answer is 42.")
        self.assertIsInstance(result["load_seconds"], float)
        self.assertTrue(all(c["ok"] for c in result["checks"]))
        self.assertEqual(FakeServer.instances[0].calls[0]["chat_template_kwargs"]["enable_thinking"], False)
        self.assertTrue(FakeServer.instances[0].stopped)

    def test_out_of_memory_at_load(self):
        FakeServer.fail = "oom"
        result = testing.smoke_test(["llama-server"], self.config, server_factory=FakeServer)
        self.assertFalse(result["ok"])
        self.assertEqual(result["stage_failed"], "start")
        self.assertIn("ran out of memory", result["message"])
        self.assertEqual(result["log_tail"], "log tail")
        self.assertTrue(FakeServer.instances[0].stopped)

    def test_garbage_reply(self):
        FakeServer.reply = "the the the the the the the the"
        result = testing.smoke_test(["llama-server"], self.config, server_factory=FakeServer)
        self.assertFalse(result["ok"])
        self.assertEqual(result["stage_failed"], "reply")
        failed = {c["name"] for c in result["checks"] if not c["ok"]}
        self.assertIn("not_repeating", failed)
        self.assertIn("correct_answer", failed)

    def test_template_leak(self):
        FakeServer.reply = "42<|im_end|>\n<|im_start|>user"
        result = testing.smoke_test(["llama-server"], self.config, server_factory=FakeServer)
        self.assertFalse(result["ok"])
        self.assertEqual([c["name"] for c in result["checks"] if not c["ok"]], ["no_template_leak"])
        self.assertIn("chat-format", result["message"])

    def test_unfinished_thinking_counts_as_empty(self):
        FakeServer.reply = "<think>Let me carefully consider 17 and 25 and"
        result = testing.smoke_test(["llama-server"], self.config, server_factory=FakeServer)
        self.assertFalse(result["ok"])
        self.assertFalse(result["checks"][0]["ok"])

    def test_old_server_without_template_kwargs(self):
        class OldServer(FakeServer):
            def chat(self, messages, max_tokens=256, temperature=0.0):
                return {"text": "42", "timings": {}}
        self.assertTrue(testing.smoke_test(["llama-server"], self.config, server_factory=OldServer)["ok"])

    def test_cancel_before_start_stops_nothing_and_raises(self):
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(Cancelled):
            testing.smoke_test(["llama-server"], self.config, cancel=cancel, server_factory=FakeServer)


class BenchTests(TestingBase):
    def test_plan_ends_at_the_users_context(self):
        plan = testing.bench_plan(8192)
        self.assertEqual(plan, {"n_prompt": 512, "n_gen": 128, "depth": 7552})
        for context in [256, 512, 700, 1024, 131072]:
            plan = testing.bench_plan(context)
            self.assertLessEqual(plan["depth"] + plan["n_prompt"] + plan["n_gen"], context)
            self.assertGreaterEqual(plan["depth"], 0)

    def test_bench_args_mirror_launch_settings(self):
        config = dict(self.config, gpu_layers=self.variant.layers, flash_attn="on", cache_type_k="q8_0",
                      cache_type_v="q8_0", batch=1024, ubatch=256)
        args = testing.bench_args(config, 512, 128, 3456)
        joined = " ".join(args)
        for part in ["-p 512", "-n 128", "-d 3456", "-o json", f"-ngl {self.variant.layers + 1}", "-ctk q8_0",
                     "-ctv q8_0", "-fa on", "-t 8", "-b 1024", "-ub 256"]:
            self.assertIn(part, joined)

    def test_parse_tolerates_logs_and_jsonl(self):
        self.assertEqual(len(testing.parse_bench_json('log line\n[{"a": 1}, {"b": 2}]\ntrailer')), 2)
        self.assertEqual(len(testing.parse_bench_json('{"a": 1}\nnoise\n{"b": 2}\n')), 2)
        self.assertEqual(testing.parse_bench_json("nothing here"), [])

    def test_run_bench_reads_both_speeds(self):
        result = testing.run_bench(self.bench, self.config, testing.bench_plan(4096))
        self.assertEqual(result["pp_tps"], 812.5)
        self.assertEqual(result["tps"], 21.5)

    def test_settings_mismatch_is_refused(self):
        self.env(FAKE_BENCH_NGL_SHIFT="3")
        with self.assertRaisesRegex(ValueError, "different settings"):
            testing.run_bench(self.bench, self.config, testing.bench_plan(4096))

    def test_bench_out_of_memory_is_plain(self):
        self.env(FAKE_BENCH_FAIL="oom")
        with self.assertRaisesRegex(ValueError, "ran out of memory"):
            testing.run_bench(self.bench, self.config, testing.bench_plan(4096))

    def test_cancel_mid_bench_kills_the_process(self):
        started = Path(self.tmp.name) / "started"
        self.env(FAKE_BENCH_SLEEP="30", FAKE_BENCH_STARTED=str(started))
        cancel = threading.Event()

        def cancel_when_running():
            deadline = time.monotonic() + 20
            while not started.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            cancel.set()
        threading.Thread(target=cancel_when_running, daemon=True).start()
        begin = time.monotonic()
        with self.assertRaises(Cancelled):
            testing.speed_test(self.bench, ["llama-server"], self.variant, self.config, hardware(), cancel=cancel,
                               server_factory=FakeServer, peak_factory=FakePeak)
        self.assertLess(time.monotonic() - begin, 20)
        self.assertEqual(FakeServer.instances, [])  # never reached the server step

    def test_timeout_kills_the_process(self):
        self.env(FAKE_BENCH_SLEEP="30", FAKE_BENCH_STARTED=str(Path(self.tmp.name) / "s"))
        with self.assertRaisesRegex(ValueError, "longer than"):
            testing.run_process(self.bench, timeout=0.5)


class SpeedTests(TestingBase):
    def run_speed(self, **kwargs):
        return testing.speed_test(self.bench, ["llama-server"], self.variant, self.config, hardware(),
                                  server_factory=FakeServer, peak_factory=FakePeak, **kwargs)

    def test_success_builds_a_full_measurement(self):
        events = []
        result = self.run_speed(progress=events.append)
        record = result["measurement"]
        for key in ["variant_id", "sha256", "fingerprint", "timestamp", "context", "users", "gpu_layers", "gpu_uuid",
                    "threads", "tps", "runtime_build", "raw", "note", "kind", "pp_tps", "ttft_s", "depth",
                    "peak_ram_bytes", "peak_vram_bytes", "estimated_ram_bytes", "estimated_vram_bytes", "settings",
                    "runtime", "id"]:
            self.assertIn(key, record)
        self.assertEqual(record["kind"], "speed_test")
        self.assertEqual(len(record["id"]), 12)
        self.assertEqual(record["tps"], 21.5)
        self.assertEqual(record["pp_tps"], 812.5)
        self.assertEqual(record["depth"], 4096 - 640)
        self.assertEqual(record["runtime"], {"version": "6500", "backend": "CUDA"})
        self.assertEqual(set(record["settings"]), {"flash_attn", "cache_type_k", "cache_type_v", "batch", "ubatch", "n_cpu_moe"})
        self.assertGreaterEqual(result["summary"]["ttft_s"], 0)
        self.assertEqual(result["summary"]["tps"], 21.5)
        self.assertTrue(result["memory"]["within_estimate"])
        self.assertEqual(result["memory"]["estimated_ram_bytes"], allocations(self.variant, 4096, 1, 0)["ram"])
        self.assertTrue(FakeServer.instances[0].stopped)
        self.assertTrue(all(e.keys() >= {"stage", "done", "total", "message"} for e in events))
        # The first-word prompt is realistic (well over a thousand tokens of text) and asks for one token.
        first = FakeServer.instances[0].calls[0]
        self.assertEqual(first["max_tokens"], 1)
        self.assertGreater(len(first["messages"][0]["content"].split()), 900)
        json.dumps(record)

    def test_record_is_accepted_by_matching_speed(self):
        record = self.run_speed()["measurement"]
        match = matching_speed([record], self.variant, hardware(), 4096, 0, None, 8)
        self.assertIs(match, record)

    def test_memory_over_estimate_is_reported(self):
        estimate = allocations(self.variant, 4096, 1, 0)["ram"]
        FakePeak.result = {"peak_ram_bytes": int(estimate * 1.25), "peak_vram_bytes": None, "samples": 3}
        memory = self.run_speed()["memory"]
        self.assertFalse(memory["within_estimate"])
        self.assertIn("RAM by 25%", memory["note"])

    def test_unknown_memory_stays_unknown(self):
        FakePeak.result = {"peak_ram_bytes": None, "peak_vram_bytes": None, "samples": 0}
        self.assertIsNone(self.run_speed()["memory"]["within_estimate"])

    def test_new_allocation_keywords_are_passed_only_when_accepted(self):
        seen = {}

        def new_allocations(variant, context, users, gpu_layers, kv_cache_type="f16", n_cpu_moe=0, unified=False):
            seen.update(kv_cache_type=kv_cache_type, n_cpu_moe=n_cpu_moe)
            return {"ram": 2 * GIB, "vram": 0, "kv_total": 0}
        config = dict(self.config, cache_type_k="q8_0", cache_type_v="q8_0")
        testing.estimate_memory(self.variant, config, hardware(), new_allocations)
        self.assertEqual(seen, {"kv_cache_type": "q8_0", "n_cpu_moe": 0})
        # The merged engine accepts kv_cache_type, so the compressed notes shrink the estimate.
        self.assertEqual(testing.estimate_memory(self.variant, config, hardware(), allocations)[0],
                         allocations(self.variant, 4096, 1, 0, kv_cache_type="q8_0")["ram"])

    def test_server_stopped_when_chat_fails(self):
        class Broken(FakeServer):
            def chat(self, *args, **kwargs):
                raise RuntimeError("connection reset")
        with self.assertRaises(RuntimeError):
            testing.speed_test(self.bench, ["llama-server"], self.variant, self.config, hardware(),
                               server_factory=Broken, peak_factory=FakePeak)
        self.assertTrue(FakeServer.instances[0].stopped)

    def test_requires_a_model_file(self):
        with self.assertRaisesRegex(ValueError, "model file"):
            testing.speed_test(self.bench, ["s"], self.variant, dict(self.config, model_path=None), hardware(),
                               server_factory=FakeServer, peak_factory=FakePeak)


class RunTests(TestingBase):
    def setUp(self):
        super().setUp()
        self.store = Store(Path(self.tmp.name) / "data")
        self.commands = {"llama-server": ["llama-server"], "llama-bench": self.bench}

    def run_kind(self, kind="full", **kwargs):
        return testing.run_tests(self.store, self.variant, self.config, self.commands, kind=kind, hardware=hardware(),
                                 server_factory=FakeServer, peak_factory=FakePeak, **kwargs)

    def test_full_run_saves_measurement(self):
        result = self.run_kind()
        self.assertEqual(result["verdict"], "works")
        self.assertTrue(result["smoke"]["ok"])
        self.assertIn("21.5 tokens per second", result["verdict_text"])
        saved = self.store.get("measurements")
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0]["id"], result["speed"]["measurement"]["id"])

    def test_slow_verdict(self):
        result = self.run_kind(min_tps=30)
        self.assertEqual(result["verdict"], "works_slowly")
        self.assertIn("slower than your target of 30", result["verdict_text"])

    def test_failed_smoke_skips_speed(self):
        FakeServer.fail = "oom"
        result = self.run_kind()
        self.assertEqual(result["verdict"], "failed")
        self.assertIsNone(result["speed"])
        self.assertIn("ran out of memory", result["verdict_text"])
        self.assertIsNone(self.store.get("measurements"))

    def test_smoke_only(self):
        result = self.run_kind("smoke")
        self.assertEqual(result["verdict"], "works")
        self.assertIsNone(result["speed"])

    def test_speed_failure_is_reported_not_saved(self):
        self.env(FAKE_BENCH_NGL_SHIFT="1")
        result = self.run_kind("speed")
        self.assertEqual(result["verdict"], "failed")
        self.assertIn("different settings", result["verdict_text"])
        self.assertIsNone(self.store.get("measurements"))

    def test_missing_runtime_is_plain(self):
        with self.assertRaisesRegex(ValueError, "not installed"):
            testing.run_tests(self.store, self.variant, self.config, {}, hardware=hardware())

    def test_cancel_propagates(self):
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(Cancelled):
            self.run_kind(cancel=cancel)


if __name__ == "__main__":
    unittest.main()
