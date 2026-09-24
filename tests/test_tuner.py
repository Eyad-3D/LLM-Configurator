import itertools
import json
import subprocess
import sys
import threading
import unittest
from unittest import mock

from llm_configurator import tuner
from llm_configurator.domain import GIB, Cancelled, Variant


def variant(**overrides):
    values = dict(id="v1", name="Test 7B", base_repo="org/base", repo="org/gguf", revision="r", base_revision="b",
                  filename="m.gguf", sha256="a" * 64, quant="Q4_K_M", size_bytes=4 * GIB, layers=32, kv_heads=8,
                  head_dim=128, max_context=32768, architecture="llama")
    values.update(overrides)
    return Variant(**values)


MOE = dict(architecture="qwen3_moe", experts=128, active_experts=8, expert_fraction=0.9, layers=48)
HARDWARE = {"cores": 8, "threads": 16, "gpus": [], "ram_available": 64 * GIB}


def config(**overrides):
    return {"model_path": "/models/m.gguf", "context": 4096, **overrides}


def flag_values(argv, flag, default):
    return argv[argv.index(flag) + 1].split(",") if flag in argv else [default]


class FakeBench:
    """Deterministic stand-in for llama-bench: parses comma lists and prints JSON rows per combination.

    Speed surface: writing peaks at 7 threads, flash attention helps, more GPU layers help, fewer CPU
    expert layers help, q8_0 notes cost 2%. Reading prefers big batches and 512 micro-batches."""

    def __init__(self, clock, fail=None, speed=None, load=2.0, per_test=1.0, noise=0.005):
        self.clock, self.fail, self.speed, self.load, self.per_test, self.noise = clock, fail, speed, load, per_test, noise
        self.calls = []

    def surface(self, t, fa, ngl, ncmoe, b, ub, ctk):
        tg = 50 * (1 - 0.04 * abs(t - 7)) * (1.10 if fa == 1 else 1.0) * (1 + 0.1 * ngl) * (1 - 0.03 * ncmoe)
        tg *= 0.98 if ctk == "q8_0" else 1.0
        pp = 500 * (1.0 if fa != 0 else 0.9) * (1 + b / 4096) * (1.1 if ub == 512 else 1.0) * (1 + 0.02 * ngl)
        return tg, pp

    def __call__(self, argv, env, timeout, cancel):
        self.calls.append(argv)
        reps = int(argv[argv.index("-r") + 1])
        combos = list(itertools.product(
            [int(x) for x in flag_values(argv, "-t", "16")], [int(x) for x in flag_values(argv, "-fa", "-1")],
            [int(x) for x in flag_values(argv, "-ngl", "99")], [int(x) for x in flag_values(argv, "-ncmoe", "0")],
            [int(x) for x in flag_values(argv, "-b", "2048")], [int(x) for x in flag_values(argv, "-ub", "512")],
            flag_values(argv, "-ctk", "f16"), flag_values(argv, "-ctv", "f16")))
        rows, elapsed, error = [], self.load, None
        for t, fa, ngl, ncmoe, b, ub, ctk, ctv in combos:
            settings = dict(n_threads=t, flash_attn=fa, n_gpu_layers=ngl, n_cpu_moe=ncmoe, n_batch=b, n_ubatch=ub,
                            type_k=ctk, type_v=ctv)
            if self.fail and self.fail(settings):
                error = "ggml_backend_cuda_buffer_type_alloc_buffer: allocating 2048 MiB on device 0: cudaMalloc failed: out of memory"
                break
            tg, pp = self.surface(t, fa, ngl, ncmoe, b, ub, ctk)
            if self.speed:
                tg, pp = self.speed(settings, reps, tg, pp)
            elapsed += 2 * self.per_test * reps
            for n_prompt, n_gen, ts in [(512, 0, pp), (0, 128, tg)]:
                rows.append({**settings, "n_prompt": n_prompt, "n_gen": n_gen, "n_depth": 1024, "avg_ts": ts,
                             "stddev_ts": ts * self.noise, "build_commit": "abc123", "build_number": 6000})
        self.clock.advance(elapsed)
        return {"returncode": 1 if error else 0, "stdout": json.dumps(rows), "stderr": error or "", "seconds": elapsed}


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def run(base=None, v=None, budget=600, hardware=HARDWARE, **kwargs):
    clock = kwargs.pop("clock", None) or Clock()
    bench = kwargs.pop("bench", None) or FakeBench(clock)
    result = tuner.tune(["llama-bench"], v or variant(), base or config(), hardware, budget_seconds=budget,
                        memory_check=kwargs.pop("memory_check", lambda c: True), run_bench=bench, clock=clock, **kwargs)
    return result, bench, clock


class BenchArgsTests(unittest.TestCase):
    def test_plain_config(self):
        self.assertEqual(tuner.bench_args(config()), [
            "-m", "/models/m.gguf", "-p", "512", "-n", "128", "-ngl", "0", "-r", "2", "-o", "json"])

    def test_every_setting(self):
        args = tuner.bench_args(config(gpu_layers=32, total_layers=32, threads=6, batch=1024, ubatch=256,
                                       flash_attn="on", cache_type_k="q8_0", cache_type_v="q8_0", n_cpu_moe=4,
                                       mmap=False, device="CUDA0"), n_prompt=256, n_gen=64, depth=2048, repetitions=3)
        self.assertEqual(args, ["-m", "/models/m.gguf", "-p", "256", "-n", "64", "-d", "2048", "-ngl", "33",
                                "-ncmoe", "4", "-t", "6", "-b", "1024", "-ub", "256", "-fa", "1", "-ctk", "q8_0",
                                "-ctv", "q8_0", "-dev", "CUDA0", "-mmp", "0", "-r", "3", "-o", "json"])

    def test_sweep_uses_comma_lists(self):
        args = tuner.bench_args(config(gpu_layers=20, total_layers=32), sweep={
            "threads": [8, 7, 4], "flash_attn": ["on", "off"], "gpu_layers": [20, 32], "cache_type_k": ["f16", "q8_0"]})
        self.assertEqual(args, ["-m", "/models/m.gguf", "-p", "512", "-n", "128", "-ngl", "20,33", "-t", "8,7,4",
                                "-fa", "1,0", "-ctk", "f16,q8_0", "-r", "2", "-o", "json"])

    def test_flash_attn_auto_is_not_passed_and_cpu_only_hides_gpu(self):
        args = tuner.bench_args(config(flash_attn="auto", gpu_backend="cuda"))
        self.assertNotIn("-fa", args)
        self.assertEqual(args[args.index("-dev") + 1], "none")
        args = tuner.bench_args(config(flash_attn="off"))
        self.assertEqual(args[args.index("-fa") + 1], "0")

    def test_rejects_bad_sweeps(self):
        with self.assertRaisesRegex(ValueError, "cannot be swept"):
            tuner.bench_args(config(), sweep={"context": [1024]})
        with self.assertRaisesRegex(ValueError, "auto"):
            tuner.bench_args(config(), sweep={"flash_attn": ["auto", "on"]})
        with self.assertRaisesRegex(ValueError, "flash attention"):
            tuner.bench_args(config(cache_type_v="q8_0"), sweep={"flash_attn": ["on", "off"]})
        with self.assertRaisesRegex(ValueError, "ubatch"):
            tuner.bench_args(config(), sweep={"batch": [256], "ubatch": [512]})
        with self.assertRaisesRegex(ValueError, "model file"):
            tuner.bench_args(config(model_path=None))
        with self.assertRaisesRegex(ValueError, "repetitions"):
            tuner.bench_args(config(), repetitions=0)


class ParseTests(unittest.TestCase):
    def test_parse_full_jsonl_and_truncated_output(self):
        rows = [{"n_prompt": 512, "avg_ts": 10.0}, {"n_gen": 128, "avg_ts": 5.0}]
        self.assertEqual(tuner.parse_rows(json.dumps(rows)), rows)
        self.assertEqual(tuner.parse_rows("\n".join(json.dumps(r) for r in rows)), rows)
        cut = json.dumps(rows, indent=2)[:-40]  # crash after the first row
        self.assertEqual(tuner.parse_rows(cut), rows[:1])
        self.assertEqual(tuner.parse_rows("garbage"), [])

    def test_score_goals(self):
        self.assertEqual(tuner.score(10, 100, "generation"), 10)
        self.assertEqual(tuner.score(10, 100, "prompt"), 100)
        self.assertAlmostEqual(tuner.score(10, 100, "balanced"), 10 ** 0.6 * 100 ** 0.4)
        self.assertIsNone(tuner.score(None, 100, "balanced"))


class TuneTests(unittest.TestCase):
    def test_finds_optimum_and_confirms(self):
        result, bench, _ = run()
        self.assertEqual(result["best"]["threads"], 7)
        self.assertEqual(result["best"]["flash_attn"], "on")
        self.assertEqual(result["stopped"], "converged")
        self.assertTrue(result["confirmed"])
        self.assertIn("-r", bench.calls[-1])
        self.assertEqual(bench.calls[-1][bench.calls[-1].index("-r") + 1], "5")
        self.assertAlmostEqual(result["improvement"], (50 * 1.1) / (50 * (1 - 0.04 * 9)), places=3)
        self.assertEqual(result["trials"][0]["changes"], {})
        self.assertTrue(all(t["status"] == "ok" for t in result["trials"]))
        self.assertTrue(any("7 CPU threads instead of the default" in n for n in result["notes"]))
        self.assertTrue(any("flash attention" in n and "writing" in n and "% faster" in n for n in result["notes"]))
        # Generation goal never spends time on prompt chunk sizes.
        self.assertFalse(any(t["step"] == "batch" for t in result["trials"]))
        # A thread sweep is one llama-bench process with a comma list.
        self.assertIn("8,7,4,16", [c[c.index("-t") + 1] for c in bench.calls if "-t" in c])
        # The unset thread count is not re-run on its own; its baseline number is reused.
        self.assertEqual(sum("-t" not in c for c in bench.calls), 1)

    def test_auto_flash_attention_never_goes_in_a_comma_list(self):
        _, bench, _ = run()
        for call in bench.calls:
            if "-fa" in call:
                self.assertNotIn("auto", call[call.index("-fa") + 1])

    def test_noise_small_gains_are_not_kept(self):
        clock = Clock()
        bench = FakeBench(clock, speed=lambda s, r, tg, pp: (50 * (1.01 if s["n_threads"] == 4 else 1.0), pp))
        result, _, _ = run(clock=clock, bench=bench, base=config(threads=8))
        self.assertEqual(result["best"]["threads"], 8)
        self.assertEqual(result["improvement"], 1.0)
        self.assertIn("already the fastest", result["notes"][0])

    def test_prompt_goal_tunes_batch_sizes(self):
        result, _, _ = run(goal="prompt", base=config(batch=512, ubatch=512))
        self.assertEqual((result["best"]["batch"], result["best"]["ubatch"]), (2048, 512))
        self.assertTrue(any("chunks of 2048/512 tokens instead of 512/512" in n for n in result["notes"]))
        self.assertGreater(result["best_result"]["pp_tps"], result["baseline"]["pp_tps"])

    def test_balanced_goal(self):
        result, _, _ = run(goal="balanced")
        self.assertEqual(result["best"]["threads"], 7)
        self.assertGreater(result["improvement"], 1.0)

    def test_respects_budget(self):
        clock = Clock()
        bench = FakeBench(clock, load=5, per_test=2)
        result, bench, clock = run(clock=clock, bench=bench, budget=40)
        self.assertEqual(result["stopped"], "budget")
        spent = clock() - 1000
        biggest_trial = max(t["seconds"] for t in result["trials"])
        self.assertLessEqual(spent, 40 + biggest_trial)
        self.assertGreaterEqual(len(result["trials"]), 1)

    def test_budget_never_runs_over_by_more_than_one_trial_across_budgets(self):
        for budget in [15, 25, 60, 90, 150]:
            clock = Clock()
            result, bench, _ = run(clock=clock, bench=FakeBench(clock), budget=budget, goal="balanced",
                                   allow_kv_compression=True)
            per_trial = max(t["seconds"] for t in result["trials"]) * (5 + 1) / 3  # the confirm run is longest
            self.assertLessEqual(clock() - 1000, budget + per_trial, budget)

    def test_failed_trials_are_recorded_and_search_continues(self):
        clock = Clock()
        bench = FakeBench(clock, fail=lambda s: s["flash_attn"] == 1)
        result, _, _ = run(clock=clock, bench=bench)
        failed = [t for t in result["trials"] if t["status"] == "failed"]
        self.assertTrue(failed)
        self.assertIn("memory", failed[0]["error"])
        self.assertEqual(result["best"]["threads"], 7)
        self.assertEqual(result["best"]["flash_attn"], "auto")
        self.assertTrue(any("failed to run" in n for n in result["notes"]))

    def test_crash_mid_sweep_keeps_earlier_rows(self):
        clock = Clock()
        base = config(gpu_layers=20, total_layers=32, gpu_backend="cuda")
        bench = FakeBench(clock, fail=lambda s: s["n_gpu_layers"] >= 24)
        result, _, _ = run(clock=clock, bench=bench, base=base)
        self.assertEqual(result["best"]["gpu_layers"], 22)
        layer_trials = {t["changes"].get("gpu_layers", 20): t for t in result["trials"] if t["step"] == "gpu_layers"}
        self.assertEqual(layer_trials[24]["status"], "failed")
        self.assertIn("memory", layer_trials[24]["error"])

    def test_unsafe_configs_are_skipped_without_running(self):
        base = config(gpu_layers=20, total_layers=32, gpu_backend="cuda")
        result, bench, _ = run(base=base, memory_check=lambda c: c["gpu_layers"] <= 22)
        self.assertEqual(result["best"]["gpu_layers"], 22)
        skipped = [t for t in result["trials"] if t["status"] == "skipped_memory"]
        self.assertTrue(skipped)
        self.assertEqual(skipped[0]["seconds"], 0.0)
        for call in bench.calls:
            self.assertTrue(all(int(x) <= 22 for x in call[call.index("-ngl") + 1].split(",")))
        self.assertTrue(any("might not fit" in n for n in result["notes"]))

    def test_memory_check_errors_count_as_unsafe(self):
        def check(c):
            if c["threads"] == 16:
                raise RuntimeError("boom")
            return True
        result, _, _ = run(memory_check=check)
        self.assertTrue(any(t["status"] == "skipped_memory" and t["changes"].get("threads") == 16 for t in result["trials"]))

    def test_tight_memory_tries_compressed_notes_with_more_layers(self):
        base = config(gpu_layers=20, total_layers=32, gpu_backend="cuda")
        # f16 notes only fit 20 layers; q8_0 notes leave room for 22.
        check = lambda c: c["gpu_layers"] <= (22 if c["cache_type_k"] == "q8_0" else 20)
        result, _, _ = run(base=base, memory_check=check)
        self.assertEqual(result["best"]["cache_type_k"], "q8_0")
        self.assertEqual(result["best"]["cache_type_v"], "q8_0")
        self.assertEqual(result["best"]["flash_attn"], "on")
        self.assertGreaterEqual(result["best"]["gpu_layers"], 22)
        self.assertTrue(any("short-term notepad" in n for n in result["notes"]))

    def test_compressed_notes_only_when_allowed_or_tight(self):
        result, _, _ = run()
        self.assertFalse(any(t["step"] == "cache" for t in result["trials"]))
        result, _, _ = run(allow_kv_compression=True)
        self.assertTrue(any(t["step"] == "cache" for t in result["trials"]))
        self.assertEqual(result["best"]["cache_type_k"], "f16")  # q8_0 is 2% slower here, so it is not kept

    def test_moe_offload_knob(self):
        base = config(gpu_layers=48, total_layers=48, n_cpu_moe=10, gpu_backend="cuda")
        result, bench, _ = run(base=base, v=variant(**MOE), memory_check=lambda c: c["n_cpu_moe"] >= 6)
        self.assertEqual(result["best"]["n_cpu_moe"], 6)
        self.assertTrue(any("expert weights of 6 layers" in n for n in result["notes"]))
        self.assertTrue(any("-ncmoe" in c and "," in c[c.index("-ncmoe") + 1] for c in bench.calls))
        self.assertFalse(any(t["step"] == "gpu_layers" for t in result["trials"]))  # full offload: nothing to split

    def test_dense_models_skip_moe_knob(self):
        result, bench, _ = run(base=config(gpu_layers=32, total_layers=32, gpu_backend="cuda"))
        self.assertFalse(any("-ncmoe" in c for c in bench.calls))

    def test_cancel_between_trials_returns_best_so_far(self):
        cancel = threading.Event()
        clock = Clock()
        inner = FakeBench(clock)

        def bench(argv, env, timeout, cancel_event):
            result = inner(argv, env, timeout, cancel_event)
            if len(inner.calls) == 2:
                cancel.set()
            return result
        result, _, _ = run(clock=clock, bench=bench, cancel=cancel)
        self.assertEqual(result["stopped"], "cancelled")
        self.assertEqual(len(inner.calls), 2)
        self.assertIn("cancelled", result["notes"][-1])
        self.assertFalse(result["confirmed"])

    def test_cancel_during_a_trial(self):
        cancel = threading.Event()
        clock = Clock()
        inner = FakeBench(clock)

        def bench(argv, env, timeout, cancel_event):
            if inner.calls:
                cancel_event.set()
                raise Cancelled("Cancelled by user")
            return inner(argv, env, timeout, cancel_event)
        result, _, _ = run(clock=clock, bench=bench, cancel=cancel)
        self.assertEqual(result["stopped"], "cancelled")
        self.assertEqual(result["best"], result["best"] | {"threads": None})

    def test_cancel_before_baseline_raises(self):
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(Cancelled):
            run(cancel=cancel)

    def test_baseline_problems_are_plain_errors(self):
        with self.assertRaisesRegex(ValueError, "not enough free memory"):
            run(memory_check=lambda c: False)
        clock = Clock()
        with self.assertRaisesRegex(ValueError, "did not run: Ran out of memory"):
            run(clock=clock, bench=FakeBench(clock, fail=lambda s: True))
        with self.assertRaisesRegex(ValueError, "Download the model"):
            run(base=config(model_path=None))
        with self.assertRaisesRegex(ValueError, "goal"):
            run(goal="fast")
        with self.assertRaisesRegex(ValueError, "budget"):
            run(budget=0)

    def test_winner_that_fails_confirmation_is_dropped(self):
        clock = Clock()
        # Seven threads look fast in short runs but not in the long confirmation run.
        speed = lambda s, reps, tg, pp: (tg * (0.5 if reps == 5 else 1.0), pp)
        result, _, _ = run(clock=clock, bench=FakeBench(clock, speed=speed))
        self.assertEqual(result["best"]["threads"], None)
        self.assertEqual(result["improvement"], 1.0)
        self.assertTrue(any("did not hold up" in n for n in result["notes"]))

    def test_progress_reports(self):
        events = []
        run(progress=events.append)
        self.assertTrue(all(e["stage"] == "tune" and e["total"] == 600 and "message" in e for e in events))
        self.assertEqual(events[-1]["message"], "Tuning finished")

    def test_argv_prefix_and_env(self):
        clock = Clock()
        seen = []
        bench = FakeBench(clock)

        def spy(argv, env, timeout, cancel):
            seen.append((argv, env, timeout))
            return bench(argv, env, timeout, cancel)
        base = config(gpu_layers=32, total_layers=32, gpu_backend="cuda", gpu_uuid="GPU-1")
        tuner.tune([sys.executable, "fake.py", "--as", "bench"], variant(), base, HARDWARE, memory_check=lambda c: True,
                   run_bench=spy, clock=clock)
        argv, env, timeout = seen[0]
        self.assertEqual(argv[:4], [sys.executable, "fake.py", "--as", "bench"])
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "GPU-1")
        self.assertLessEqual(timeout, tuner.DEFAULT_TIMEOUT)
        self.assertIn("-d", argv)
        self.assertEqual(argv[argv.index("-d") + 1], "1024")


class MemoryCheckTests(unittest.TestCase):
    def test_default_memory_check_uses_allocations_and_free_memory(self):
        hw = {"ram_available": 16 * GIB, "gpus": [{"index": 0, "uuid": "GPU-1", "available": 8 * GIB}]}
        from llm_configurator import launch
        with mock.patch("llm_configurator.hardware.scan", return_value=hw) as scan:
            check = tuner.default_memory_check(variant())
            cpu = launch.normalize(config())
            self.assertTrue(check(cpu))
            self.assertTrue(check(launch.normalize(config(gpu_layers=32, total_layers=32, gpu_uuid="GPU-1"))))
            self.assertFalse(check(launch.normalize(config(context=131072))))
            scan.assert_called_once_with(False)
        with mock.patch("llm_configurator.hardware.scan", return_value={**hw, "gpus": []}):
            check = tuner.default_memory_check(variant())
            self.assertFalse(check(launch.normalize(config(gpu_layers=10, total_layers=32))))

    def test_default_memory_check_passes_v04_arguments_when_supported(self):
        calls = []

        def allocations(v, context, users, gpu_layers, kv_cache_type="f16", n_cpu_moe=0, unified=False):
            calls.append((kv_cache_type, n_cpu_moe, unified))
            return {"ram": GIB, "vram": 0, "kv_total": 0}
        from llm_configurator import launch
        with mock.patch("llm_configurator.hardware.scan", return_value={"ram_available": 8 * GIB, "unified_memory": True}), \
                mock.patch("llm_configurator.engine.allocations", allocations):
            check = tuner.default_memory_check(variant(**MOE))
            self.assertTrue(check(launch.normalize(config(cache_type_k="q8_0", cache_type_v="q8_0", n_cpu_moe=3))))
        self.assertEqual(calls, [("q8_0", 3, True)])


class ProcessTests(unittest.TestCase):
    def test_run_process_collects_output(self):
        result = tuner.run_process([sys.executable, "-c", "print('[]')"], timeout=30)
        self.assertEqual((result["returncode"], result["stdout"].strip(), result["timed_out"]), (0, "[]", False))

    def test_run_process_kills_on_cancel(self):
        cancel = threading.Event()
        timer = threading.Timer(0.3, cancel.set)
        timer.start()
        with self.assertRaises(Cancelled):
            tuner.run_process([sys.executable, "-c", "import time; time.sleep(30)"], timeout=60, cancel=cancel)
        timer.cancel()

    def test_run_process_times_out(self):
        result = tuner.run_process([sys.executable, "-c", "import time; print('[', flush=True); time.sleep(30)"],
                                   timeout=0.5)
        self.assertTrue(result["timed_out"])
        self.assertIn("ran out of time", tuner._plain_failure(result))

    def test_no_shell(self):
        with mock.patch("subprocess.Popen", side_effect=OSError("nope")) as popen:
            with self.assertRaisesRegex(ValueError, "could not be started"):
                tuner.run_process(["llama-bench"], timeout=1)
        self.assertNotIn("shell", popen.call_args.kwargs)
        self.assertIsInstance(popen.call_args.args[0], list)


if __name__ == "__main__":
    unittest.main()
