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
            [int(x) for x in flag_values(argv, "-t", "16")], [{"on": 1, "off": 0, "auto": -1}[x] for x in flag_values(argv, "-fa", "auto")],
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
            "-m", "/models/m.gguf", "-p", "512", "-n", "128", "-ngl", "0", "-r", "2", "-o", "json", "-v"])

    def test_every_setting(self):
        args = tuner.bench_args(config(gpu_layers=32, total_layers=32, threads=6, batch=1024, ubatch=256,
                                       flash_attn="on", cache_type_k="q8_0", cache_type_v="q8_0", n_cpu_moe=4,
                                       mmap=False, device="CUDA0"), n_prompt=256, n_gen=64, depth=2048, repetitions=3)
        self.assertEqual(args, ["-m", "/models/m.gguf", "-p", "256", "-n", "64", "-d", "2048", "-ngl", "33",
                                "-ncmoe", "4", "-t", "6", "-b", "1024", "-ub", "256", "-fa", "on", "-ctk", "q8_0",
                                "-ctv", "q8_0", "-dev", "CUDA0", "-lm", "none", "-r", "3", "-o", "json", "-v"])

    def test_sweep_uses_comma_lists(self):
        args = tuner.bench_args(config(gpu_layers=20, total_layers=32), sweep={
            "threads": [8, 7, 4], "flash_attn": ["on", "off"], "gpu_layers": [20, 32], "cache_type_k": ["f16", "q8_0"]})
        self.assertEqual(args, ["-m", "/models/m.gguf", "-p", "512", "-n", "128", "-ngl", "21,33", "-t", "8,7,4",
                                "-fa", "on,off", "-ctk", "f16,q8_0", "-r", "2", "-o", "json", "-v"])

    def test_flash_attn_auto_is_not_passed_and_cpu_only_hides_gpu(self):
        args = tuner.bench_args(config(flash_attn="auto", gpu_backend="cuda"))
        self.assertNotIn("-fa", args)
        self.assertEqual(args[args.index("-dev") + 1], "none")
        args = tuner.bench_args(config(flash_attn="off"))
        self.assertEqual(args[args.index("-fa") + 1], "off")

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
        confirm = [c for c in bench.calls if c[c.index("-r") + 1] == "5"]
        self.assertEqual(len(confirm), 1)
        # The last runs re-measure the start and the winner with the full conversation length in memory.
        self.assertEqual([c[c.index("-d") + 1] for c in bench.calls[-2:]], ["3456", "3456"])
        self.assertEqual(result["settings"]["depth"], 3456)
        self.assertAlmostEqual(result["improvement"], (50 * 1.1) / (50 * (1 - 0.04 * 9)), places=3)
        self.assertEqual(result["trials"][0]["changes"], {})
        self.assertTrue(all(t["status"] == "ok" for t in result["trials"]))
        self.assertTrue(any("7 CPU threads instead of the default" in n for n in result["notes"]))
        self.assertTrue(any("flash attention" in n and "writing" in n and "% faster" in n for n in result["notes"]))
        # Generation goal never spends time on prompt chunk sizes.
        self.assertFalse(any(t["step"] == "batch" for t in result["trials"]))
        # A thread sweep is one llama-bench process with a comma list.
        self.assertIn("8,7,4,16", [c[c.index("-t") + 1] for c in bench.calls if "-t" in c])
        # The unset thread count is re-measured in its own run during the thread step (baseline + that one), so the
        # sweep is compared with a fresh number, not the first cold run.
        # (The full-length check runs the start once more.)
        self.assertEqual(sum("-t" not in c and c[c.index("-d") + 1] == "1024" for c in bench.calls), 2)

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
            # 22 transformer blocks run as -ngl 23 (the output layer comes too).
            self.assertTrue(all(int(x) <= 23 for x in call[call.index("-ngl") + 1].split(",")))
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

    def test_confirmation_needs_the_noise_margin_too(self):
        clock = Clock()
        # The short runs say 7 threads + flash attention is much faster; the long check says only 1% faster.
        def speed(s, reps, tg, pp):
            return (tg if reps != 5 else 50 * (1 - 0.04 * 9) * 1.01, pp)
        result, _, _ = run(clock=clock, bench=FakeBench(clock, speed=speed))
        self.assertIsNone(result["best"]["threads"])
        self.assertFalse(result["confirmed"])
        # The step notes that claimed gains are gone; only the fallback explanation stays.
        self.assertFalse(any(" made " in n for n in result["notes"]))
        self.assertTrue(any("did not hold up" in n for n in result["notes"]))

    def test_drift_between_runs_raises_the_bar(self):
        clock = Clock()
        calls = {"n": 0}

        # Every process is 8% faster than the one before (machine warming up), whatever the settings.
        def speed(s, reps, tg, pp):
            return (50 * 1.08 ** calls["n"], pp)

        class Drifting(FakeBench):
            def __call__(self, argv, env, timeout, cancel):
                out = super().__call__(argv, env, timeout, cancel)
                calls["n"] += 1
                return out
        result, _, _ = run(clock=clock, bench=Drifting(clock, speed=speed), confirm_repetitions=2)
        self.assertGreater(result["drift"], 0.05)
        # Only the first step can be fooled by drift nobody has seen yet; after that the bar includes it.
        self.assertLessEqual(sum(1 for n in result["notes"] if "made" in n and "Overall" not in n), 1)

    def test_negative_room_never_runs_more_trials(self):
        clock = Clock()
        # Baseline costs 14 s of a 40 s budget and the double-check needs more than what is left: stop, don't run.
        result, bench, clock = run(budget=40, clock=clock, bench=FakeBench(clock, load=10.0))
        self.assertEqual(result["stopped"], "budget")
        self.assertEqual(len(bench.calls), 1)

    def test_zero_second_runs_do_not_divide_by_zero(self):
        clock = Clock()
        bench = FakeBench(clock)

        def instant(argv, env, timeout, cancel):
            out = bench(argv, env, timeout, cancel)
            return {**out, "seconds": 0}
        result, _, _ = run(clock=clock, bench=instant)
        self.assertIn(result["stopped"], {"converged", "budget"})

    def test_baseline_gets_the_full_timeout_and_a_plain_message(self):
        seen = []

        def slow(argv, env, timeout, cancel):
            seen.append(timeout)
            return {"returncode": -9, "stdout": "[", "stderr": "", "seconds": timeout, "timed_out": True}
        with self.assertRaisesRegex(ValueError, "too slow to tune"):
            tuner.tune(["llama-bench"], variant(), config(), HARDWARE, budget_seconds=60, memory_check=lambda c: True,
                       run_bench=slow, timeout=900)
        self.assertEqual(seen, [900])

    def test_result_carries_what_a_tune_measurement_needs(self):
        result, _, _ = run()
        self.assertEqual(result["runtime"], {"version": "b6000", "backend": "cpu"})
        self.assertEqual(result["runtime_build"], "abc123")
        self.assertEqual(result["threads"], 7)
        self.assertEqual(result["bench"], {"n_prompt": 512, "n_gen": 128, "depth": 1024, "repetitions": 2})
        # best_result and baseline were measured again at context - 640, and settings.depth says so.
        self.assertEqual(result["depth"], 3456)
        self.assertEqual(result["settings"], {"flash_attn": "on", "cache_type_k": "f16", "cache_type_v": "f16",
                                              "batch": None, "ubatch": None, "n_cpu_moe": 0, "depth": 3456})
        self.assertEqual(result["search"]["depth"], 1024)
        self.assertGreater(result["search"]["improvement"], 1.0)
        json.dumps(result)

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


class FullDepthTests(unittest.TestCase):
    """The winner is measured again with the user's full conversation length in memory (-d context-640)."""

    def depth_bench(self, clock, long_speed=None, long_fail=None):
        bench = FakeBench(clock)

        def run_it(argv, env, timeout, cancel):
            d = int(argv[argv.index("-d") + 1])
            out = bench(argv, env, timeout, cancel)
            if d > 1024:
                rows = json.loads(out["stdout"])
                for row in rows:
                    row["n_depth"] = d
                    if long_speed:
                        row["avg_ts"] = long_speed(row, row["avg_ts"])
                if long_fail and any(long_fail(r) for r in rows):
                    keep = []
                    for row in rows:
                        if long_fail(row):
                            break
                        keep.append(row)
                    return {**out, "returncode": 1, "stdout": json.dumps(keep)[:-1],
                            "stderr": "llama_bench: error: failed to create context with model '/models/m.gguf'\n"}
                out = {**out, "stdout": json.dumps(rows)}
            return out
        run_it.calls = bench.calls
        return run_it

    def test_full_depth_numbers_replace_the_short_ones(self):
        clock = Clock()
        bench = self.depth_bench(clock, long_speed=lambda row, ts: ts / 2)
        result, _, _ = run(clock=clock, bench=bench)
        self.assertEqual(result["depth"], 3456)
        self.assertAlmostEqual(result["best_result"]["tps"] * 2, result["search"]["best_result"]["tps"], delta=1.0)
        self.assertAlmostEqual(result["baseline"]["tps"] * 2, result["search"]["baseline"]["tps"], delta=1.0)
        self.assertTrue(any("3,456 tokens of conversation" in n for n in result["notes"]))
        self.assertTrue(any(t["step"] == "full_depth" for t in result["trials"]))

    def test_winner_not_faster_at_full_length_keeps_the_start(self):
        clock = Clock()
        # At full length flash attention on is slower than the default, so the tuned settings lose.
        bench = self.depth_bench(clock, long_speed=lambda row, ts: ts * (0.5 if row["flash_attn"] == 1 else 1.0))
        result, _, _ = run(clock=clock, bench=bench)
        self.assertEqual(result["best"]["flash_attn"], "auto")
        self.assertEqual(result["best"]["threads"], None)
        self.assertEqual(result["improvement"], 1.0)
        self.assertTrue(any("were not clearly faster, so your starting settings are kept" in n for n in result["notes"]))
        self.assertEqual(result["settings"]["depth"], 3456)

    def test_winner_that_does_not_fit_at_full_length_keeps_the_start(self):
        clock = Clock()
        # Threads and flash attention already at their best, so the winner differs from the start only in layers.
        base = config(gpu_layers=20, total_layers=32, gpu_backend="cuda", threads=7, flash_attn="on")
        bench = self.depth_bench(clock, long_fail=lambda row: row["n_gpu_layers"] >= 23)
        result, _, _ = run(clock=clock, bench=bench, base=base)
        self.assertEqual(result["best"]["gpu_layers"], 20)
        note = next(n for n in result["notes"] if "did not work with your full conversation length" in n)
        # llama-bench named no cause, but the lighter start loaded in the same run: memory is the likely reason.
        self.assertIn("Probably not enough memory", note)
        self.assertEqual(result["depth"], 3456)

    def long_bench(self, clock, on_long):
        """FakeBench whose full-length runs (-d > 1024) are answered by on_long(argv, out) instead."""
        bench = FakeBench(clock)

        def run_it(argv, env, timeout, cancel):
            out = bench(argv, env, timeout, cancel)
            return on_long(argv, out, timeout) if int(argv[argv.index("-d") + 1]) > 1024 else out
        run_it.calls = bench.calls
        return run_it

    def test_running_out_of_time_at_full_length_keeps_the_winner(self):
        clock = Clock()
        seen = []

        def out_of_time(argv, out, timeout):
            seen.append(timeout)
            return {"returncode": -9, "stdout": "[", "stderr": "", "seconds": timeout, "timed_out": True}
        result, _, _ = run(clock=clock, bench=self.long_bench(clock, out_of_time))
        self.assertEqual((result["best"]["threads"], result["best"]["flash_attn"]), (7, "on"))
        self.assertTrue(result["confirmed"])
        self.assertEqual(result["depth"], 1024)
        self.assertEqual(result["best_result"], result["search"]["best_result"])
        self.assertTrue(any("ran out of time" in n for n in result["notes"]))
        self.assertFalse(any("failed to run" in n for n in result["notes"]))
        self.assertFalse(any("already the fastest" in n for n in result["notes"]))
        self.assertEqual(len(seen), 1)  # the start timed out; the winner was never run (not blamed)

    def test_start_fails_at_full_length_but_the_winner_runs(self):
        clock = Clock()

        def start_fails(argv, out, timeout):
            if "-t" not in argv:  # the start leaves threads to llama.cpp
                return {**out, "returncode": 1, "stdout": "[\n",
                        "stderr": "llama_bench: error: failed to create context with model '/models/m.gguf'\n"}
            return out
        result, _, _ = run(clock=clock, bench=self.long_bench(clock, start_fails))
        self.assertEqual(result["best"]["threads"], 7)
        self.assertEqual(result["depth"], 1024)
        note = next(n for n in result["notes"] if "your starting settings did not run" in n)
        self.assertIn("but the tuned settings did", note)

    def test_nothing_runs_at_full_length_says_the_context_may_not_fit(self):
        clock = Clock()

        def all_fail(argv, out, timeout):
            return {**out, "returncode": 1, "stdout": "[\n",
                    "stderr": "llama_bench: error: failed to create context with model '/models/m.gguf'\n"}
        result, _, _ = run(clock=clock, bench=self.long_bench(clock, all_fail))
        self.assertTrue(any("may not fit in memory" in n for n in result["notes"]))
        self.assertEqual(result["depth"], 1024)
        self.assertEqual(result["best"]["threads"], 7)

    def test_winner_within_noise_at_full_length_keeps_the_start(self):
        clock = Clock()

        def same_speed(argv, out, timeout):
            rows = json.loads(out["stdout"])
            for row in rows:
                row["avg_ts"] = 50.05 if "-t" in argv else 50.0
            return {**out, "stdout": json.dumps(rows)}
        result, _, _ = run(clock=clock, bench=self.long_bench(clock, same_speed))
        self.assertEqual(result["best"]["threads"], None)
        self.assertEqual(result["improvement"], 1.0)
        self.assertTrue(any("not clearly faster" in n for n in result["notes"]))
        self.assertIn("Your starting settings are kept", result["notes"][0])
        self.assertFalse(any("already the fastest" in n for n in result["notes"]))

    def test_step_notes_are_labelled_as_the_short_test(self):
        result, _, _ = run()
        self.assertTrue(any(n.startswith("In the shorter search test: using 7 CPU threads") for n in result["notes"]))

    def test_cancel_during_full_length_keeps_the_short_results(self):
        clock = Clock()

        def cancel_now(argv, out, timeout):
            raise Cancelled()
        result, _, _ = run(clock=clock, bench=self.long_bench(clock, cancel_now))
        self.assertEqual(result["stopped"], "cancelled")
        self.assertEqual(result["depth"], 1024)
        self.assertEqual(result["best"]["threads"], 7)

    def test_each_full_length_run_gets_its_share_of_the_time(self):
        clock = Clock()
        seen = []

        def spy(argv, out, timeout):
            seen.append(timeout)
            return out
        run(clock=clock, bench=self.long_bench(clock, spy), budget=200)
        self.assertEqual(len(seen), 2)
        self.assertLess(seen[0], seen[1] + 60)  # the first run did not get everything that was left

    def test_no_time_left_keeps_the_short_depth_and_says_so(self):
        clock = Clock()
        bench = FakeBench(clock, load=5, per_test=2)
        result, _, _ = run(clock=clock, bench=bench, budget=60)
        self.assertEqual(result["depth"], 1024)
        self.assertEqual(result["settings"]["depth"], 1024)
        self.assertTrue(any("no time left to measure with your full conversation length" in n for n in result["notes"]))
        self.assertFalse(any(c[c.index("-d") + 1] != "1024" for c in bench.calls))

    def test_short_contexts_are_already_measured_at_full_length(self):
        result, bench, _ = run(base=config(context=640))
        self.assertEqual(result["depth"], 0)
        self.assertFalse(any(t["step"] == "full_depth" for t in result["trials"]))
        self.assertEqual(result["settings"]["depth"], 0)
        result, bench, _ = run(base=config(context=1024))
        self.assertEqual(result["depth"], 384)
        self.assertEqual(result["search"]["depth"], 256)

    def test_can_be_switched_off(self):
        result, bench, _ = run(verify_full_depth=False)
        self.assertEqual(result["depth"], 1024)
        self.assertFalse(any(t["step"] == "full_depth" for t in result["trials"]))

    def test_long_contexts_are_capped(self):
        result, bench, _ = run(base=config(context=131072), v=variant(max_context=131072), budget=86400)
        self.assertEqual(result["depth"], tuner.MAX_BENCH_DEPTH)


class FailureTextTests(unittest.TestCase):
    def test_only_stderr_is_read_and_json_stdout_is_ignored(self):
        text = tuner.bench_failure("main: something broke\n", "[\n", 1)
        self.assertIn("something broke", text)
        self.assertNotIn("[", text.split(":", 1)[1])

    def test_no_cause_is_not_called_out_of_memory(self):
        text = tuner.bench_failure("llama_bench: error: failed to load model '/m/x.gguf'\n", "[", 1)
        self.assertEqual(text, tuner.NO_CAUSE)

    def test_llama_cpp_named_cause_wins(self):
        oom = ("llama_model_load: error loading model: unable to allocate CUDA0 buffer\n"
               "llama_bench: error: failed to load model '/m/x.gguf'\n")
        self.assertEqual(tuner.bench_failure(oom, "[", 1), "Ran out of memory with these settings.")
        ctx = ("ggml_backend_cpu_buffer_type_alloc_buffer: failed to allocate buffer of size 68719476768\n"
               "llama_init_from_model: failed to initialize the context: failed to allocate buffer for kv cache\n"
               "llama_bench: error: failed to create context with model '/m/x.gguf'\n")
        self.assertEqual(tuner.bench_failure(ctx, "[", 1), "Ran out of memory with these settings.")

    def test_memory_warnings_that_carry_on_are_not_the_cause(self):
        text = ("warning: failed to allocate 512.00 MiB of pinned memory: out of memory\n"
                "llama_model_load: error loading model: unknown model architecture: 'notarealarch'\n"
                "llama_bench: error: failed to load model '/m/x.gguf'\n")
        self.assertIn("unknown model architecture", tuner.bench_failure(text, "[", 1))
        only_warning = "warning: failed to allocate 512.00 MiB of pinned memory\nllama_bench: error: failed to load model 'x'\n"
        self.assertEqual(tuner.bench_failure(only_warning, "[", 1), tuner.NO_CAUSE)

    def test_deprecation_glued_to_the_next_line_is_split_off(self):
        text = ("DEPRECATED: -mmp and --mmap are deprecated in favour of --load-mode. Please use --load-mode mmap "
                "instead.llama_model_load: error loading model: tensor 'blk.1.ffn_down.weight' data is not within "
                "the file bounds, model is corrupted or incomplete\nllama_bench: error: failed to load model 'x'\n")
        self.assertIn("not within the file bounds", tuner.bench_failure(text, "[", 1))

    def test_missing_file_names_only_the_file(self):
        # Real llama-bench -v for a missing file (the cause line itself only says "failed to load model from").
        text = ("gguf_init_from_file: failed to open GGUF file '/home/me/models/nope.gguf' (No such file or directory)\n"
                "llama_model_load: error loading model: llama_model_loader: failed to load model from /home/me/models/nope.gguf\n"
                "llama_model_load_from_file_impl: failed to load model\n"
                "llama_bench: error: failed to load model '/home/me/models/nope.gguf'\n")
        self.assertEqual(tuner.bench_failure(text, "[\n", 1), "The model file is missing or cannot be read (nope.gguf).")

    def test_quantized_v_cache_needs_flash_attention(self):
        text = ("llama_init_from_model: quantized V cache requires flash_attn to be enabled\n"
                "llama_bench: error: failed to create context with model 'x'\n")
        self.assertIn("flash attention", tuner.bench_failure(text, "[", 1))

    def test_probably_memory_only_after_a_lighter_setting_ran(self):
        clock = Clock()
        bench = FakeBench(clock)

        def no_cause(argv, env, timeout, cancel):
            out = bench(argv, env, timeout, cancel)
            if "-ngl" in argv and "," in argv[argv.index("-ngl") + 1]:
                rows = [r for r in json.loads(out["stdout"]) if r["n_gpu_layers"] < 24]  # -ngl 25 (24 blocks) fails
                return {**out, "returncode": 1, "stdout": json.dumps(rows)[:-1],
                        "stderr": "llama_bench: error: failed to load model '/models/m.gguf'\n"}
            return out
        base = config(gpu_layers=20, total_layers=32, gpu_backend="cuda")
        result, _, _ = run(clock=clock, bench=no_cause, base=base, verify_full_depth=False)
        errors = {t["changes"].get("gpu_layers"): t.get("error") for t in result["trials"] if t["step"] == "gpu_layers"}
        self.assertEqual(errors[24], tuner.PROBABLY_MEMORY)
        self.assertNotIn("/models", errors[24])

    def test_lighter_means_only_less_offload_or_smaller_chunks(self):
        from llm_configurator import launch
        base = launch.normalize(config(gpu_layers=20, total_layers=32, gpu_backend="cuda"))
        more = {**base, "gpu_layers": 22}
        self.assertTrue(tuner._lighter(base, more))
        self.assertFalse(tuner._lighter(more, base))
        # A compressed cache with flash attention on can fail for reasons other than memory.
        q8 = {**base, "cache_type_k": "q8_0", "cache_type_v": "q8_0", "flash_attn": "on"}
        self.assertFalse(tuner._lighter(base, q8))
        self.assertFalse(tuner._lighter({**base, "threads": 4}, {**more, "threads": 8}))
        self.assertTrue(tuner._lighter({**base, "batch": 512, "ubatch": 512}, base))  # unset = 2048/512

    def test_killed_and_bad_alloc_are_worded_honestly(self):
        self.assertIn("possibly because memory ran out", tuner.bench_failure("", "[", -9))
        text = "terminate called after throwing an instance of 'std::bad_alloc'\n  what():  std::bad_alloc\n"
        self.assertEqual(tuner.bench_failure(text, "", -6), "Ran out of memory with these settings.")
        glued = "some error happened\nDEPRECATED: --x is deprecated. Please use --y instead.\n"
        self.assertIn("some error happened", tuner.bench_failure(glued, "", 2))

    def test_first_failure_in_a_run_gets_no_memory_guess(self):
        clock = Clock()
        bench = FakeBench(clock)

        def all_fail(argv, env, timeout, cancel):
            out = bench(argv, env, timeout, cancel)
            if "-ngl" in argv and "," in argv[argv.index("-ngl") + 1]:
                return {**out, "returncode": 1, "stdout": "[\n",
                        "stderr": "llama_bench: error: failed to load model '/models/m.gguf'\n"}
            return out
        base = config(gpu_layers=20, total_layers=32, gpu_backend="cuda")
        result, _, _ = run(clock=clock, bench=all_fail, base=base, verify_full_depth=False)
        errors = [t.get("error") for t in result["trials"] if t["step"] == "gpu_layers" and t["status"] == "failed"]
        self.assertIn(tuner.NO_CAUSE, errors)
        self.assertNotIn(tuner.PROBABLY_MEMORY, errors)


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

    def test_run_process_kills_the_whole_tree(self):
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            pidfile = os.path.join(tmp, "child.pid")
            code = ("import subprocess, sys, time; "
                    f"p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
                    f"open({pidfile!r}, 'w').write(str(p.pid)); time.sleep(60)")
            result = tuner.run_process([sys.executable, "-c", code], timeout=1.5)
            self.assertTrue(result["timed_out"])
            import psutil
            child = int(open(pidfile).read())
            deadline = __import__("time").monotonic() + 5
            while psutil.pid_exists(child) and psutil.Process(child).status() != psutil.STATUS_ZOMBIE \
                    and __import__("time").monotonic() < deadline:
                __import__("time").sleep(0.05)
            self.assertFalse(psutil.pid_exists(child) and psutil.Process(child).status() != psutil.STATUS_ZOMBIE)

    def test_failure_text_comes_from_stderr_without_paths(self):
        # Real llama-bench: stdout is just "[", stderr names the model file.
        text = tuner.bench_failure("llama_bench: error: failed to load model '/home/me/models/x.gguf'\n", "[\n", 1)
        self.assertIn("could not load the model", text)
        self.assertNotIn("/home/me", text)
        cause = ("llama_model_load: error loading model: tensor 'blk.1.ffn_down.weight' data is not within the file "
                 "bounds, model is corrupted or incomplete\nllama_bench: error: failed to load model '/m/x.gguf'")
        self.assertIn("data is not within the file bounds", tuner.bench_failure(cause, "[", 1))
        other = tuner.bench_failure("something odd happened at C:\\Users\\me\\m.gguf\n", "[", 2)
        self.assertNotIn("Users", other)
        self.assertNotIn("[", other.split(":", 1)[1])

    def test_no_shell(self):
        with mock.patch("subprocess.Popen", side_effect=OSError("nope")) as popen:
            with self.assertRaisesRegex(ValueError, "could not be started"):
                tuner.run_process(["llama-bench"], timeout=1)
        self.assertNotIn("shell", popen.call_args.kwargs)
        self.assertIsInstance(popen.call_args.args[0], list)


if __name__ == "__main__":
    unittest.main()
