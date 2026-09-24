import copy
import math
from dataclasses import replace
import unittest

from llm_configurator.catalogue import demo_variants
from llm_configurator.domain import GIB, Requirements, now
from llm_configurator.engine import allocations, maximum_context, recommend


def hardware(ram=24, vram=8):
    return {"timestamp": now(), "fingerprint": "test-machine", "ram_available": int(ram * GIB), "ram_total": 32 * GIB,
            "cores": 8, "threads": 16, "gpus": [{"index": 0, "uuid": "GPU-test", "available": int(vram * GIB), "total": int(vram * GIB)}] if vram else [],
            "processes": [{"pid": 123, "reclaimable": 8 * GIB}]}


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.model = demo_variants()[0]

    def test_kv_cache_has_k_and_v_and_all_sequences(self):
        result = allocations(self.model, 8192, 2, 0)
        expected = 2 * 32 * 4 * 128 * 2 * 8192 * 2
        self.assertEqual(result["kv_total"], expected)
        self.assertEqual(result["vram"], 0)

    def test_concurrency_increases_memory(self):
        self.assertGreater(allocations(self.model, 8192, 4, 16)["vram"], allocations(self.model, 8192, 1, 16)["vram"])

    def test_larger_context_increases_both_memory_pools_when_split(self):
        short, long = [allocations(self.model, c, 1, 16) for c in [4096, 16384]]
        self.assertGreater(long["ram"], short["ram"])
        self.assertGreater(long["vram"], short["vram"])

    def test_cpu_only_does_not_require_vram(self):
        report = recommend([self.model], hardware(vram=0), Requirements())
        self.assertTrue(report["candidates"])
        self.assertTrue(all(c["mode"] == "cpu" for c in report["candidates"]))

    def test_not_enough_ram_even_when_total_is_large(self):
        report = recommend([self.model], hardware(ram=1, vram=0), Requirements())
        self.assertEqual(report["candidates"], [])

    def test_max_context_is_bounded_by_model_limit(self):
        self.assertEqual(maximum_context(self.model, 1, 0, 1000 * GIB, 0), self.model.max_context)

    def test_context_over_limit_rejected(self):
        report = recommend([self.model], hardware(), Requirements(context=65536))
        self.assertEqual(report["rejected"]["context"], 1)

    def test_selected_apps_only_improve_hypothetical_scenario(self):
        report = recommend([self.model], hardware(ram=2, vram=0), Requirements(reclaim_pids=[123]))
        self.assertTrue(report["candidates"])
        self.assertTrue(all(c["scenario"] == "after_closing" for c in report["candidates"]))
        self.assertEqual(report["reclaim_estimate_bytes"], 8 * GIB)

    def test_stale_pid_does_not_free_memory(self):
        report = recommend([self.model], hardware(ram=2, vram=0), Requirements(reclaim_pids=[999]))
        self.assertEqual(report["candidates"], [])

    def test_unknown_speed_cannot_pass_strict_filter(self):
        report = recommend([self.model], hardware(), Requirements(strict_speed=True))
        self.assertEqual(report["candidates"], [])

    def test_measurement_requires_same_hash_hardware_context_and_users(self):
        model = replace(self.model, sha256="abc", demo=False)
        record = {"variant_id": model.id, "sha256": "abc", "fingerprint": "test-machine", "context": 8192,
                  "gpu_layers": 0, "gpu_uuid": None, "threads": 8, "users": 1, "tps": 25, "timestamp": now()}
        good = recommend([model], hardware(vram=0), Requirements(strict_speed=True), [record])
        self.assertEqual(len(good["candidates"]), 1)
        self.assertEqual(good["candidates"][0]["tps"], 25)
        for field, bad in [("sha256", "other"), ("fingerprint", "other"), ("context", 2048), ("users", 2), ("timestamp", "2000-01-01T00:00:00+00:00")]:
            modified = {**record, field: bad}
            with self.subTest(field=field):
                self.assertEqual(recommend([model], hardware(vram=0), Requirements(strict_speed=True), [modified])["candidates"], [])

    def test_quality_is_workload_specific(self):
        one = replace(self.model, name="A", scores={"general": 70, "coding": 20}, score_version="4.3")
        two = replace(self.model, id="other", base_repo="test/model-b", name="B", scores={"general": 30, "coding": 80}, score_version="4.3")
        self.assertEqual(recommend([one, two], hardware(), Requirements(workload="coding"))["candidates"][0]["name"], "B")
        self.assertEqual(recommend([one, two], hardware(), Requirements(workload="general"))["candidates"][0]["name"], "A")

    def test_versions_not_numerically_compared(self):
        one = replace(self.model, scores={"general": 90}, score_version="3.0")
        two = replace(self.model, id="other", scores={"general": 20}, score_version="4.3")
        self.assertFalse(recommend([one, two], hardware(), Requirements())["quality_comparable"])

    def test_requirements_reject_nan_negative_and_fractional_context(self):
        for change in [{"min_tps": float("nan")}, {"reserve_gib": -1}, {"users": 0}, {"context": 8192.5}, {"workload": "images"}, {"reclaim_pids": ["123"]}]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                Requirements(**change)


if __name__ == "__main__":
    unittest.main()


# ---- v0.4: compressed notes, MoE offload, unified memory, evidence ladder and verdicts ----
import time
from datetime import datetime, timedelta, timezone

from llm_configurator.domain import KV_BYTES_PER_ELEMENT
from llm_configurator.engine import interpolated_speed, placements, verdict
from llm_configurator.launch import from_candidate, server_args


def real(model, **changes):
    return replace(model, demo=False, sha256=changes.pop("sha256", "hash"), **changes)


def moe_model():
    # 30B-A3B-like: 48 layers, 128 experts with 8 active, 90% of bytes in expert tensors.
    return real(demo_variants()[0], id="moe", base_repo="test/moe", name="MoE", architecture="qwen3_moe",
                size_bytes=int(18 * GIB), layers=48, kv_heads=4, head_dim=128, max_context=65536,
                experts=128, active_experts=8, expert_fraction=0.9)


def record(model, **changes):
    base = {"variant_id": model.id, "sha256": model.sha256, "fingerprint": "test-machine", "context": 8192,
            "gpu_layers": 0, "gpu_uuid": None, "threads": 8, "users": 1, "tps": 25, "timestamp": now()}
    return {**base, **changes}


class CompressedNotesTests(unittest.TestCase):
    def setUp(self):
        self.model = demo_variants()[0]

    def test_kv_types_scale_by_bytes_per_element(self):
        f16 = allocations(self.model, 8192, 1, 0)["kv_total"]
        for kind, size in KV_BYTES_PER_ELEMENT.items():
            with self.subTest(kind=kind):
                self.assertEqual(allocations(self.model, 8192, 1, 0, kind)["kv_total"], math.ceil(f16 * size / 2))
        self.assertLess(allocations(self.model, 8192, 1, 0, "q4_0")["ram"], allocations(self.model, 8192, 1, 0)["ram"])

    def test_bad_kv_type_and_expert_layers_rejected(self):
        with self.assertRaises(ValueError):
            allocations(self.model, 8192, 1, 0, "q2")
        with self.assertRaises(ValueError):
            allocations(self.model, 8192, 1, 0, "f16", self.model.layers + 1)

    def test_sliding_window_layers_cap_their_memory(self):
        sliding = replace(self.model, sliding_window=1024, sliding_layers=24)
        short = allocations(sliding, 1024, 1, 0)["kv_total"]
        self.assertEqual(short, allocations(self.model, 1024, 1, 0)["kv_total"])  # below the window: no saving
        per_token_layer = 2 * 4 * 128 * 2
        expected = per_token_layer * (8 * 65536 + 24 * (1024 + 512))
        self.assertEqual(allocations(sliding, 65536, 1, 0)["kv_total"], expected)
        self.assertEqual(allocations(sliding, 65536, 2, 0)["kv_total"], per_token_layer * (8 * 65536 * 2 + 24 * (2048 + 512)))

    def test_user_choice_is_respected_and_ids_are_marked(self):
        report = recommend([self.model], hardware(), Requirements(kv_cache_type="q8_0"))
        self.assertTrue(report["candidates"])
        for c in report["candidates"]:
            self.assertEqual(c["kv_cache_type"], "q8_0")
            self.assertTrue(c["id"].endswith("|kv:q8_0"))
            self.assertEqual((c["launch"]["cache_type_k"], c["launch"]["cache_type_v"], c["launch"]["flash_attn"]), ("q8_0", "q8_0", "on"))
        self.assertIn("8-bit compression (q8_0)", " ".join(report["notes"]))
        f16 = recommend([self.model], hardware(), Requirements())
        self.assertTrue(all(c["kv_cache_type"] == "f16" for c in f16["candidates"]))
        self.assertIn("full precision (f16)", " ".join(f16["notes"]))

    def test_note_suggests_compressed_notes_only_when_it_would_help(self):
        big_context = replace(self.model, name="Long", max_context=131072)
        use = lambda kind: allocations(big_context, 131072, 1, 0, kind)["ram"]
        ram = (use("q8_0") + use("f16")) / 2 / GIB + 2  # between the two, plus the default reserve
        hw = hardware(ram=ram, vram=0)
        report = recommend([big_context], hw, Requirements(context=131072))
        self.assertEqual(report["candidates"], [])
        self.assertTrue(any("Long would fit with compressed conversation memory" in n for n in report["notes"]))
        self.assertTrue(recommend([big_context], hw, Requirements(context=131072, kv_cache_type="q8_0"))["candidates"])
        hopeless = recommend([big_context], hardware(ram=1, vram=0), Requirements(context=131072))
        self.assertFalse(any("compressed conversation memory" in n for n in hopeless["notes"]))


class ExpertOffloadTests(unittest.TestCase):
    def setUp(self):
        self.model = moe_model()

    def test_n_cpu_moe_moves_only_expert_bytes_of_gpu_layers(self):
        full = allocations(self.model, 8192, 1, 48)
        some = allocations(self.model, 8192, 1, 48, "f16", 12)
        expert_layer = self.model.size_bytes * 1.10 * 0.9 / 48
        self.assertAlmostEqual(full["vram"] - some["vram"], 12 * expert_layer, delta=2)
        self.assertAlmostEqual(some["ram"] - full["ram"], 12 * expert_layer, delta=2)
        self.assertEqual(some["kv_total"], full["kv_total"])
        self.assertEqual(some["kv_vram"], full["kv_vram"])  # attention and its notes stay on the GPU
        # llama.cpp counts --n-cpu-moe from layer 0; layers already on the CPU do not move twice.
        split = allocations(self.model, 8192, 1, 40)
        self.assertEqual(allocations(self.model, 8192, 1, 40, "f16", 8)["vram"], split["vram"])
        self.assertLess(allocations(self.model, 8192, 1, 40, "f16", 12)["vram"], split["vram"])

    def test_dense_models_ignore_n_cpu_moe(self):
        dense = real(demo_variants()[0])
        self.assertEqual(allocations(dense, 8192, 1, 32, "f16", 8), allocations(dense, 8192, 1, 32))

    def test_recommend_adds_smallest_fitting_expert_offload(self):
        hw = hardware(ram=64, vram=8)
        report = recommend([self.model], hw, Requirements())
        offload = [c for c in report["candidates"] if c["n_cpu_moe"]]
        self.assertTrue(offload)
        c = offload[0]
        self.assertEqual((c["mode"], c["gpu_layers"]), ("split", 48))
        self.assertTrue(c["id"].endswith(f"|moe:{c['n_cpu_moe']}"))
        gpu_budget = 8 * GIB - 0.5 * GIB
        self.assertLessEqual(c["vram_bytes"], gpu_budget)
        self.assertGreater(allocations(self.model, c["context"], 1, 48, "f16", c["n_cpu_moe"] - 1)["vram"], gpu_budget)
        config = from_candidate(c, "/m/moe.gguf", hw)
        self.assertEqual(config["n_cpu_moe"], c["n_cpu_moe"])
        self.assertIn("--n-cpu-moe", server_args(config))
        self.assertTrue(any("Experts on CPU" in n for n in report["notes"]))

    def test_no_expert_offload_when_everything_fits_or_no_gpu(self):
        self.assertFalse(any(c["n_cpu_moe"] for c in recommend([self.model], hardware(ram=64, vram=48), Requirements())["candidates"]))
        self.assertFalse(any(c["n_cpu_moe"] for c in recommend([self.model], hardware(ram=64, vram=0), Requirements())["candidates"]))

    def test_binary_search_matches_exhaustive_scan(self):
        import random
        rng = random.Random(7)
        models = [self.model, real(demo_variants()[4]), replace(self.model, sliding_window=4096, sliding_layers=36)]
        for _ in range(300):
            model = rng.choice(models)
            budget, gpu_budget = rng.uniform(1, 40) * GIB, rng.uniform(0, 24) * GIB
            kind, unified, context = rng.choice(list(KV_BYTES_PER_ELEMENT)), rng.random() < 0.3, rng.choice([2048, 8192, 32768])
            if unified:
                gpu_budget = min(gpu_budget, budget)
            fast = placements(model, context, 1, budget, gpu_budget, True, kind, unified)
            fits = [(g, allocations(model, context, 1, g, kind, 0, unified)) for g in range(model.layers + 1)]
            fits = [g for g, m in fits if m["ram"] <= budget and m["vram"] <= gpu_budget]
            expected = {0} & set(fits) | {model.layers} & set(fits) | ({max(g for g in fits if 0 < g < model.layers)} if any(0 < g < model.layers for g in fits) else set())
            with self.subTest(model=model.id, budget=budget, gpu=gpu_budget, kind=kind, unified=unified):
                self.assertEqual({g for g, k, _ in fast if not k}, expected)
                offload = [k for g, k, _ in fast if k]
                if model.moe and model.layers not in fits:
                    ok = [k for k in range(1, model.layers + 1) if (m := allocations(model, context, 1, model.layers, kind, k, unified))["ram"] <= budget and m["vram"] <= gpu_budget]
                    vram_ok = [k for k in range(1, model.layers + 1) if allocations(model, context, 1, model.layers, kind, k, unified)["vram"] <= gpu_budget]
                    self.assertEqual(offload, [min(vram_ok)] if vram_ok and min(vram_ok) in ok else [])
                else:
                    self.assertEqual(offload, [])


class UnifiedMemoryTests(unittest.TestCase):
    def setUp(self):
        self.model = real(demo_variants()[4])

    def apple(self, ram=16, limit=12, available=None):
        hw = hardware(ram=ram, vram=0)
        hw["unified_memory"] = True
        hw["gpus"] = [{"index": 0, "uuid": "apple-m2", "name": "Apple M2", "backend": "metal", "unified": True,
                       "total": int(limit * GIB), "available": int((available if available is not None else min(limit, ram)) * GIB)}]
        return hw

    def test_one_pool_is_never_counted_twice(self):
        split = allocations(self.model, 8192, 1, 32, unified=True)
        discrete = allocations(self.model, 8192, 1, 32)
        self.assertTrue(split["unified"])
        self.assertEqual(split["total"], split["ram"])
        self.assertLessEqual(split["vram"], split["ram"])
        self.assertLess(split["total"], discrete["total"])  # no staging copy
        self.assertEqual(discrete["total"], discrete["ram"] + discrete["vram"])

    def test_gpu_budget_never_exceeds_the_shared_ram(self):
        report = recommend([self.model], self.apple(ram=16, limit=48), Requirements())
        for c in report["candidates"]:
            self.assertTrue(c["unified_memory"])
            self.assertLessEqual(c["ram_bytes"], 14 * GIB)
            self.assertLessEqual(c["vram_bytes"], c["ram_bytes"])
            self.assertEqual(c["memory_total_bytes"], c["ram_bytes"])
        self.assertTrue(any(c["mode"] == "gpu" for c in report["candidates"]))
        self.assertIn("shares main memory", " ".join(report["notes"]))
        gpu = next(c for c in report["candidates"] if c["mode"] == "gpu")
        self.assertEqual(from_candidate(gpu, "/m/x.gguf", self.apple())["gpu_backend"], "metal")

    def test_working_set_limit_forces_a_split(self):
        need = allocations(self.model, 8192, 1, 32, unified=True)["vram"] / GIB
        report = recommend([self.model], self.apple(ram=64, limit=need * 0.6 + 0.5), Requirements(context=8192))
        modes = {c["mode"] for c in report["candidates"] if c["context"] == 8192}
        self.assertNotIn("gpu", modes)
        self.assertIn("split", modes)


class UnknownGpuTests(unittest.TestCase):
    def test_gpu_without_free_memory_is_skipped_safely(self):
        hw = hardware(vram=8)
        hw["gpus"][0].update(available=None, name="Intel Arc")
        report = recommend([demo_variants()[0]], hw, Requirements())
        self.assertTrue(report["candidates"])
        self.assertTrue(all(c["mode"] == "cpu" and c["gpu_index"] is None for c in report["candidates"]))
        self.assertTrue(any("Intel Arc does not report its free memory" in n for n in report["notes"]))

    def test_gpu_with_missing_key_is_skipped(self):
        hw = hardware(vram=8)
        del hw["gpus"][0]["available"]
        self.assertTrue(all(c["mode"] == "cpu" for c in recommend([demo_variants()[0]], hw, Requirements())["candidates"]))


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.model = real(demo_variants()[0])
        self.hw = hardware(vram=0)

    def pick(self, report, context=8192):
        return next(c for c in report["candidates"] if c["context"] == context and c["mode"] == "cpu")

    def test_interpolation_between_two_contexts_is_labelled(self):
        records = [record(self.model, context=4096, tps=30), record(self.model, context=16384, tps=20, id="b")]
        c = self.pick(recommend([self.model], self.hw, Requirements(context=8192), records))
        self.assertEqual(c["evidence"], "interpolated")
        self.assertIsNone(c["tps"])
        self.assertIsNone(c["speed_meets_target"])
        self.assertEqual(c["verdict"], "unknown")
        self.assertAlmostEqual(c["speed_interpolated"]["tps"], 30 - 10 * (4096 / 12288), places=2)
        self.assertEqual(c["speed_interpolated"]["measurement_ids"], ["b"])
        self.assertIn("Not tested at this length yet", c["verdict_text"])
        self.assertIn("Interpolated", c["speed_evidence"])

    def test_single_point_scaling_is_conservative_and_bounded(self):
        records = [record(self.model, context=4096, tps=30)]
        up = interpolated_speed(records, self.model, self.hw, 8192, 0, None, 8)
        self.assertLess(up["tps"], 30 * 0.9 + 1e-9)
        self.assertIsNone(interpolated_speed(records, self.model, self.hw, 32768, 0, None, 8))  # > 4x
        down = interpolated_speed([record(self.model, context=16384, tps=18)], self.model, self.hw, 8192, 0, None, 8)
        self.assertEqual(down["tps"], 18)

    def test_interpolation_needs_same_placement_and_settings(self):
        for change in [{"gpu_layers": 4}, {"fingerprint": "other"}, {"sha256": "x"}, {"threads": 4},
                       {"settings": {"cache_type_k": "q8_0"}}, {"timestamp": "2000-01-01T00:00:00+00:00"}]:
            with self.subTest(change=change):
                self.assertIsNone(interpolated_speed([record(self.model, context=4096, **change)], self.model, self.hw, 8192, 0, None, 8))

    def test_old_measurements_without_settings_still_match(self):
        old = record(self.model)
        self.assertNotIn("settings", old)
        c = self.pick(recommend([self.model], self.hw, Requirements(), [old]))
        self.assertEqual((c["evidence"], c["tps"], c["verdict"]), ("measured", 25, "runs_well"))
        explicit = record(self.model, settings={"cache_type_k": "f16", "n_cpu_moe": 0}, kind="speed_test")
        self.assertEqual(self.pick(recommend([self.model], self.hw, Requirements(), [explicit]))["evidence"], "measured")
        compressed = record(self.model, settings={"cache_type_k": "q8_0"})
        self.assertNotEqual(self.pick(recommend([self.model], self.hw, Requirements(), [compressed]))["evidence"], "measured")
        self.assertEqual(self.pick(recommend([self.model], self.hw, Requirements(kv_cache_type="q8_0"), [compressed]))["evidence"], "measured")

    def tune(self, **changes):
        base = {"variant_id": self.model.id, "sha256": self.model.sha256, "fingerprint": "test-machine", "timestamp": now(),
                "best": {"context": 8192, "gpu_layers": 0, "parallel": 1, "threads": 6, "batch": 1024, "flash_attn": "on",
                         "cache_type_k": "f16", "n_cpu_moe": 0},
                "best_result": {"tps": 40, "pp_tps": 300}, "improvement": 1.2, "stopped": "converged"}
        return {**base, **changes}

    def test_measurement_beats_tuned_and_tuned_beats_everything_else(self):
        tuned = [self.tune()]
        c = self.pick(recommend([self.model], self.hw, Requirements(), [record(self.model)], tuned=tuned))
        self.assertEqual((c["evidence"], c["tps"]), ("measured", 25))
        self.assertEqual(c["tuned"]["tps"], 40)
        c = self.pick(recommend([self.model], self.hw, Requirements(), [record(self.model, context=4096)], tuned=tuned,
                                community=[{"x": 1}], community_evidence=lambda *a: {"median_tps": 99, "n": 3}))
        self.assertEqual((c["evidence"], c["tps"], c["verdict"]), ("tuned", 40, "runs_well"))
        self.assertIn("Tuned and tested on this computer", c["verdict_text"])
        self.assertEqual((c["launch"]["threads"], c["launch"]["batch"], c["launch"]["flash_attn"]), (6, 1024, "on"))
        self.assertEqual(c["threads"], 6)

    def test_tuned_results_for_other_placements_or_machines_do_not_count(self):
        for change in [{"fingerprint": "other"}, {"variant_id": "x"}, {"timestamp": "2000-01-01T00:00:00+00:00"},
                       {"best": {"context": 4096, "gpu_layers": 0}}, {"best": {"context": 8192, "gpu_layers": 3}},
                       {"best_result": {"tps": float("nan")}}]:
            with self.subTest(change=change):
                self.assertIsNone(self.pick(recommend([self.model], self.hw, Requirements(), tuned=[self.tune(**change)]))["tuned"])

    def test_interpolated_beats_community_beats_estimate(self):
        from test_calibration import fixture
        calibration = fixture(self.hw)
        crowd = lambda records, variant, hardware, config: {"median_tps": 33.3, "n": 4, "similar": "same_cpu", "range": [30, 36]}
        c = self.pick(recommend([self.model], self.hw, Requirements(), [record(self.model, context=4096)], calibration,
                                community=[{"r": 1}], community_evidence=crowd))
        self.assertEqual(c["evidence"], "interpolated")
        c = self.pick(recommend([self.model], self.hw, Requirements(), [], calibration, community={"records": [{"r": 1}]},
                                community_evidence=crowd))
        self.assertEqual((c["evidence"], c["verdict"], c["tps"]), ("community", "unknown", None))
        self.assertEqual(c["community"]["n"], 4)
        self.assertIn("4 similar computers report about 33", c["verdict_text"])
        c = self.pick(recommend([self.model], self.hw, Requirements(), [], calibration, community=[{"r": 1}],
                                community_evidence=lambda *a: None))
        self.assertEqual(c["evidence"], "estimated")
        self.assertIn("rough estimate", c["verdict_text"])
        self.assertEqual(self.pick(recommend([self.model], self.hw, Requirements()))["evidence"], "none")

    def test_community_module_is_optional_and_malformed_answers_ignored(self):
        c = self.pick(recommend([self.model], self.hw, Requirements(), community=[{"r": 1}]))
        self.assertIn(c["evidence"], {"community", "none"})
        for bad in [{"median_tps": -1, "n": 2}, {"median_tps": 10, "n": 0}, "nope"]:
            c = self.pick(recommend([self.model], self.hw, Requirements(), community=[{"r": 1}], community_evidence=lambda *a, b=bad: b))
            self.assertEqual(c["evidence"], "none")

    def test_community_evidence_receives_launch_config(self):
        seen = []
        recommend([self.model], self.hw, Requirements(), community=[{"r": 1}],
                  community_evidence=lambda records, variant, hardware, config: seen.append(config))
        self.assertTrue(seen)
        self.assertEqual(seen[0]["cache_type_k"], "f16")
        self.assertIn("gpu_layers", seen[0])


class VerdictTests(unittest.TestCase):
    def test_thresholds(self):
        self.assertEqual(verdict("measured", 15, 34)[0], "runs_well")
        self.assertIn("comfortably above your 15", verdict("measured", 15, 34)[1])
        self.assertEqual(verdict("measured", 15, 15), ("runs_well", "Tested on this computer: about 15 tokens per second — above your 15."))
        self.assertEqual(verdict("measured", 15, 7.5)[0], "runs_slowly")
        self.assertEqual(verdict("measured", 15, 7.4)[0], "too_slow")
        self.assertEqual(verdict("tuned", 0, 1)[0], "runs_well")

    def test_estimates_never_run_well(self):
        estimate = {"available": True, "low_tps": 90, "high_tps": 200, "target_status": "likely_meets"}
        self.assertEqual(verdict("estimated", 15, speed_estimate=estimate), ("unknown", "Not tested yet; a rough estimate says 90–200 tokens per second — likely fast enough."))
        self.assertEqual(verdict("interpolated", 15, interpolated={"tps": 99})[0], "unknown")
        self.assertEqual(verdict("community", 15, community={"median_tps": 99, "n": 1})[0], "unknown")
        self.assertEqual(verdict("none", 15)[0], "unknown")

    def test_slow_verified_configs_are_shown_unless_strict(self):
        model = real(demo_variants()[0])
        slow = [record(model, tps=5)]
        report = recommend([model], hardware(vram=0), Requirements(), slow)
        c = next(c for c in report["candidates"] if c["context"] == 8192)
        self.assertEqual((c["verdict"], c["speed_meets_target"]), ("too_slow", False))
        self.assertEqual(recommend([model], hardware(vram=0), Requirements(strict_speed=True), slow)["candidates"], [])

    def test_balanced_order_puts_verified_slow_below_untested(self):
        a = real(demo_variants()[0], id="a", base_repo="test/a", sha256="a")
        b = real(demo_variants()[0], id="b", base_repo="test/b", sha256="b")
        report = recommend([a, b], hardware(vram=0), Requirements(), [record(a, tps=5)])
        self.assertEqual(report["candidates"][0]["variant_id"], "b")


class CandidateShapeTests(unittest.TestCase):
    def test_ids_are_unique_and_launch_fragments_valid(self):
        models = [real(v, sha256=v.id) for v in demo_variants()] + [moe_model()]
        hw = hardware(ram=64, vram=12)
        for kind in KV_BYTES_PER_ELEMENT:
            report = recommend(models, hw, Requirements(context=4096, kv_cache_type=kind, reclaim_pids=[123]))
            ids = [c["id"] for c in report["candidates"]]
            self.assertEqual(len(ids), len(set(ids)))
            for c in report["candidates"]:
                config = from_candidate(c, "/m/x.gguf", hw)
                self.assertEqual((config["gpu_layers"], config["context"], config["cache_type_v"]), (c["gpu_layers"], c["context"], kind))
                for key in ["verdict", "verdict_text", "evidence", "community", "tuned", "unified_memory", "n_cpu_moe"]:
                    self.assertIn(key, c)

    def test_cpu_candidates_hide_known_gpu_backends(self):
        hw = hardware(ram=64, vram=1)
        hw["gpus"][0]["backend"] = "cuda"
        c = next(c for c in recommend([demo_variants()[0]], hw, Requirements())["candidates"] if c["mode"] == "cpu")
        self.assertIn("-dev", server_args(from_candidate(c, "/m/x.gguf", hw)))


class PerformanceTests(unittest.TestCase):
    def test_recommend_is_fast_for_a_large_catalogue(self):
        from test_calibration import fixture
        base = demo_variants()
        models = []
        for i in range(50):
            for j, quant in enumerate(["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "IQ4_XS"]):
                template = moe_model() if i % 5 == 0 else real(base[i % len(base)])
                models.append(replace(template, id=f"m{i}:{quant}", base_repo=f"test/m{i}", quant=quant, sha256=f"h{i}{j}",
                                      max_context=65536, size_bytes=int(template.size_bytes * (0.8 + 0.1 * j))))
        hw = hardware(ram=64, vram=12)
        records = [record(m, context=4096, tps=20) for m in models[:40]]
        start = time.perf_counter()
        report = recommend(models, hw, Requirements(context=4096, reclaim_pids=[123]), records, fixture(hw))
        elapsed = time.perf_counter() - start
        self.assertTrue(report["candidates"])
        self.assertLess(elapsed, 1.5, f"recommend took {elapsed:.2f}s (target 0.5s on a typical machine)")
