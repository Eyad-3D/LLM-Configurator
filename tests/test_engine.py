import copy
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
        two = replace(self.model, id="other", name="B", scores={"general": 30, "coding": 80}, score_version="4.3")
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
