from dataclasses import replace
import math
import unittest

from llm_configurator.catalogue import demo_variants
from llm_configurator.domain import GIB
from llm_configurator.speed import active_fraction, estimate, kv_bytes, moved_expert_layers
from test_calibration import fixture
from test_engine import hardware, moe_model


class SpeedModelTests(unittest.TestCase):
    def setUp(self):
        self.hw = hardware(ram=128, vram=128)
        self.cal = fixture(self.hw)
        self.dense = replace(demo_variants()[0], demo=False, sha256="d", size_bytes=moe_model().size_bytes, layers=48)
        self.moe = moe_model()

    def speed(self, variant, **changes):
        args = dict(variant=variant, hardware=self.hw, calibration=self.cal, context=8192, layers=0, users=1,
                    gpu=self.hw["gpus"][0], target=15)
        args.update(changes)
        return estimate(**args)

    def test_active_fraction(self):
        self.assertEqual(active_fraction(self.dense), 1.0)
        self.assertAlmostEqual(active_fraction(self.moe), 0.1 + 0.9 * 8 / 128)

    def test_moe_reads_only_active_bytes(self):
        dense, moe = self.speed(self.dense), self.speed(self.moe)
        self.assertGreater(moe["high_tps"], dense["high_tps"] * 3)

    def test_compressed_notes_speed_up_long_contexts(self):
        full = self.speed(self.dense, context=65536)
        small = self.speed(self.dense, context=65536, kv_cache_type="q4_0")
        self.assertGreater(small["high_tps"], full["high_tps"])

    def test_expert_offload_sits_between_gpu_and_cpu(self):
        gpu = self.speed(self.moe, layers=48)
        offload = self.speed(self.moe, layers=48, n_cpu_moe=24)
        cpu = self.speed(self.moe)
        self.assertTrue(gpu["high_tps"] > offload["high_tps"] > cpu["high_tps"])
        self.assertIn("CPU", offload["method"])

    def test_expert_offload_needs_transfer_calibration(self):
        cal = {**self.cal, "gpus": {"GPU-test": {"vram_bytes_s": 500e9}}}
        self.assertTrue(self.speed(self.moe, layers=48, calibration=cal)["available"])
        self.assertFalse(self.speed(self.moe, layers=48, n_cpu_moe=4, calibration=cal)["available"])

    def test_unified_memory_uses_gpu_bandwidth_without_transfer_costs(self):
        cal = {**self.cal, "gpus": {"GPU-test": {"vram_bytes_s": 200e9}}}
        result = self.speed(self.dense, layers=24, unified=True, calibration=cal)
        self.assertTrue(result["available"])
        missing = self.speed(self.dense, layers=48, unified=True, calibration={**self.cal, "gpus": {}})
        self.assertFalse(missing["available"])
        self.assertIn("graphics chip", missing["reason"])

    def test_moved_expert_layers_follow_llama_cpp_layer_order(self):
        self.assertEqual(moved_expert_layers(self.moe, 48, 10), 10)
        # -ngl 40 offloads the output layer plus blocks 9-47, so blocks 0-8 are already on the CPU.
        self.assertEqual(moved_expert_layers(self.moe, 40, 10), 1)
        self.assertEqual(moved_expert_layers(self.moe, 0, 10), 0)
        self.assertEqual(moved_expert_layers(self.dense, 48, 10), 0)

    def test_sliding_window_kv(self):
        sliding = replace(self.dense, sliding_window=1024, sliding_layers=40)
        self.assertLess(kv_bytes(sliding, 32768, 1), kv_bytes(self.dense, 32768, 1))
        self.assertEqual(kv_bytes(sliding, 512, 1), kv_bytes(self.dense, 512, 1))

    def test_efficiency_narrows_and_moves_the_range(self):
        base = self.speed(self.dense)
        fit = {"factor": 2.0, "spread": 0.1, "n": 3, "by_mode": {"cpu": {"factor": 2.0, "spread": 0.1, "n": 3}}}
        adjusted = self.speed(self.dense, efficiency=fit)
        middle = math.sqrt(base["low_tps"] * base["high_tps"]) * 2
        self.assertAlmostEqual(adjusted["low_tps"], middle * math.exp(-0.1), delta=0.02)
        self.assertAlmostEqual(adjusted["high_tps"], middle * math.exp(0.1), delta=0.02)
        self.assertIn("adjusted from 3 local measurements", adjusted["method"])
        self.assertEqual(adjusted["raw_low_tps"], base["low_tps"])
        self.assertEqual(adjusted["confidence"], "low")
        mode_fit = {**fit, "by_mode": {"cpu": {"factor": 1.0, "spread": 0.2, "n": 2}}}
        self.assertIn("adjusted from 2 local", self.speed(self.dense, efficiency=mode_fit)["method"])
        # No tests in this placement: shifted by the machine-wide factor, but never narrower than the raw range.
        other = self.speed(self.dense, efficiency={**fit, "by_mode": {"gpu": fit["by_mode"]["cpu"]}})
        self.assertAlmostEqual(other["high_tps"] / other["low_tps"], base["high_tps"] / base["low_tps"], places=1)
        self.assertAlmostEqual(math.sqrt(other["low_tps"] * other["high_tps"]), middle, delta=0.05)

    def test_unknown_quant_with_known_parameters_still_estimates(self):
        odd = replace(self.dense, quant="TQ1_0")
        self.assertFalse(self.speed(odd)["available"])
        self.assertTrue(self.speed(replace(odd, parameters=30 * 10**9))["available"])


if __name__ == "__main__":
    unittest.main()
