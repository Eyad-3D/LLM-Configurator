from dataclasses import replace
import math
import unittest

from llm_configurator.catalogue import demo_variants
from llm_configurator.domain import Requirements, now
from llm_configurator.engine import recommend
from llm_configurator.learning import fit_efficiency, ratios
from llm_configurator.speed import estimate
from test_calibration import fixture
from test_engine import hardware


class LearningTests(unittest.TestCase):
    def setUp(self):
        self.hw = hardware(ram=128, vram=0)
        self.cal = fixture(self.hw)
        self.model = replace(demo_variants()[0], demo=False, sha256="hash")

    def middle(self, context):
        e = estimate(self.model, self.hw, self.cal, context, 0, 1, None, 0)
        return math.sqrt(e["low_tps"] * e["high_tps"])

    def record(self, context, factor, **changes):
        base = {"variant_id": self.model.id, "sha256": "hash", "fingerprint": "test-machine", "context": context,
                "gpu_layers": 0, "gpu_uuid": None, "threads": 8, "users": 1, "tps": self.middle(context) * factor,
                "timestamp": now()}
        return {**base, **changes}

    def test_needs_two_measurements(self):
        self.assertIsNone(fit_efficiency([], self.hw, self.cal, [self.model]))
        self.assertIsNone(fit_efficiency([self.record(4096, 2)], self.hw, self.cal, [self.model]))
        self.assertIsNone(fit_efficiency([self.record(4096, 2)] * 3, self.hw, self.cal, []))  # variants unknown

    def test_median_is_robust_to_one_odd_result(self):
        records = [self.record(c, f) for c, f in [(2048, 1.5), (4096, 1.6), (8192, 1.55), (16384, 20)]]
        fit = fit_efficiency(records, self.hw, self.cal, [self.model])
        self.assertAlmostEqual(fit["factor"], math.sqrt(1.55 * 1.6), places=2)
        self.assertLess(fit["spread"], 0.5)
        self.assertEqual(fit["n"], 4)
        self.assertEqual(fit["label"], "adjusted from 4 local measurements")
        self.assertEqual(set(fit["by_mode"]), {"cpu"})

    def test_only_this_machines_recent_single_user_results_count(self):
        bad = [self.record(4096, 2, fingerprint="other"), self.record(4096, 2, users=2), self.record(4096, 2, tps=0),
               self.record(4096, 2, timestamp="2000-01-01T00:00:00+00:00"), self.record(4096, 2, sha256="other"),
               self.record(4096, 2, tps=float("nan")), {"variant_id": self.model.id},
               self.record(8192, 2, kind="tune", depth=1024)]  # a shallow tune trial is not an 8k measurement
        self.assertEqual(ratios(bad, self.hw, self.cal, [self.model]), [])
        self.assertEqual(len(ratios([self.record(8192, 2, kind="speed_test", depth=7552)], self.hw, self.cal, [self.model])), 1)

    def test_recommend_uses_the_fit_but_keeps_it_an_estimate(self):
        records = [self.record(2048, 3), self.record(4096, 3)]
        report = recommend([self.model], self.hw, Requirements(context=32768), records, self.cal)
        c = report["candidates"][0]
        self.assertEqual(c["evidence"], "estimated")
        self.assertIsNone(c["tps"])
        self.assertIn("adjusted from 2 local measurements", c["speed_estimate"]["method"])
        self.assertEqual(report["speed_adjustment"]["n"], 2)
        self.assertTrue(any("adjusted from 2 local measurements" in n for n in report["notes"]))
        plain = recommend([self.model], self.hw, Requirements(context=32768), [], self.cal)
        self.assertIsNone(plain["speed_adjustment"])
        self.assertGreater(c["speed_estimate"]["low_tps"], plain["candidates"][0]["speed_estimate"]["low_tps"])


if __name__ == "__main__":
    unittest.main()
