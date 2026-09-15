from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import subprocess
import unittest
from unittest.mock import patch

from llm_configurator.calibration import calibrate, valid, VERSION
from llm_configurator.catalogue import demo_variants
from llm_configurator.domain import GIB, Requirements, now
from llm_configurator.engine import recommend
from llm_configurator.speed import estimate
from test_engine import hardware


def fixture(hw):
    return {'version': VERSION, 'fingerprint': hw['fingerprint'], 'timestamp': now(),
            'cpu': {'ram_bytes_s': 40e9, 'q4_parameters_s': 5e9},
            'gpus': {'GPU-test': {'vram_bytes_s': 500e9, 'h2d_bytes_s': 20e9,
                                 'd2h_bytes_s': 20e9, 'transfer_latency_s': 0.00002}}, 'warnings': []}


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.hw = hardware(ram=128, vram=128)
        self.hw['gpus'][0]['uuid'] = 'GPU-test'
        self.record = fixture(self.hw)
        self.model = replace(demo_variants()[0], demo=False, sha256='hash')

    def speed(self, **changes):
        args = dict(variant=self.model, hardware=self.hw, calibration=self.record,
                    context=8192, layers=0, users=1, gpu=self.hw['gpus'][0], target=15)
        args.update(changes)
        return estimate(**args)

    def test_real_cpu_probe_produces_finite_measurements(self):
        result = calibrate(self.hw)
        self.assertIsNotNone(result['cpu'], result['warnings'])
        self.assertGreater(result['cpu']['ram_bytes_s'], 0)
        self.assertGreater(result['cpu']['q4_parameters_s'], 0)
        self.assertLessEqual(result['cpu']['buffer_bytes'], 128 * 1024**2)
        self.assertLessEqual(result['cpu']['threads'], 8)

    def test_size_context_and_offload_affect_ranges(self):
        base = self.speed()
        bigger = self.speed(variant=replace(self.model, size_bytes=self.model.size_bytes * 2))
        longer = self.speed(context=131072)
        gpu = self.speed(layers=self.model.layers)
        split = self.speed(layers=self.model.layers // 2)
        self.assertTrue(base['available'])
        self.assertLess(base['low_tps'], base['high_tps'])
        self.assertLess(bigger['high_tps'], base['high_tps'])
        self.assertLess(longer['high_tps'], base['high_tps'])
        self.assertGreater(gpu['high_tps'], split['high_tps'])
        self.assertEqual(base['confidence'], 'low')

    def test_invalid_expired_foreign_and_missing_gpu_calibration(self):
        for record in [None, {}, {**self.record, 'fingerprint':'other'},
                       {**self.record, 'timestamp': (datetime.now(timezone.utc)-timedelta(days=8)).isoformat()}]:
            self.assertFalse(self.speed(calibration=record)['available'])
        self.assertFalse(self.speed(calibration={**self.record, 'gpus':{}}, layers=1)['available'])
        self.assertFalse(self.speed(calibration={**self.record, 'cpu':{'ram_bytes_s':float('nan')} })['available'])
        self.assertFalse(self.speed(users=2)['available'])
        self.assertFalse(self.speed(variant=replace(self.model, demo=True))['available'])

    def test_target_assessment_and_strict_filter_stays_verified_only(self):
        self.assertEqual(self.speed(target=0)['target_status'], 'likely_meets')
        self.assertEqual(self.speed(target=100000)['target_status'], 'likely_below')
        interval = self.speed()
        target = (interval['low_tps'] + interval['high_tps']) / 2
        self.assertEqual(self.speed(target=target)['target_status'], 'borderline')
        report = recommend([self.model], self.hw, Requirements(strict_speed=True), calibration=self.record)
        self.assertEqual(report['candidates'], [])

    def test_speed_priority_can_order_estimates_without_verifying_them(self):
        larger = replace(self.model, id='larger', base_repo='test/larger', size_bytes=self.model.size_bytes*2)
        report = recommend([larger, self.model], self.hw, Requirements(priority='speed'), calibration=self.record)
        self.assertEqual(report['candidates'][0]['variant_id'], self.model.id)
        self.assertIsNone(report['candidates'][0]['tps'])
        self.assertIsNone(report['candidates'][0]['speed_meets_target'])

    def test_timeout_preserves_completed_cpu_and_bounds_worker(self):
        partial = json.dumps({'cpu':self.record['cpu'], 'gpus':{}, 'warnings':[]}) + '\n'
        with patch('llm_configurator.calibration.subprocess.run', side_effect=subprocess.TimeoutExpired('worker', 12, output=partial.encode())) as run:
            result = calibrate(self.hw)
        self.assertEqual(result['cpu'], self.record['cpu'])
        self.assertTrue(valid(result, self.hw))
        self.assertIn('time limit', result['warnings'][0])
        self.assertEqual(run.call_args.kwargs['timeout'], 12)
        self.assertEqual(run.call_args.kwargs['env']['OPENBLAS_NUM_THREADS'], '1')
        self.assertNotIn('shell', run.call_args.kwargs)

    def test_failed_worker_does_not_invent_measurements(self):
        with patch('llm_configurator.calibration.subprocess.run', return_value=subprocess.CompletedProcess([], 1, '', 'failed')):
            result = calibrate(self.hw)
        self.assertIsNone(result['cpu'])
        self.assertFalse(self.speed(calibration=result)['available'])
