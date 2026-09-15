from dataclasses import replace
import unittest
from unittest.mock import patch
import tempfile
from llm_configurator.catalogue import demo_variants, refresh
from llm_configurator.domain import Requirements, now
from llm_configurator.engine import recommend
from llm_configurator.storage import Store
from test_engine import hardware


class GuidedRequirementsTests(unittest.TestCase):
    def test_evaluation_only_scans_selected_processes(self):
        from llm_configurator.app import evaluate
        with tempfile.TemporaryDirectory() as directory:
            with patch('llm_configurator.app.scan', return_value=hardware()) as scan:
                evaluate(Store(directory), {}, demo=True)
                scan.assert_called_once_with(include_processes=False, process_ids=set())
                scan.reset_mock()
                evaluate(Store(directory), {'reclaim_pids': [123]}, demo=True)
                scan.assert_called_once_with(include_processes=True, process_ids={123})

    def test_skip_rankings_ignores_cached_quality(self):
        model = replace(demo_variants()[0], scores={'general': 80}, score_source='https://example.com', score_version='4.3')
        report = recommend([model], hardware(), Requirements(include_rankings=False))
        self.assertTrue(report['candidates'])
        self.assertTrue(all(c['quality_score'] is None and c['quality_comparison']['rank'] is None for c in report['candidates']))
        self.assertIsNone(report['candidates'][0]['score_source'])

    def test_metadata_only_refresh_does_not_call_benchmark_provider(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch('llm_configurator.catalogue.definitions', return_value=[]), patch('llm_configurator.catalogue.fetch_scores') as scores:
                refresh(Store(directory), include_scores=False)
                scores.assert_not_called()

    def test_shortlist_prefers_three_distinct_models(self):
        report = recommend(demo_variants(), hardware(ram=128), Requirements(context=4096))
        selected = [c for c in report['candidates'] if c['id'] in report['shortlist']]
        self.assertEqual(len(selected), 3)
        self.assertEqual(len({c['base_repo'] for c in selected}), 3)

    def test_priority_changes_order_using_evidence(self):
        a = replace(demo_variants()[0], id='a', base_repo='test/a', sha256='a', scores={'general':80}, score_version='4.3')
        b = replace(a, id='b', base_repo='test/b', sha256='b', scores={'general':40})
        common = {'fingerprint':'test-machine','context':8192,'gpu_layers':0,'gpu_uuid':None,'threads':8,'users':1,'timestamp':now()}
        records = [{**common,'variant_id':'a','sha256':'a','tps':20}, {**common,'variant_id':'b','sha256':'b','tps':60}]
        quality = recommend([a,b],hardware(vram=0),Requirements(priority='quality'),records)
        speed = recommend([a,b],hardware(vram=0),Requirements(priority='speed'),records)
        self.assertEqual(quality['candidates'][0]['base_repo'],'test/a')
        self.assertEqual(speed['candidates'][0]['base_repo'],'test/b')
