import unittest
from llm_configurator.engine import add_quality_comparison


def row(model, score):
    return {'base_repo': model, 'quality_score': score}


class ComparisonTests(unittest.TestCase):
    def test_variants_and_contexts_do_not_inflate_rank(self):
        rows = [row('a', 50), row('a', 50), row('b', 30), row('c', None)]
        summary = add_quality_comparison(rows, True)
        self.assertEqual(summary['eligible_models'], 3)
        self.assertEqual(summary['rated_models'], 2)
        self.assertEqual(rows[0]['quality_comparison']['rank'], 1)
        self.assertEqual(rows[1]['quality_comparison']['rank'], 1)
        self.assertEqual(rows[2]['quality_comparison']['rank'], 2)
        self.assertEqual(rows[2]['quality_comparison']['points_behind_best'], 20)
        self.assertIsNone(rows[3]['quality_comparison']['rank'])

    def test_ties_are_competition_ranks(self):
        rows = [row('a', 50), row('b', 50), row('c', 20)]
        add_quality_comparison(rows, True)
        self.assertEqual([r['quality_comparison']['rank'] for r in rows], [1, 1, 3])
        self.assertTrue(rows[0]['quality_comparison']['tied'])
        self.assertEqual(rows[0]['quality_comparison']['models_below'], 1)

    def test_incompatible_versions_withhold_ranks(self):
        rows = [row('a', 50), row('b', 30)]
        add_quality_comparison(rows, False)
        self.assertTrue(all(r['quality_comparison']['rank'] is None for r in rows))

    def test_conflicting_model_scores_withhold_ranks(self):
        rows = [row('a', 50), row('a', 30)]
        self.assertFalse(add_quality_comparison(rows, True)['comparable'])

    def test_zero_is_a_real_score_and_missing_is_not(self):
        rows = [row('a', 0), row('b', None)]
        add_quality_comparison(rows, True)
        self.assertEqual(rows[0]['quality_comparison']['rank'], 1)
        self.assertIsNone(rows[1]['quality_comparison']['rank'])
