import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
R = REPO_ROOT / 'tools' / 'restore_or_regenerate_s5_dense_tum.py'
C = REPO_ROOT / 'tools' / 'rerun_s5d8_selected_edge_dense_comparison.py'


class TestS5D9RestoreDenseSelectedEdgeStatic(unittest.TestCase):
    def test_tools_exist(self):
        self.assertTrue(R.exists())
        self.assertTrue(C.exists())

    def test_paths_classes_and_caveat(self):
        t = R.read_text(encoding='utf-8') + '\n' + C.read_text(encoding='utf-8')
        self.assertIn('reports/s5d9_restore_s5_dense_and_selected_edge_comparison.md', t)
        self.assertIn('checkpoints/S5D9_restore_s5_dense_and_selected_edge_comparison.json', t)
        self.assertIn('external_baselines/results/s5_dense/scene01_seq03_s5_dense_est_tum.txt', t)
        self.assertIn('selected_k1 sparse protocol caveat', t)
        for k in [
            'S5D9_DENSE_RESTORED_COMPARISON_COMPLETE',
            'S5D9_DENSE_REGENERATED_COMPARISON_COMPLETE',
            'S5D9_DENSE_UNAVAILABLE',
            'S5D9_SELECTED_EDGE_MATCH_FAILED',
            'S5D9_DENSE_METRIC_MISMATCH',
            'S5D9_ERROR',
        ]:
            self.assertIn(k, t)


if __name__ == '__main__':
    unittest.main()
