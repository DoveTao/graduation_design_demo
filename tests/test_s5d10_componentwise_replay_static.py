import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
A = REPO_ROOT / 'tools' / 'replay_s5d10_selected_components.py'
B = REPO_ROOT / 'tools' / 'analyze_s5d10_dense_edge_contributions.py'
C = REPO_ROOT / 'tools' / 'audit_s5d10_cuda_stable_validation.py'


class TestS5D10ComponentwiseReplayStatic(unittest.TestCase):
    def test_tools_exist(self):
        self.assertTrue(A.exists())
        self.assertTrue(B.exists())
        self.assertTrue(C.exists())

    def test_paths_classes_terms(self):
        t = A.read_text(encoding='utf-8') + '\n' + B.read_text(encoding='utf-8') + '\n' + C.read_text(encoding='utf-8')
        self.assertIn('reports/s5d10_componentwise_selected_replay_and_cuda_validation.md', t)
        self.assertIn('checkpoints/S5D10_componentwise_selected_replay_and_cuda_validation.json', t)
        self.assertIn('model_convention_as_declared', t)
        self.assertIn('inverse_relative_variant', t)
        self.assertIn('selected_vs_nonselected_dense_edges', t)
        self.assertIn('logs/s5d10_s6_eval_only.log', t)
        for k in [
            'S5D10_COMPLETE_VALIDATION_CLEAN',
            'S5D10_COMPLETE_WITH_CUDA_BLOCKER',
            'S5D10_CONVENTION_ISSUE_SUSPECTED',
            'S5D10_NONSELECTED_DENSE_EDGES_DOMINATE',
            'S5D10_SELECTED_PAIRWISE_TDIR_CONFIRMED',
            'S5D10_ERROR',
        ]:
            self.assertIn(k, t)


if __name__ == '__main__':
    unittest.main()
