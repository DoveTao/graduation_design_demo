import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
R = REPO_ROOT / 'tools' / 'replay_s5d7_selected_pairwise_chain.py'
E = REPO_ROOT / 'tools' / 'evaluate_s5d8_selected_replay_comparison.py'


class TestS5D8SelectedPairwiseReplayStatic(unittest.TestCase):
    def test_tools_exist(self):
        self.assertTrue(R.exists())
        self.assertTrue(E.exists())

    def test_paths_and_classes(self):
        t = R.read_text(encoding='utf-8') + '\n' + E.read_text(encoding='utf-8')
        self.assertIn('reports/s5d8_selected_pairwise_replay_comparison.md', t)
        self.assertIn('checkpoints/S5D8_selected_pairwise_replay_comparison.json', t)
        for k in [
            'S5D8_SELECTED_REPLAY_COMPLETE',
            'S5D8_REPLAY_GRAPH_DISCONNECTED',
            'S5D8_DENSE_EXPORT_MISMATCH_SUSPECTED',
            'S5D8_SELECTED_PAIRWISE_TDIR_ISSUE_CONFIRMED',
            'S5D8_SELECTED_SPARSE_ONLY',
            'S5D8_ERROR',
        ]:
            self.assertIn(k, t)

    def test_selected_k1_caveat(self):
        t = R.read_text(encoding='utf-8') + '\n' + E.read_text(encoding='utf-8')
        self.assertIn('selected_k1', t)
        self.assertIn('sparse protocol', t)
        self.assertIn('diagnostic only', t)


if __name__ == '__main__':
    unittest.main()
