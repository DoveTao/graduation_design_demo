import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
E = REPO_ROOT / "tools" / "export_final_s5_pairwise_vectors.py"
R = REPO_ROOT / "tools" / "replay_s5_pairwise_dense_chain.py"
D = REPO_ROOT / "tools" / "evaluate_s5_pairwise_replay_diagnostics.py"


class TestS5D6PairwiseReplayStatic(unittest.TestCase):
    def test_tools_exist(self):
        self.assertTrue(E.exists())
        self.assertTrue(R.exists())
        self.assertTrue(D.exists())

    def test_paths_and_classes(self):
        t = D.read_text(encoding="utf-8") + "\n" + E.read_text(encoding="utf-8") + "\n" + R.read_text(encoding="utf-8")
        self.assertIn("reports/s5d6_final_s5_pairwise_replay_report.md", t)
        self.assertIn("checkpoints/S5D6_final_s5_pairwise_vectors_and_replay.json", t)
        for k in [
            "S5D6_PAIRWISE_REPLAY_COMPLETE",
            "S5D6_INTEGRATION_GAP_CONFIRMED",
            "S5D6_PAIRWISE_PREDICTION_ISSUE_CONFIRMED",
            "S5D6_SELECTED_SPARSE_ONLY",
            "S5D6_FINAL_S5_PAIRWISE_EXPORT_UNAVAILABLE",
            "S5D6_REPLAY_BLOCKED",
            "S5D6_ERROR",
        ]:
            self.assertIn(k, t)

    def test_output_paths(self):
        t = D.read_text(encoding="utf-8") + E.read_text(encoding="utf-8")
        for k in [
            "final_s5_pairwise_vectors_scene01_seq03.jsonl",
            "final_s5_pairwise_vectors_scene01_seq03.npz",
            "final_s5_replayed_dense_tum.txt",
            "final_s5_pairwise_tdir_diagnostics.json",
        ]:
            self.assertIn(k, t)

    def test_replay_variants_and_metrics(self):
        t = D.read_text(encoding="utf-8") + R.read_text(encoding="utf-8")
        for k in [
            "model_convention_as_declared",
            "inverse_relative_variant",
            "local_translation_rotated_by_current_pose",
            "local_translation_not_rotated",
            "tdir_mean_deg",
            "tdir_abs_mean_deg",
            "tmag_mean_ratio",
            "rot_mean_deg",
        ]:
            self.assertIn(k, t)

    def test_caveats(self):
        t = D.read_text(encoding="utf-8")
        for k in [
            "diagnostic only",
            "does not modify predictions",
            "does not replace official S5 locked result",
            "no GT used to generate predictions",
        ]:
            self.assertIn(k, t)


if __name__ == "__main__":
    unittest.main()
