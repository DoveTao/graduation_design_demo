import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "tools" / "audit_pairwise_to_dense_tdir_gap.py"


class TestS5D5PairwiseToDenseGapStatic(unittest.TestCase):
    def test_exists(self):
        self.assertTrue(SCRIPT.exists())

    def test_paths(self):
        t = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("reports/s5d5_pairwise_to_dense_tdir_gap_audit.md", t)
        self.assertIn("checkpoints/S5D5_pairwise_to_dense_tdir_gap_audit.json", t)

    def test_allowed_classifications(self):
        t = SCRIPT.read_text(encoding="utf-8")
        for k in [
            "S5D5_PAIRWISE_DENSE_GAP_EXPLAINED",
            "S5D5_PAIRWISE_DENSE_GAP_PARTIAL",
            "S5D5_FINAL_S5_PAIRWISE_ARTIFACT_UNAVAILABLE",
            "S5D5_DIFFERENT_CANDIDATE_OR_SEQUENCE",
            "S5D5_ERROR",
        ]:
            self.assertIn(k, t)

    def test_historical_paths(self):
        t = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("S11_B_dt_aware_affine_calib_heldout_scene01_seq01", t)
        self.assertIn("S11_C_loss_only_consistency_heldout_scene01_seq01_u100", t)
        self.assertIn("S16_stronger_visual_backbone_feasibility_candidates.json", t)

    def test_bridge_variants(self):
        t = SCRIPT.read_text(encoding="utf-8")
        for k in ["dense_relative", "world_delta", "tdir_abs", "k_step", "thresholded"]:
            self.assertIn(k, t)

    def test_k_values(self):
        t = SCRIPT.read_text(encoding="utf-8")
        for k in ["k=1", "k=2", "k=5", "k=10", "k=20"]:
            self.assertIn(k, t)

    def test_guardrails(self):
        t = SCRIPT.read_text(encoding="utf-8")
        for k in ["does not modify predictions", "does not replace official S5 locked result", "S5 locked metrics/policy were not changed", "different candidate/sequence/protocol"]:
            self.assertIn(k, t)


if __name__ == "__main__":
    unittest.main()
