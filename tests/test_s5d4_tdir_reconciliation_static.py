import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "tools" / "audit_tdir_metric_provenance.py"


class TestS5D4TdirReconciliationStatic(unittest.TestCase):
    def test_script_exists(self):
        self.assertTrue(SCRIPT.exists())

    def test_paths(self):
        t = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("reports/s5d4_reconcile_old_tdir_with_dense_tdir.md", t)
        self.assertIn("checkpoints/S5D4_reconcile_old_tdir_with_dense_tdir.json", t)

    def test_allowed_classifications(self):
        t = SCRIPT.read_text(encoding="utf-8")
        for k in [
            "S5D4_RECONCILIATION_COMPLETE",
            "S5D4_OLD_TDIR_SOURCE_NOT_FOUND",
            "S5D4_FRAME_CONVENTION_ISSUE_SUSPECTED",
            "S5D4_METRIC_PROVENANCE_PARTIAL",
            "S5D4_ERROR",
        ]:
            self.assertIn(k, t)

    def test_keywords(self):
        t = SCRIPT.read_text(encoding="utf-8")
        for k in ["tdir", "translation direction", "tdir_mean_deg", "mean_cosine"]:
            self.assertIn(k, t)

    def test_variants(self):
        t = SCRIPT.read_text(encoding="utf-8")
        for k in ["dense_Twc_adjacent", "dense_Tcw_adjacent", "world_delta_adjacent", "axis_flip_sweep", "axis_permutation_and_sign_sweep", "k_step_pairs"]:
            self.assertIn(k, t)

    def test_k_values(self):
        t = SCRIPT.read_text(encoding="utf-8")
        for k in ["k1", "k2", "k5", "k10"]:
            self.assertIn(k, t)

    def test_provenance_classes(self):
        t = SCRIPT.read_text(encoding="utf-8")
        for k in ["OLD_METRIC_IS_ROT_NOT_TDIR", "OLD_TDIR_PAIRWISE_LOCAL_FRAME", "OLD_TDIR_SPARSE_SELECTED_CHAIN", "OLD_TDIR_DENSE_TUM_DERIVED", "OLD_TDIR_UNKNOWN_CONTEXT"]:
            self.assertIn(k, t)

    def test_no_fake_hardcoded_values(self):
        t = SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn("tdir_mean_deg = 20", t)

    def test_guardrails_and_caveats(self):
        t = SCRIPT.read_text(encoding="utf-8")
        for k in ["does_not_modify_s5_policy", "does_not_modify_final_manifest", "does_not_modify_train_test_split", "does_not_modify_official_evaluator", "does_not_modify_s5_locked_metrics"]:
            self.assertIn(k, t)
        for k in ["diagnostic only", "does not modify predictions", "does not replace official S5 locked result", "axis/sign/permutation sweep is diagnostic only"]:
            self.assertIn(k, t)


if __name__ == "__main__":
    unittest.main()
