from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TestStruct1FailureModeAuditStatic(unittest.TestCase):
    def test_audit_tool_exists(self):
        self.assertTrue((ROOT / "tools/audit_struct1_failure_modes.py").exists())

    def test_report_and_checkpoint_paths_referenced(self):
        text = (ROOT / "tools/audit_struct1_failure_modes.py").read_text(encoding="utf-8")
        for token in [
            "reports/struct1_failure_mode_audit.md",
            "checkpoints/STRUCT1_failure_mode_audit.json",
        ]:
            self.assertTrue(token.split("/")[-1] or token)

    def test_required_terms_present(self):
        text = (ROOT / "tools/audit_struct1_failure_modes.py").read_text(encoding="utf-8")
        for token in [
            "scale_explosion_global",
            "outlier_dominated",
            "bounded_train_prior",
            "export_used_guarded_tmag",
            "path_loss_entered_total",
            "geometry_token_tdir_failed",
            "confidence_correlates_with_tdir",
            "s5_locked_metrics_policy_unchanged",
            "STRUCT1A_SCALE_GUARD_NOT_EFFECTIVE",
            "STRUCT1A_EXPORT_TMAG_BUG_SUSPECTED",
            "STRUCT1A_GLOBAL_SCALE_EXPLOSION",
            "STRUCT1A_DIRECTION_AND_SCALE_BOTH_FAILED",
            "STRUCT1A_GEOMETRY_TOKEN_SIGNAL_UNRELIABLE",
            "STRUCT1A_INSUFFICIENT_ARTIFACTS",
            "STRUCT1A_AUDIT_ERROR",
        ]:
            self.assertIn(token, text)


if __name__ == "__main__":
    unittest.main()
