import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TestS5E19BNoopExportPathStatic(unittest.TestCase):
    def test_audit_tool_exists(self):
        self.assertTrue((ROOT / "tools/audit_s5e19_noop_export_path.py").exists())

    def test_report_and_checkpoint_paths_referenced(self):
        text = (ROOT / "tools/audit_s5e19_noop_export_path.py").read_text(encoding="utf-8")
        self.assertIn("reports/s5e19b_noop_export_path_audit.md", text)
        self.assertIn("checkpoints/S5E19B_noop_export_path_audit.json", text)

    def test_diff_terms_present(self):
        text = (ROOT / "tools/audit_s5e19_noop_export_path.py").read_text(encoding="utf-8")
        for token in [
            "trajectory_exact_copy",
            "trajectory_near_noop",
            "mean_abs_diff_rot",
            "mean_abs_diff_tdir",
            "delta_tdir_zero_count",
            "delta_log_tmag_zero_count",
            "smoke_policy_only",
            "fallback_to_s5e15",
            "likely_old_metrics_reuse",
        ]:
            self.assertIn(token, text)

    def test_allowed_classifications_present(self):
        text = (ROOT / "tools/audit_s5e19_noop_export_path.py").read_text(encoding="utf-8")
        for token in [
            "S5E19B_TRUE_NO_IMPROVEMENT",
            "S5E19B_NOOP_EXPORT_CONFIRMED",
            "S5E19B_SMOKE_POLICY_NO_REAL_TRAINING",
            "S5E19B_EVALUATOR_REUSED_OLD_METRICS",
            "S5E19B_MIXED_NOOP_AND_SMOKE",
            "S5E19B_AUDIT_ERROR",
        ]:
            self.assertIn(token, text)


if __name__ == "__main__":
    unittest.main()
