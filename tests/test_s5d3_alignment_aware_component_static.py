import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "tools" / "evaluate_alignment_aware_component_diagnostics.py"


class TestS5D3AlignmentAwareComponentStatic(unittest.TestCase):
    def test_script_exists(self) -> None:
        self.assertTrue(SCRIPT.exists())

    def test_paths_referenced(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("reports/s5d3_validation_and_alignment_aware_component_audit.md", text)
        self.assertIn("checkpoints/S5D3_validation_and_alignment_aware_component_audit.json", text)

    def test_allowed_classifications_present(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        for key in [
            "S5D3_COMPLETE_VALIDATION_CLEAN",
            "S5D3_COMPLETE_WITH_VALIDATION_BLOCKER",
            "S5D3_DIAGNOSTICS_PARTIAL",
            "S5D3_BLOCKED",
            "S5D3_ERROR",
        ]:
            self.assertIn(key, text)

    def test_alignments_and_thresholds_present(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        for key in ["none", "se3", "sim3", "1e-8", "1e-6", "1e-4", "1e-3", "1e-2", "median_gt_step"]:
            self.assertIn(key, text)

    def test_metric_names_present(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        for key in [
            "rot_mean_deg",
            "tdir_mean_deg",
            "tdir_mean_cosine",
            "tmag_mean_log_error",
            "tmag_mean_ratio",
            "valid_tdir_pair_ratio",
        ]:
            self.assertIn(key, text)

    def test_matched_pair_fairness_present(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("no_orb_interpolation", text)
        self.assertIn("same_matched_timestamp_pairs", text)
        self.assertIn("orbslam3_partial_coverage", text)

    def test_validation_logs_referenced(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        for key in [
            "logs/s5d3_verify_final_candidate.log",
            "logs/s5d3_project_health_check.log",
            "logs/s5d3_s6_lockdown_eval_only.log",
        ]:
            self.assertIn(key, text)

    def test_no_fake_component_metrics_hardcoded(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn("rot_mean_deg = 20.", text)
        self.assertNotIn("tdir_mean_deg = 90.", text)
        self.assertNotIn("tmag_mean_ratio = 27.", text)

    def test_guardrails_not_modify_locked_assets(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        for key in [
            "does_not_modify_s5_policy",
            "does_not_modify_final_manifest",
            "does_not_modify_train_test_split",
            "does_not_modify_official_evaluator",
            "does_not_modify_s5_locked_metrics",
        ]:
            self.assertIn(key, text)

    def test_report_includes_required_caveats(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        for key in [
            "does not replace the official S5 locked result",
            "same_input_protocol=false",
            "fitted KB8 compatibility calibration",
            "ORB-SLAM3 has partial coverage",
        ]:
            self.assertIn(key, text)


if __name__ == "__main__":
    unittest.main()
