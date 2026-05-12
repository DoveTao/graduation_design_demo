from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TestData2TdirScaleObservabilityStatic(unittest.TestCase):
    def test_audit_tool_exists(self):
        self.assertTrue((ROOT / "tools/audit_data2_tdir_scale_observability.py").exists())

    def test_required_terms_present(self):
        text = (ROOT / "tools/audit_data2_tdir_scale_observability.py").read_text(encoding="utf-8")
        for token in [
            "DATA2_tdir_scale_observability_and_split_shift_audit",
            "gt_step_median_ratio_eval_over_train",
            "low_motion_tdir_unstable",
            "tdir_observability_by_gt_step",
            "scale_path_error_audit",
            "train_eval_scale_gap",
            "S5E15",
            "STRUCT1B",
            "s5_locked_metrics_policy_unchanged",
            "DATA2_LOW_MOTION_TDIR_UNOBSERVABLE",
            "DATA2_TRAIN_EVAL_MOTION_DISTRIBUTION_SHIFT",
            "DATA2_SCALE_ERROR_SMALL_MOTION_COUPLED",
            "DATA2_GEOMETRY_CONFIDENCE_NOT_RELIABLE",
            "DATA2_TDIR_AND_SCALE_OBSERVABILITY_LIMITED",
            "DATA2_NO_MAJOR_DATA_ISSUE_FOUND",
            "DATA2_INSUFFICIENT_ARTIFACTS",
            "DATA2_AUDIT_ERROR",
        ]:
            self.assertIn(token, text)


if __name__ == "__main__":
    unittest.main()
