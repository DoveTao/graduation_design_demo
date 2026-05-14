import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
FILES = [
    ROOT / "configs" / "s5e9_scale_unit_small_motion_signed_direction_fix.yaml",
    ROOT / "tools" / "s5e9_scale_pipeline_audit.py",
    ROOT / "tools" / "s5e9_small_motion_geometry_lib.py",
    ROOT / "tools" / "train_s5e9_scale_calibrated_candidate.py",
    ROOT / "tools" / "train_s5e9_signed_direction_order_candidate.py",
    ROOT / "tools" / "export_s5e9_adjacent_dense_predictions.py",
    ROOT / "tools" / "evaluate_s5e9_traceable_dense.py",
    ROOT / "tools" / "compare_s5e9_s5e8_s5e7_s5e6_s5e5_s5e4_s5e3_s5e2_orbslam3.py",
]


class TestS5E9ScaleUnitSmallMotionSignedDirectionStatic(unittest.TestCase):
    def _text(self):
        return "\n".join(p.read_text(encoding="utf-8") for p in FILES)

    def test_files_exist(self):
        for p in FILES:
            self.assertTrue(p.exists(), p)

    def test_policy_and_language(self):
        t = self._text()
        self.assertIn("official_s5_locked_metrics_unchanged", t)
        self.assertIn("中文", t)
        for bad in [
            "S5_clean_tmag_calibration_policy.json').write_text",
            "final_clean_candidate_manifest.json').write_text",
            "train_test_split.write",
        ]:
            self.assertNotIn(bad, t)

    def test_required_terms(self):
        t = self._text()
        for x in [
            "unit_mismatch_suspected",
            "export_eval_tmag_consistency",
            "log_scale_clip",
            "near_static",
            "small_motion",
            "pair_order_dir_flip_success_rate",
            "signed_direction_order_observable",
            "raw_vs_guarded_gap",
            "tight bounded log-scale residual",
        ]:
            self.assertIn(x, t)

    def test_classifications_and_paths(self):
        t = self._text()
        for x in [
            "reports/s5e9_scale_unit_small_motion_signed_direction_fix_report.md",
            "reports/s5e9_s5e8_s5e7_s5e6_s5e5_s5e4_s5e3_s5e2_orbslam3_comparison.md",
            "checkpoints/S5E9_scale_unit_small_motion_signed_direction_fix_candidate.json",
            "S5E9_SCALE_UNIT_BUG_FOUND",
            "S5E9_SCALE_RAW_IMPROVED",
            "S5E9_SIGNED_DIRECTION_IMPROVED",
            "S5E9_SMALL_MOTION_IMPROVED",
            "S5E9_RAW_GEOMETRY_IMPROVED",
            "S5E9_GUARD_DEPENDENT",
            "S5E9_INPUT_GEOMETRY_INSUFFICIENT",
            "S5E9_NO_IMPROVEMENT",
            "S5E9_REGRESSION",
        ]:
            self.assertIn(x, t)

    def test_metrics(self):
        t = self._text()
        for x in [
            "raw_tmag_to_gt_tmag_median_ratio",
            "raw_tmag_to_gt_tmag_p95_ratio",
            "pred_log_scale_delta",
            "residual_saturation_rate",
            "over_scale_rate",
            "under_scale_rate",
            "small_motion_amplification",
            "tmag_p95_gap",
            "sim3_ATE_gap",
        ]:
            self.assertIn(x, t)


if __name__ == "__main__":
    unittest.main()
