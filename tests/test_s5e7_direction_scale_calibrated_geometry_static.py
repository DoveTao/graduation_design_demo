import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
FILES = [
    ROOT / "configs" / "s5e7_direction_scale_calibrated_geometry.yaml",
    ROOT / "tools" / "s5e7_direction_scale_lib.py",
    ROOT / "tools" / "train_s5e7_direction_scale_calibrated_geometry.py",
    ROOT / "tools" / "export_s5e7_adjacent_dense_predictions.py",
    ROOT / "tools" / "evaluate_s5e7_traceable_dense.py",
    ROOT / "tools" / "compare_s5e7_s5e6_s5e5_s5e4_s5e3_s5e2_orbslam3.py",
    ROOT / "tools" / "audit_s5e7_direction_scale_failure.py",
]


class TestS5E7DirectionScaleStatic(unittest.TestCase):
    def _text(self):
        return "\n".join(p.read_text(encoding="utf-8") for p in FILES)

    def test_files_exist(self):
        for p in FILES:
            self.assertTrue(p.exists(), p)

    def test_factorization_and_bounded_scale(self):
        t = self._text()
        for x in [
            "pred_t = pred_dir_unit * pred_tmag",
            "pred_dir_unit",
            "pred_log_scale_delta",
            "pred_tmag",
            "pred_t",
            "clamped_log_scale_delta",
            "bounded_log_scale_delta",
            "log_scale_clip",
            "scale_guard_applied",
        ]:
            self.assertIn(x, t)

    def test_paths_and_classifications(self):
        t = self._text()
        for x in [
            "reports/s5e7_direction_scale_calibrated_geometry_report.md",
            "reports/s5e7_s5e6_s5e5_s5e4_s5e3_s5e2_orbslam3_comparison.md",
            "checkpoints/S5E7_direction_scale_calibrated_geometry_candidate.json",
            "external_baselines/results/s5e7_traceable_dense",
            "S5E7_DIRECTION_SCALE_IMPROVED",
            "S5E7_DIRECTION_ONLY_IMPROVED",
            "S5E7_SCALE_ONLY_IMPROVED",
            "S5E7_GUARD_DEPENDENT",
            "S5E7_NO_GEOMETRY_IMPROVEMENT",
            "S5E7_REGRESSION",
        ]:
            self.assertIn(x, t)

    def test_fallback_only_and_no_locked_mutation(self):
        t = self._text()
        self.assertIn("guard_policy: fallback_only", t)
        self.assertIn("official_s5_locked_metrics_unchanged", t)
        self.assertIn("not_official_replacement", t)
        for bad in [
            "S5_clean_tmag_calibration_policy.json').write_text",
            "final_clean_candidate_manifest.json').write_text",
            "train_test_split.write",
            "s6_final_clean_candidate_lockdown_audit.py').write_text",
        ]:
            self.assertNotIn(bad, t)

    def test_required_metrics_and_terms(self):
        t = self._text()
        for x in [
            "signed_tdir",
            "tdir_abs",
            "anti_parallel_rate",
            "severe_wrong_sign_rate",
            "direction_abs_good_but_signed_bad_rate",
            "tmag_p95_ratio",
            "tmag_max_ratio",
            "path_ratio",
            "small_motion",
            "normal_motion",
            "large_motion",
            "train_magnitude_prior",
            "raw_prediction_metrics",
            "guarded_prediction_metrics",
        ]:
            self.assertIn(x, t)


if __name__ == "__main__":
    unittest.main()
