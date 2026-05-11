import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
FILES = [
    ROOT / "configs" / "s5e10_order_aware_signed_direction_scale_recalibration.yaml",
    ROOT / "tools" / "s5e10_order_geometry_lib.py",
    ROOT / "tools" / "audit_s5e10_pair_order_observability.py",
    ROOT / "tools" / "train_s5e10_order_aware_signed_direction_candidate.py",
    ROOT / "tools" / "train_s5e10_scale_recalibrated_candidate.py",
    ROOT / "tools" / "export_s5e10_adjacent_dense_predictions.py",
    ROOT / "tools" / "evaluate_s5e10_traceable_dense.py",
    ROOT / "tools" / "compare_s5e10_s5e9_s5e8_s5e7_s5e6_s5e5_s5e4_s5e3_s5e2_orbslam3.py",
]


class TestS5E10OrderAwareSignedDirectionScaleRecalibrationStatic(unittest.TestCase):
    def _text(self):
        return "\n".join(p.read_text(encoding="utf-8") for p in FILES)

    def test_files_exist(self):
        for p in FILES:
            self.assertTrue(p.exists(), p)

    def test_required_terms(self):
        t = self._text()
        for x in [
            "uses_ordered_concat",
            "uses_feature_difference",
            "uses_absolute_difference",
            "pair_order_label_flip_consistent",
            "reversed_pair_export_consistent",
            "signed_direction_label_convention_valid",
            "reversed_pair_count",
            "pair_order_dir_flip_success_rate",
            "axis + sign",
            "order-aware",
            "中文",
        ]:
            self.assertIn(x, t)

    def test_policy_and_paths(self):
        t = self._text()
        self.assertIn("official_s5_locked_metrics_unchanged", t)
        for x in [
            "reports/s5e10_order_aware_signed_direction_scale_recalibration_report.md",
            "reports/s5e10_s5e9_s5e8_s5e7_s5e6_s5e5_s5e4_s5e3_s5e2_orbslam3_comparison.md",
            "checkpoints/S5E10_order_aware_signed_direction_scale_recalibration_candidate.json",
        ]:
            self.assertIn(x, t)
        for bad in [
            "S5_clean_tmag_calibration_policy.json').write_text",
            "final_clean_candidate_manifest.json').write_text",
            "train_test_split.write",
        ]:
            self.assertNotIn(bad, t)

    def test_classifications(self):
        t = self._text()
        for x in [
            "S5E10_ORDER_INVARIANT_BUG_FOUND",
            "S5E10_PAIR_ORDER_OBSERVABLE",
            "S5E10_SIGNED_DIRECTION_IMPROVED",
            "S5E10_SCALE_RECALIBRATED",
            "S5E10_SMALL_MOTION_IMPROVED",
            "S5E10_RAW_GEOMETRY_IMPROVED",
            "S5E10_GUARD_DEPENDENT",
            "S5E10_INPUT_GEOMETRY_INSUFFICIENT",
            "S5E10_NO_IMPROVEMENT",
            "S5E10_REGRESSION",
        ]:
            self.assertIn(x, t)

    def test_metrics(self):
        t = self._text()
        for x in [
            "signed_tdir_mean",
            "anti_parallel_rate",
            "pair_order_failure_rate",
            "raw_vs_guarded_gap",
            "tmag_p95_gap",
            "path_ratio_gap",
            "sim3_ATE_gap",
        ]:
            self.assertIn(x, t)


if __name__ == "__main__":
    unittest.main()
