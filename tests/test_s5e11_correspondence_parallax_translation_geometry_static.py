from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TestS5E11CorrespondenceParallaxTranslationGeometryStatic(unittest.TestCase):
    def test_config_exists(self):
        self.assertTrue((ROOT / "configs/s5e11_correspondence_parallax_translation_geometry.yaml").exists())

    def test_required_files_exist(self):
        for rel in [
            "tools/s5e11_correspondence_geometry_lib.py",
            "tools/audit_s5e11_geometric_observability.py",
            "tools/extract_s5e11_pair_correspondence_features.py",
            "tools/train_s5e11_observability_candidate.py",
            "tools/train_s5e11_correspondence_direction_candidate.py",
            "tools/train_s5e11_scale_s5e9_stabilized_candidate.py",
            "tools/export_s5e11_adjacent_dense_predictions.py",
            "tools/evaluate_s5e11_traceable_dense.py",
            "tools/compare_s5e11_s5e10_s5e9_s5e8_s5e7_s5e6_s5e5_s5e4_s5e3_s5e2_orbslam3.py",
        ]:
            self.assertTrue((ROOT / rel).exists(), rel)

    def test_observability_and_correspondence_terms_present(self):
        txt = (ROOT / "tools/s5e11_correspondence_geometry_lib.py").read_text(encoding="utf-8")
        for term in [
            "matched_keypoint_count",
            "inlier_ratio",
            "parallax_proxy",
            "essential_translation_axis",
            "near_static_unobservable",
            "low_parallax_unreliable",
            "normal_observable",
            "large_motion_observable",
            "outlier_correspondence",
        ]:
            self.assertIn(term, txt)

    def test_s5e9_scale_default_and_s5e10_not_default(self):
        txt = (ROOT / "configs/s5e11_correspondence_parallax_translation_geometry.yaml").read_text(encoding="utf-8")
        self.assertIn("default_source: S5E9_tight_bounded_residual", txt)
        self.assertIn("reuse_s5e10_scale_recalibration: false", txt)

    def test_raw_vs_guarded_and_classifications_present(self):
        txt = (ROOT / "tools/evaluate_s5e11_traceable_dense.py").read_text(encoding="utf-8")
        for term in [
            "raw_vs_guarded_gap",
            "S5E11_GEOMETRY_FEATURES_UNAVAILABLE",
            "S5E11_OBSERVABILITY_LIMITED",
            "S5E11_OBSERVABILITY_MASK_IMPROVED",
            "S5E11_CORRESPONDENCE_DIRECTION_IMPROVED",
            "S5E11_SIGNED_DIRECTION_IMPROVED",
            "S5E11_SCALE_STABLE_DIRECTION_NOT_IMPROVED",
            "S5E11_RAW_GEOMETRY_IMPROVED",
            "S5E11_GUARD_DEPENDENT",
            "S5E11_INPUT_GEOMETRY_INSUFFICIENT",
            "S5E11_REGRESSION",
        ]:
            self.assertIn(term, txt)

    def test_chinese_report_requirement_documented(self):
        txt = (ROOT / "configs/s5e11_correspondence_parallax_translation_geometry.yaml").read_text(encoding="utf-8")
        self.assertIn("report_language: zh", txt)


if __name__ == "__main__":
    unittest.main()
