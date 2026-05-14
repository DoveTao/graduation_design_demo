import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
FILES = [
    ROOT / "configs" / "s5e8_translation_geometry_diagnostic_ablation.yaml",
    ROOT / "tools" / "s5e8_translation_geometry_lib.py",
    ROOT / "tools" / "audit_s5e8_translation_supervision_quality.py",
    ROOT / "tools" / "run_s5e8_oracle_ablation.py",
    ROOT / "tools" / "train_s5e8_direction_only_candidate.py",
    ROOT / "tools" / "train_s5e8_prior_only_baselines.py",
    ROOT / "tools" / "export_s5e8_adjacent_dense_predictions.py",
    ROOT / "tools" / "evaluate_s5e8_traceable_dense.py",
    ROOT / "tools" / "compare_s5e8_s5e7_s5e6_s5e5_s5e4_s5e3_s5e2_orbslam3.py",
]


class TestS5E8TranslationGeometryDiagnosticStatic(unittest.TestCase):
    def _text(self):
        return "\n".join(p.read_text(encoding="utf-8") for p in FILES)

    def test_files_exist(self):
        for p in FILES:
            self.assertTrue(p.exists(), p)

    def test_required_terms(self):
        t = self._text()
        for x in [
            "oracle_direction",
            "oracle_scale",
            "direction-only",
            "prior-only",
            "raw_prediction_metrics",
            "guarded_prediction_metrics",
            "bucketed_evaluation",
            "small_motion",
            "normal_motion",
            "large_motion",
            "中文",
        ]:
            self.assertIn(x, t)

    def test_policy_and_factorization(self):
        t = self._text()
        self.assertIn("official_s5_locked_metrics_unchanged", t)
        self.assertIn("not_official_replacement", t)
        self.assertIn("translation geometry diagnostic", t)
        for bad in [
            "S5_clean_tmag_calibration_policy.json').write_text",
            "final_clean_candidate_manifest.json').write_text",
            "train_test_split.write",
        ]:
            self.assertNotIn(bad, t)

    def test_paths_and_classifications(self):
        t = self._text()
        for x in [
            "reports/s5e8_translation_geometry_diagnostic_ablation_report.md",
            "reports/s5e8_s5e7_s5e6_s5e5_s5e4_s5e3_s5e2_orbslam3_comparison.md",
            "checkpoints/S5E8_translation_geometry_diagnostic_ablation_candidate.json",
            "S5E8_DIRECTION_SUPERVISION_LIMITED",
            "S5E8_SCALE_SUPERVISION_LIMITED",
            "S5E8_INPUT_GEOMETRY_INSUFFICIENT",
            "S5E8_SMALL_MOTION_DOMINATED",
            "S5E8_DIRECTION_ONLY_IMPROVED",
            "S5E8_RAW_GEOMETRY_IMPROVED",
            "S5E8_GUARD_DEPENDENT",
            "S5E8_NO_IMPROVEMENT",
            "S5E8_REGRESSION",
        ]:
            self.assertIn(x, t)

    def test_metrics(self):
        t = self._text()
        for x in [
            "signed_tdir_mean",
            "tdir_abs_mean",
            "anti_parallel_rate",
            "tmag_p95",
            "path_ratio",
            "small_motion_error_contribution",
            "tmag_p95_gap",
            "tmag_max_gap",
            "sim3_ATE_gap",
        ]:
            self.assertIn(x, t)


if __name__ == "__main__":
    unittest.main()
