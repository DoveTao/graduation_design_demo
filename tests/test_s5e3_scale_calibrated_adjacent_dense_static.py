import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG = REPO_ROOT / "configs" / "s5e3_scale_calibrated_adjacent_dense.yaml"
TRAIN = REPO_ROOT / "tools" / "train_s5e3_scale_calibrated_candidate.py"
EXPORT = REPO_ROOT / "tools" / "export_s5e3_adjacent_dense_predictions.py"
EVAL = REPO_ROOT / "tools" / "evaluate_s5e3_traceable_dense.py"
COMPARE = REPO_ROOT / "tools" / "compare_s5e3_s5e2_orbslam3.py"


class TestS5E3ScaleCalibratedAdjacentDenseStatic(unittest.TestCase):
    def _text(self):
        return "\n".join(path.read_text(encoding="utf-8") for path in [CONFIG, TRAIN, EXPORT, EVAL, COMPARE])

    def test_files_exist(self):
        for path in [CONFIG, TRAIN, EXPORT, EVAL, COMPARE]:
            self.assertTrue(path.exists(), path)

    def test_report_checkpoint_paths_referenced(self):
        text = self._text()
        self.assertIn("reports/s5e3_scale_calibrated_adjacent_dense_report.md", text)
        self.assertIn("reports/s5e3_s5e2_orbslam3_comparison.md", text)
        self.assertIn("checkpoints/S5E3_scale_calibrated_adjacent_dense_candidate.json", text)
        self.assertIn("external_baselines/results/s5e3_traceable_dense", text)

    def test_allowed_classifications_present(self):
        text = self._text()
        for name in [
            "S5E3_GEOMETRY_IMPROVED",
            "S5E3_TMAG_IMPROVED_TDIR_STILL_BAD",
            "S5E3_TRACEABLE_DENSE_EXPORTED_NO_IMPROVEMENT",
            "S5E3_TRAINING_BLOCKED",
            "S5E3_EXPORT_BLOCKED",
            "S5E3_ERROR",
        ]:
            self.assertIn(name, text)

    def test_metrics_present(self):
        text = self._text()
        for term in ["rot", "tdir", "tdir_abs", "tmag", "ATE", "path_ratio"]:
            self.assertIn(term, text)

    def test_geometry_losses_present(self):
        text = self._text()
        for term in ["so3_geodesic", "tdir", "tmag_log", "path_length_consistency", "short_window_consistency"]:
            self.assertIn(term, text)

    def test_no_gt_leakage_caveat_present(self):
        text = self._text()
        self.assertIn("uses_scene01_seq03_for_training", text)
        self.assertIn("scene01/seq03 GT is used only for evaluation", text)
        self.assertIn("No restored dense artifact", text)

    def test_known_references_only(self):
        text = self._text()
        for known in ["32.77577273937451", "3.5557784814144453", "4.097680633241629", "0.30854441069248173", "0.224292165986624"]:
            self.assertIn(known, text)
        self.assertNotIn("fabricated", text.lower())

    def test_no_policy_manifest_split_or_evaluator_mutation(self):
        text = self._text()
        for forbidden in [
            "S5_clean_tmag_calibration_policy.json').write_text",
            "final_clean_candidate_manifest.json').write_text",
            "train_test_split.write",
            "s6_final_clean_candidate_lockdown_audit.py').write_text",
        ]:
            self.assertNotIn(forbidden, text)
        self.assertIn("official_s5_unchanged", text)
        self.assertIn("not_official_replacement", text)
        self.assertIn("S5 locked metrics/policy unchanged", text)


if __name__ == "__main__":
    unittest.main()
