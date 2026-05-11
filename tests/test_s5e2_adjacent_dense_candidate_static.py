import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
BUILD = REPO_ROOT / "tools" / "build_s5e2_adjacent_dense_dataset.py"
TRAIN = REPO_ROOT / "tools" / "train_s5e2_adjacent_dense_candidate.py"
EXPORT = REPO_ROOT / "tools" / "export_s5e2_adjacent_dense_predictions.py"
EVAL = REPO_ROOT / "tools" / "evaluate_s5e2_traceable_dense.py"
COMPARE = REPO_ROOT / "tools" / "compare_s5e2_orbslam3_external.py"
LIB = REPO_ROOT / "tools" / "s5e2_adjacent_dense_lib.py"
CONFIG = REPO_ROOT / "configs" / "s5e2_adjacent_dense_candidate.yaml"


class TestS5E2AdjacentDenseCandidateStatic(unittest.TestCase):
    def _text(self):
        paths = [BUILD, TRAIN, EXPORT, EVAL, COMPARE, LIB, CONFIG]
        return "\n".join(path.read_text(encoding="utf-8") for path in paths)

    def test_tools_exist(self):
        for path in [BUILD, TRAIN, EXPORT, EVAL, COMPARE]:
            self.assertTrue(path.exists(), path)

    def test_config_path_exists(self):
        self.assertTrue(CONFIG.exists())
        self.assertIn("configs/s5e2_adjacent_dense_candidate.yaml", self._text())

    def test_report_checkpoint_paths_referenced(self):
        text = self._text()
        self.assertIn("reports/s5e2_adjacent_dense_candidate_report.md", text)
        self.assertIn("reports/s5e2_orbslam3_external_comparison.md", text)
        self.assertIn("checkpoints/S5E2_adjacent_dense_candidate.json", text)
        self.assertIn("external_baselines/results/s5e2_traceable_dense", text)

    def test_allowed_classifications_present(self):
        text = self._text()
        for name in [
            "S5E2_TRACEABLE_DENSE_IMPROVED",
            "S5E2_TRACEABLE_DENSE_EXPORTED_NO_IMPROVEMENT",
            "S5E2_TRACEABLE_DENSE_EXPORTED_FAILED_METRICS",
            "S5E2_ADJACENT_DATASET_BLOCKED",
            "S5E2_TRAINING_BLOCKED",
            "S5E2_ADJACENT_DENSE_EXPORT_BLOCKED",
            "S5E2_ERROR",
            "S5E2_ADJACENT_DATASET_READY",
            "S5E2_TRAINING_SMOKE_ONLY",
        ]:
            self.assertIn(name, text)

    def test_no_gt_leakage_caveat_present(self):
        text = self._text()
        self.assertIn("no_gt_leakage", text)
        self.assertIn("test_gt_used_for_training", text)
        self.assertIn("scene01/seq03 GT is used only for evaluation", text)

    def test_edge_provenance_schema_present(self):
        text = self._text()
        for term in [
            "edge_index",
            "timestamp_i",
            "timestamp_j",
            "source_type",
            "direct_adjacent_prediction",
            "source_model",
            "uses_gt_for_prediction",
            "rotation",
            "translation",
            "notes",
        ]:
            self.assertIn(term, text)

    def test_geometry_losses_present(self):
        text = self._text()
        for term in [
            "SO(3) geodesic",
            "tdir",
            "tmag log",
            "path length consistency",
            "short-window consistency",
        ]:
            self.assertIn(term, text)

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
