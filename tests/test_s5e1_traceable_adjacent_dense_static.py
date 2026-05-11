import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
EXPORT = REPO_ROOT / "tools" / "export_s5e1_adjacent_dense_predictions.py"
TRAIN = REPO_ROOT / "tools" / "train_s5e1_geometry_candidate.py"
EVAL = REPO_ROOT / "tools" / "evaluate_s5e1_traceable_dense.py"
COMPARE = REPO_ROOT / "tools" / "compare_s5e1_orbslam3_external.py"
CONFIG = REPO_ROOT / "configs" / "s5e1_geometry_candidate.yaml"


class TestS5E1TraceableAdjacentDenseStatic(unittest.TestCase):
    def _text(self):
        return "\n".join(
            path.read_text(encoding="utf-8")
            for path in [EXPORT, TRAIN, EVAL, COMPARE, CONFIG]
        )

    def test_tools_exist(self):
        for path in [EXPORT, TRAIN, EVAL, COMPARE]:
            self.assertTrue(path.exists(), path)

    def test_config_path_referenced(self):
        text = self._text()
        self.assertTrue(CONFIG.exists())
        self.assertIn("configs/s5e1_geometry_candidate.yaml", text)

    def test_report_checkpoint_paths_referenced(self):
        text = self._text()
        self.assertIn("reports/s5e1_traceable_adjacent_dense_candidate_report.md", text)
        self.assertIn("reports/s5e1_orbslam3_external_comparison.md", text)
        self.assertIn("checkpoints/S5E1_traceable_adjacent_dense_candidate.json", text)
        self.assertIn("external_baselines/results/s5e1_traceable_dense", text)

    def test_allowed_classifications_present(self):
        text = self._text()
        for name in [
            "S5E1_TRACEABLE_DENSE_IMPROVED",
            "S5E1_TRACEABLE_DENSE_EXPORTED_NO_IMPROVEMENT",
            "S5E1_ADJACENT_DENSE_EXPORT_BLOCKED",
            "S5E1_TRAINING_BLOCKED",
            "S5E1_EXPERIMENT_FAILED",
            "S5E1_ERROR",
        ]:
            self.assertIn(name, text)

    def test_edge_provenance_schema_present(self):
        text = self._text()
        for term in [
            "edge_index",
            "timestamp_i",
            "timestamp_j",
            "source_type",
            "source_model",
            "prediction_type",
            "rotation_representation",
            "translation_vector",
            "translation_frame",
            "uses_gt_for_prediction",
        ]:
            self.assertIn(term, text)

    def test_losses_mentioned(self):
        text = self._text()
        for term in [
            "SO(3) geodesic",
            "tdir",
            "tmag log",
            "path length consistency",
            "short-window consistency",
        ]:
            self.assertIn(term, text)

    def test_no_fake_metrics_except_known_references(self):
        text = self._text()
        for known in [
            "7.352288",
            "1.327343",
            "0.932379",
            "0.30854441069248173",
            "0.224292165986624",
            "2.777267572676944",
        ]:
            self.assertIn(known, text)
        self.assertNotIn("S5E1_TRACEABLE_DENSE_IMPROVED = True", text)
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
        self.assertIn("do_not_modify_train_test_split", text)


if __name__ == "__main__":
    unittest.main()
