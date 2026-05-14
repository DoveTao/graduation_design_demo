import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPORT = REPO_ROOT / "tools" / "export_s5_traceable_dense_trajectory.py"
EVAL = REPO_ROOT / "tools" / "evaluate_s5_traceable_dense_export.py"


class TestS5D13TraceableDenseExportStatic(unittest.TestCase):
    def _text(self):
        return EXPORT.read_text(encoding="utf-8") + "\n" + EVAL.read_text(encoding="utf-8")

    def test_tools_exist(self):
        self.assertTrue(EXPORT.exists())
        self.assertTrue(EVAL.exists())

    def test_report_checkpoint_paths_referenced(self):
        text = self._text()
        self.assertIn("checkpoints/S5D13_traceable_dense_export.json", text)
        self.assertIn("reports/s5d13_traceable_dense_export_report.md", text)
        self.assertIn("external_baselines/results/s5_traceable_dense_s5d13", text)

    def test_allowed_classifications_present(self):
        text = self._text()
        for name in [
            "S5D13_TRACEABLE_DENSE_EXPORT_COMPLETE",
            "S5D13_TRACEABLE_DENSE_UNAVAILABLE_SELECTED_ONLY",
            "S5D13_DENSE_FILL_BUG_CONFIRMED",
            "S5D13_PROVENANCE_UNAVAILABLE",
            "S5D13_ERROR",
        ]:
            self.assertIn(name, text)

    def test_edge_provenance_schema_terms_present(self):
        text = self._text()
        for term in [
            "direct_adjacent_prediction",
            "selected_prediction",
            "interpolated_fill",
            "composed_fill",
            "unavailable",
            "gap_num_adjacent_steps",
            "gap_endpoint_delta",
            "emitted_step_delta",
            "uses_gt_for_prediction",
            "rotation_convention",
        ]:
            self.assertIn(term, text)

    def test_no_gt_used_for_prediction_caveat_present(self):
        self.assertIn("no GT used for prediction", self._text())

    def test_no_fake_metrics_except_known_references(self):
        text = self._text()
        self.assertIn("7.352288", text)
        self.assertIn("1.327343", text)
        self.assertIn("0.932379", text)
        self.assertNotIn("29.375292292292", text)

    def test_no_policy_manifest_split_or_evaluator_mutation(self):
        text = self._text()
        for forbidden in [
            "S5_clean_tmag_calibration_policy.json').write_text",
            "final_clean_candidate_manifest.json').write_text",
            "train_test_split.write",
            "s6_final_clean_candidate_lockdown_audit.py').write_text",
        ]:
            self.assertNotIn(forbidden, text)
        self.assertIn("not_replaced_by_s5d13", text)
        self.assertIn("unchanged", text)


if __name__ == "__main__":
    unittest.main()
