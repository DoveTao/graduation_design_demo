import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
AUDIT = REPO_ROOT / "tools" / "audit_s5d12_dense_fill_path.py"
INSTRUMENT = REPO_ROOT / "tools" / "instrument_s5_dense_export_fill_path.py"


class TestS5D12DenseFillPathStatic(unittest.TestCase):
    def _text(self):
        return AUDIT.read_text(encoding="utf-8") + "\n" + INSTRUMENT.read_text(encoding="utf-8")

    def test_tools_exist(self):
        self.assertTrue(AUDIT.exists())
        self.assertTrue(INSTRUMENT.exists())

    def test_report_checkpoint_paths_referenced(self):
        text = self._text()
        self.assertIn("checkpoints/S5D12_dense_fill_path_audit.json", text)
        self.assertIn("reports/s5d12_dense_fill_path_audit.md", text)
        self.assertIn("external_baselines/results/s5d12_dense_fill_path_audit", text)

    def test_allowed_classifications_present(self):
        text = self._text()
        for name in [
            "S5D12_DENSE_FILL_BUG_IDENTIFIED",
            "S5D12_DENSE_FILL_PROTOCOL_MISMATCH_IDENTIFIED",
            "S5D12_NONSELECTED_SOURCE_UNKNOWN",
            "S5D12_AUDIT_COMPLETE_NO_FIX",
            "S5D12_ERROR",
        ]:
            self.assertIn(name, text)

    def test_hypothesis_names_present(self):
        text = self._text()
        for name in [
            "total_gap_translation_repeated_per_edge",
            "timestamp_unit_bug",
            "missing_division_by_gap_length",
            "endpoint_displacement_not_normalized",
            "selected_nonselected_protocol_mismatch",
            "nonselected_edges_not_direct_predictions",
        ]:
            self.assertIn(name, text)

    def test_dominant_run_referenced(self):
        text = self._text()
        self.assertIn("72-360", text)
        self.assertIn("nonselected_run_72_360_trace.json", text)

    def test_no_fake_metrics_except_known_s5d11_references(self):
        text = self._text()
        self.assertIn("29.375292", text)
        self.assertIn("88.130943", text)
        self.assertIn("335.512628", text)
        self.assertNotIn("29.375292292292", text)

    def test_no_policy_manifest_split_or_official_evaluator_mutation(self):
        text = self._text()
        forbidden = [
            "S5_clean_tmag_calibration_policy.json').write_text",
            "final_clean_candidate_manifest.json').write_text",
            "train_test_split.write",
            "s6_final_clean_candidate_lockdown_audit.py').write_text",
        ]
        for term in forbidden:
            self.assertNotIn(term, text)
        self.assertIn("does_not_modify_policy", text)
        self.assertIn("does_not_modify_manifest", text)
        self.assertIn("does_not_modify_official_evaluator", text)
        self.assertIn("not_replaced_by_s5d12", text)


if __name__ == "__main__":
    unittest.main()
