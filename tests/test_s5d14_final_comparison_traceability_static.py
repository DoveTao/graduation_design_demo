import json
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT = REPO_ROOT / "reports" / "s5_orbslam3_final_comparison_with_traceability_caveats.md"
CHECKPOINT = REPO_ROOT / "checkpoints" / "S5D14_final_comparison_traceability_summary.json"


class TestS5D14FinalComparisonTraceabilityStatic(unittest.TestCase):
    def test_paths_exist_and_referenced(self):
        self.assertTrue(REPORT.exists())
        self.assertTrue(CHECKPOINT.exists())
        combined = REPORT.read_text(encoding="utf-8") + CHECKPOINT.read_text(encoding="utf-8")
        self.assertIn("reports/s5_orbslam3_final_comparison_with_traceability_caveats.md", combined)
        self.assertIn("checkpoints/S5D14_final_comparison_traceability_summary.json", combined)

    def test_allowed_classifications_present(self):
        text = CHECKPOINT.read_text(encoding="utf-8")
        for name in [
            "S5D14_TRACEABILITY_SUMMARY_COMPLETE",
            "S5D14_REPORT_UPDATE_PARTIAL",
            "S5D14_ERROR",
        ]:
            self.assertIn(name, text + REPORT.read_text(encoding="utf-8"))

    def test_report_contains_traceability_terms(self):
        text = REPORT.read_text(encoding="utf-8")
        for term in [
            "official locked result",
            "verified external ORB-SLAM3 baseline",
            "diagnostic unverified dense artifact",
            "traceable selected_k1 artifact",
            "不能作为 official main result",
            "restored dense artifact is diagnostic-only",
            "S5 locked metrics/policy unchanged",
        ]:
            self.assertIn(term, text)

    def test_checkpoint_claims(self):
        obj = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
        self.assertEqual(obj["report_language"], "zh")
        self.assertFalse(obj["claims"]["verified_s5_dense_export_available"])
        self.assertFalse(obj["claims"]["same_evaluator_dense_table_allowed_as_official_main"])
        self.assertFalse(obj["claims"]["same_input_protocol"])

    def test_no_unexpected_fake_metrics(self):
        text = REPORT.read_text(encoding="utf-8") + CHECKPOINT.read_text(encoding="utf-8")
        self.assertIn("7.352288", text)
        self.assertIn("0.30854441069248173", text)
        self.assertIn("2.777267572676944", text)
        self.assertIn("0.981941", text)
        self.assertNotIn("7.352288999", text)
        self.assertNotIn("29.375292292292", text)

    def test_no_policy_manifest_split_or_evaluator_mutation(self):
        text = REPORT.read_text(encoding="utf-8") + CHECKPOINT.read_text(encoding="utf-8")
        for forbidden in [
            "S5_clean_tmag_calibration_policy.json').write_text",
            "final_clean_candidate_manifest.json').write_text",
            "train_test_split.write",
            "s6_final_clean_candidate_lockdown_audit.py').write_text",
        ]:
            self.assertNotIn(forbidden, text)
        self.assertIn("S5 locked metrics/policy unchanged", text)


if __name__ == "__main__":
    unittest.main()
