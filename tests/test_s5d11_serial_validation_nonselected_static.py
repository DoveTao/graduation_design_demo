import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GUARD = REPO_ROOT / "scripts" / "run_serial_validation_guard.sh"
CONCURRENCY = REPO_ROOT / "tools" / "audit_s5d11_validation_concurrency.py"
EDGE_AUDIT = REPO_ROOT / "tools" / "audit_s5d11_nonselected_edge_failures.py"


class TestS5D11SerialValidationNonselectedStatic(unittest.TestCase):
    def _text(self):
        return "\n".join(p.read_text(encoding="utf-8") for p in [GUARD, CONCURRENCY, EDGE_AUDIT])

    def test_files_exist(self):
        self.assertTrue(GUARD.exists())
        self.assertTrue(CONCURRENCY.exists())
        self.assertTrue(EDGE_AUDIT.exists())

    def test_report_checkpoint_paths_referenced(self):
        text = self._text()
        self.assertIn("reports/s5d11_serial_validation_and_nonselected_edge_audit.md", text)
        self.assertIn("checkpoints/S5D11_serial_validation_and_nonselected_edge_audit.json", text)
        self.assertIn("checkpoints/S5D11_validation_concurrency_audit.json", text)

    def test_allowed_classifications_present(self):
        text = self._text()
        for name in [
            "S5D11_COMPLETE_VALIDATION_CLEAN",
            "S5D11_COMPLETE_WITH_VALIDATION_BLOCKER",
            "S5D11_NONSELECTED_EXTREME_TMAG_CONFIRMED",
            "S5D11_SERIAL_VALIDATION_STILL_OOM",
            "S5D11_ERROR",
        ]:
            self.assertIn(name, text)

    def test_lock_guard_and_serial_terms_present(self):
        text = self._text()
        self.assertIn("flock", text)
        self.assertIn("/tmp/graduation_design_demo_s6_eval.lock", text)
        self.assertIn("PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", text)
        self.assertIn("s6_final_clean_candidate_lockdown_audit.py", text)

    def test_selected_nonselected_and_worst_outputs_referenced(self):
        text = self._text()
        for term in [
            "selected_edges",
            "nonselected_edges",
            "nonselected_worst_edges.json",
            "nonselected_segment_summary.json",
            "selected_edge_ids.json",
            "nonselected_edge_metrics.json",
        ]:
            self.assertIn(term, text)

    def test_no_policy_manifest_split_or_official_evaluator_mutation(self):
        text = self._text()
        forbidden_write_targets = [
            "S5_clean_tmag_calibration_policy.json').write_text",
            "final_clean_candidate_manifest.json').write_text",
            "train/test split",
            "tools/s6_final_clean_candidate_lockdown_audit.py').write_text",
        ]
        for term in forbidden_write_targets:
            self.assertNotIn(term, text)
        self.assertIn("not_replaced_by_s5d11", text)
        self.assertIn("unchanged", text)

    def test_no_fake_metrics_except_known_s5_locked_references(self):
        text = self._text()
        self.assertIn("7.352288", text)
        self.assertIn("1.327343", text)
        self.assertIn("0.932379", text)
        self.assertNotIn("29.375292292292", text)


if __name__ == "__main__":
    unittest.main()
