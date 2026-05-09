import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT = REPO_ROOT / "reports" / "MF1_final_closeout_summary.md"
SUMMARY = REPO_ROOT / "checkpoints" / "MF1_final_closeout_summary.json"
RUNNER = REPO_ROOT / "scripts" / "run_mf1_closeout_summary.sh"


class TestMF1CloseoutStatic(unittest.TestCase):
    def test_closeout_files_exist(self):
        self.assertTrue(REPORT.exists(), str(REPORT))
        self.assertTrue(SUMMARY.exists(), str(SUMMARY))
        self.assertTrue(RUNNER.exists(), str(RUNNER))

    def test_runner_guard(self):
        text = RUNNER.read_text(encoding="utf-8")
        self.assertIn("set -euo pipefail", text)
        self.assertIn("verify_final_candidate.sh", text)
        self.assertIn("[mf1-closeout-summary] PASS", text)

    def test_report_required_decisions(self):
        text = REPORT.read_text(encoding="utf-8")
        self.assertIn("NO_STABLE_MF1_CHAIN_GAIN", text)
        self.assertIn("no final test", text.lower())
        self.assertIn("S5 remains final clean candidate", text)
        self.assertIn("selected_candidate_for_next_stage = null", text)

    def test_no_disallowed_positive_claims(self):
        text = REPORT.read_text(encoding="utf-8").lower()
        self.assertNotIn("replaces " + "s5", text)
        self.assertNotIn("new final " + "candidate", text)
        self.assertNotIn("final clean " + "gain", text)
        self.assertNotIn("is practical-ready", text)

    def test_json_locked_s5_metrics(self):
        payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
        self.assertEqual(payload["final_classification"], "NO_STABLE_MF1_CHAIN_GAIN")
        self.assertIsNone(payload["selected_candidate"])
        self.assertTrue(payload["no_final_test"])
        self.assertTrue(payload["s5_remains_final"])
        self.assertAlmostEqual(payload["s5_locked_metrics"]["ATE"], 7.352288, places=9)
        self.assertAlmostEqual(payload["s5_locked_metrics"]["drift"], 1.327343, places=9)
        self.assertAlmostEqual(payload["s5_locked_metrics"]["path_ratio"], 0.932379, places=9)


if __name__ == "__main__":
    unittest.main()
