import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT = REPO_ROOT / "reports" / "JRT1_final_closeout_summary.md"
SUMMARY = REPO_ROOT / "checkpoints" / "JRT1_final_closeout_summary.json"


class TestJRT1CloseoutStatic(unittest.TestCase):
    def test_report_and_json_exist(self):
        self.assertTrue(REPORT.exists())
        self.assertTrue(SUMMARY.exists())

    def test_report_contains_required_closeout(self):
        text = REPORT.read_text(encoding="utf-8")
        self.assertIn("NO_STABLE_JRT1_TRAJECTORY_GAIN", text)
        self.assertIn("no final test", text)
        self.assertIn("S5 remains final clean candidate", text)

    def test_report_has_no_replacement_or_readiness_claim(self):
        lower = REPORT.read_text(encoding="utf-8").lower()
        self.assertNotIn("replaces " + "s5", lower)
        self.assertNotIn("new final " + "candidate", lower)
        self.assertNotIn("clean " + "gain", lower)
        self.assertNotIn("is practical-ready", lower)

    def test_locked_s5_metrics_unchanged(self):
        payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
        metrics = payload["s5_locked_metrics"]
        self.assertAlmostEqual(metrics["ATE"], 7.352288, places=9)
        self.assertAlmostEqual(metrics["drift"], 1.327343, places=9)
        self.assertAlmostEqual(metrics["path_ratio"], 0.932379, places=9)
        self.assertTrue(payload["no_final_test"])
        self.assertIsNone(payload["selected_candidate"])
        self.assertTrue(payload["s5_remains_final"])


if __name__ == "__main__":
    unittest.main()
