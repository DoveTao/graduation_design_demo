import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
TOOL = REPO_ROOT / "tools" / "audit_s5_dense_export_convention.py"
RUNNER = REPO_ROOT / "scripts" / "run_s5d2_dense_export_convention_audit.sh"
REPORT = REPO_ROOT / "reports" / "S5D2_dense_export_convention_audit.md"
RESULTS = REPO_ROOT / "checkpoints" / "S5D2_dense_export_convention_audit.json"


class TestS5D2DenseExportConventionStatic(unittest.TestCase):
    def test_expected_files_exist(self):
        self.assertTrue(TOOL.exists(), str(TOOL))
        self.assertTrue(RUNNER.exists(), str(RUNNER))
        self.assertTrue(REPORT.exists(), str(REPORT))
        self.assertTrue(RESULTS.exists(), str(RESULTS))

    def test_runner_guard_and_outputs(self):
        text = RUNNER.read_text(encoding="utf-8")
        self.assertIn("set -euo pipefail", text)
        self.assertIn("verify_final_candidate.sh", text)
        self.assertIn("checkpoints/S5D2_dense_export_convention_audit.json", text)
        self.assertIn("reports/S5D2_dense_export_convention_audit.md", text)

    def test_report_references_path_ratio_mismatch(self):
        text = REPORT.read_text(encoding="utf-8")
        self.assertTrue("path_ratio = 2.777268" in text or "dense path_ratio mismatch" in text)
        self.assertIn("official path_ratio = 0.932379", text)

    def test_no_disallowed_claims(self):
        combined = "\n".join(
            p.read_text(encoding="utf-8")
            for p in (TOOL, RUNNER, REPORT)
            if p.exists()
        ).lower()
        self.assertNotIn("s5 improved", combined)
        self.assertNotIn("replaces " + "s5", combined)
        self.assertNotIn("new final " + "candidate", combined)
        self.assertNotIn("practical-ready", combined)

    def test_locked_s5_metrics_unchanged(self):
        payload = json.loads(RESULTS.read_text(encoding="utf-8"))
        self.assertAlmostEqual(payload["official_locked_metrics"]["ATE"], 7.352288, places=9)
        self.assertAlmostEqual(payload["official_locked_metrics"]["drift"], 1.327343, places=9)
        self.assertAlmostEqual(payload["official_locked_metrics"]["path_ratio"], 0.932379, places=9)
        self.assertTrue(payload["s5_policy_unchanged"])
        self.assertTrue(payload["s5_locked_metrics_unchanged"])


if __name__ == "__main__":
    unittest.main()
