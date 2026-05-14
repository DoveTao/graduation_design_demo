from __future__ import annotations

import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
TOOL = REPO_ROOT / "tools" / "audit_s5_external_export_compatibility.py"
SCRIPT = REPO_ROOT / "scripts" / "run_s5_export_compatibility_audit.sh"
REPORT = REPO_ROOT / "reports" / "s5_external_export_compatibility_audit.md"
RESULT_JSON = REPO_ROOT / "checkpoints" / "SB2b_s5_export_compatibility_audit.json"
MANIFEST = REPO_ROOT / "checkpoints" / "final_clean_candidate_manifest.json"


class S5ExportCompatibilityAuditStaticTest(unittest.TestCase):
    def test_tool_and_runner_exist(self) -> None:
        self.assertTrue(TOOL.is_file())
        self.assertTrue(SCRIPT.is_file())

    def test_runner_contract(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("set -euo pipefail", text)
        self.assertIn("bash scripts/verify_final_candidate.sh", text)

    def test_report_and_json_exist(self) -> None:
        self.assertTrue(REPORT.is_file())
        self.assertTrue(RESULT_JSON.is_file())

    def test_report_content(self) -> None:
        text = REPORT.read_text(encoding="utf-8")
        self.assertIn("path_ratio mismatch", text)
        self.assertIn("tracking_success_rate", text)
        self.assertIn("diagnostic-only", text)
        self.assertNotIn("uniformly outperforms Pano-ORB-VO", text)
        self.assertNotIn("replaces S5", text)

    def test_locked_metrics_unchanged(self) -> None:
        payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(payload["final_candidate_name"], "S5_clean_tmag_calibration_policy")
        self.assertAlmostEqual(float(payload["final_metrics"]["ATE"]), 7.352288, places=6)
        self.assertAlmostEqual(float(payload["final_metrics"]["drift"]), 1.327343, places=6)
        self.assertAlmostEqual(float(payload["final_metrics"]["path_ratio"]), 0.932379, places=6)


if __name__ == "__main__":
    unittest.main()
