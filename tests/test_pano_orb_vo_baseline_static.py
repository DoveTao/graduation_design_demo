from __future__ import annotations

import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
TOOL = REPO_ROOT / "tools" / "run_pano_orb_vo_baseline.py"
SCRIPT = REPO_ROOT / "scripts" / "run_pano_orb_vo_baseline.sh"
REPORT = REPO_ROOT / "reports" / "pano_orb_vo_baseline_report.md"
RESULT_JSON = REPO_ROOT / "checkpoints" / "SB1_pano_orb_vo_baseline_results.json"
MANIFEST = REPO_ROOT / "checkpoints" / "final_clean_candidate_manifest.json"


class PanoOrbVoBaselineStaticTest(unittest.TestCase):
    def test_tool_and_script_exist(self) -> None:
        self.assertTrue(TOOL.is_file())
        self.assertTrue(SCRIPT.is_file())

    def test_script_contract(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("set -euo pipefail", text)
        self.assertIn("bash scripts/verify_final_candidate.sh", text)
        self.assertIn("reports/pano_orb_vo_baseline_report.md", text)
        self.assertIn("checkpoints/SB1_pano_orb_vo_baseline_results.json", text)

    def test_report_scope(self) -> None:
        text = REPORT.read_text(encoding="utf-8")
        self.assertIn("panorama-derived virtual pinhole", text)
        self.assertNotIn("This is ORB-SLAM3", text)
        self.assertIn("It is not ORB-SLAM3", text)
        self.assertIn("not an original fisheye-camera baseline", text)

    def test_result_json_exists(self) -> None:
        self.assertTrue(RESULT_JSON.is_file())
        payload = json.loads(RESULT_JSON.read_text(encoding="utf-8"))
        self.assertEqual(payload["name"], "SB1_pano_orb_vo_baseline")

    def test_s5_locked_metrics_unchanged(self) -> None:
        payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(payload["final_candidate_name"], "S5_clean_tmag_calibration_policy")
        self.assertAlmostEqual(float(payload["final_metrics"]["ATE"]), 7.352288, places=6)
        self.assertAlmostEqual(float(payload["final_metrics"]["drift"]), 1.327343, places=6)
        self.assertAlmostEqual(float(payload["final_metrics"]["path_ratio"]), 0.932379, places=6)


if __name__ == "__main__":
    unittest.main()
