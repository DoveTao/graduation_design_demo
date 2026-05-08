from __future__ import annotations

import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT = REPO_ROOT / "reports" / "alignment_consistent_strong_baseline_comparison.md"
RESULT_JSON = REPO_ROOT / "checkpoints" / "SB2_alignment_consistent_baseline_results.json"
SCRIPT = REPO_ROOT / "scripts" / "run_alignment_consistent_baseline_comparison.sh"
MANIFEST = REPO_ROOT / "checkpoints" / "final_clean_candidate_manifest.json"


class AlignmentConsistentBaselineStaticTest(unittest.TestCase):
    def test_report_and_json_exist(self) -> None:
        self.assertTrue(REPORT.is_file())
        self.assertTrue(RESULT_JSON.is_file())

    def test_report_content(self) -> None:
        text = REPORT.read_text(encoding="utf-8")
        self.assertIn("Pano-ORB-VO", text)
        self.assertIn("path_ratio", text)
        self.assertNotIn("uniformly outperforms Pano-ORB-VO", text)
        self.assertNotIn("replaces S5", text)
        self.assertNotIn("S20", text)

    def test_script_contract(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("set -euo pipefail", text)
        self.assertIn("bash scripts/verify_final_candidate.sh", text)
        self.assertIn("bash scripts/run_pano_orb_vo_baseline.sh", text)

    def test_locked_metrics_unchanged(self) -> None:
        payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(payload["final_candidate_name"], "S5_clean_tmag_calibration_policy")
        self.assertAlmostEqual(float(payload["final_metrics"]["ATE"]), 7.352288, places=6)
        self.assertAlmostEqual(float(payload["final_metrics"]["drift"]), 1.327343, places=6)
        self.assertAlmostEqual(float(payload["final_metrics"]["path_ratio"]), 0.932379, places=6)


if __name__ == "__main__":
    unittest.main()
