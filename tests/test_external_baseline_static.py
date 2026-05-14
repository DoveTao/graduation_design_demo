from __future__ import annotations

import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_SCRIPT = REPO_ROOT / "scripts" / "run_external_baseline_comparison.sh"
EXPORT_TOOL = REPO_ROOT / "tools" / "export_external_baseline_dataset.py"
EVAL_TOOL = REPO_ROOT / "tools" / "evaluate_external_baseline_trajectory.py"
RUN_TOOL = REPO_ROOT / "tools" / "run_external_baseline_comparison.py"
PROTOCOL_MD = REPO_ROOT / "reports" / "external_algorithm_baseline_protocol.md"
COMPARISON_MD = REPO_ROOT / "reports" / "external_algorithm_baseline_comparison.md"
MANIFEST = REPO_ROOT / "checkpoints" / "final_clean_candidate_manifest.json"


class ExternalBaselineStaticTest(unittest.TestCase):
    def test_tools_and_script_exist(self) -> None:
        self.assertTrue(EXPORT_TOOL.is_file())
        self.assertTrue(EVAL_TOOL.is_file())
        self.assertTrue(RUN_TOOL.is_file())
        self.assertTrue(RUN_SCRIPT.is_file())

    def test_run_script_contract(self) -> None:
        text = RUN_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("set -euo pipefail", text)
        self.assertIn("bash scripts/verify_final_candidate.sh", text)

    def test_reports_contain_protocol_caveats(self) -> None:
        protocol = PROTOCOL_MD.read_text(encoding="utf-8")
        comparison = COMPARISON_MD.read_text(encoding="utf-8")
        self.assertIn("Published results from different datasets are not compared directly against S5.", protocol)
        self.assertIn("alignment modes: `none / se3 / sim3`", protocol)
        self.assertIn("scale-sensitive path_ratio", protocol)
        self.assertIn("Published results from different datasets are not compared directly against S5.", comparison)

    def test_no_s20_or_reselection_logic(self) -> None:
        for path in (EXPORT_TOOL, EVAL_TOOL, RUN_TOOL, RUN_SCRIPT):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("S20", text)
            self.assertNotIn("reselect", text.lower())
            self.assertNotIn("candidate search", text.lower())

    def test_locked_metrics_unchanged(self) -> None:
        payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(payload["final_candidate_name"], "S5_clean_tmag_calibration_policy")
        self.assertAlmostEqual(float(payload["final_metrics"]["ATE"]), 7.352288, places=6)
        self.assertAlmostEqual(float(payload["final_metrics"]["drift"]), 1.327343, places=6)
        self.assertAlmostEqual(float(payload["final_metrics"]["path_ratio"]), 0.932379, places=6)


if __name__ == "__main__":
    unittest.main()
