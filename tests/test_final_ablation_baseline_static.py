from __future__ import annotations

import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPO_ROOT / "tools" / "run_final_ablation_baseline_comparison.py"
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_final_ablation_baselines.sh"
MANIFEST_PATH = REPO_ROOT / "checkpoints" / "final_clean_candidate_manifest.json"


class FinalAblationBaselineStaticTest(unittest.TestCase):
    def test_paths_exist(self) -> None:
        self.assertTrue(TOOL_PATH.is_file())
        self.assertTrue(SCRIPT_PATH.is_file())

    def test_script_contains_expected_commands(self) -> None:
        text = SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertIn("set -euo pipefail", text)
        self.assertIn("bash scripts/verify_final_candidate.sh", text)
        self.assertIn("tools/run_final_ablation_baseline_comparison.py --verify-s5", text)

    def test_locked_metrics_unchanged(self) -> None:
        obj = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        metrics = obj["final_metrics"]
        self.assertEqual(metrics["ATE"], 7.352288)
        self.assertEqual(metrics["drift"], 1.327343)
        self.assertEqual(metrics["path_ratio"], 0.932379)

    def test_tool_contains_expected_output_paths(self) -> None:
        text = TOOL_PATH.read_text(encoding="utf-8")
        self.assertIn("AB1_final_ablation_baseline_comparison_results.json", text)
        self.assertIn("final_ablation_baseline_comparison.md", text)

    def test_no_s20_or_reselection_logic(self) -> None:
        tool = TOOL_PATH.read_text(encoding="utf-8")
        script = SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertNotIn("S20", tool)
        self.assertNotIn("S20", script)
        lowered = tool.lower() + "\n" + script.lower()
        self.assertNotIn("reselect", lowered)
        self.assertNotIn("candidate search", lowered)


if __name__ == "__main__":
    unittest.main()
