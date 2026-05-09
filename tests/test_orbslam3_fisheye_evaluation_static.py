from __future__ import annotations

import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT = REPO_ROOT / "reports" / "orbslam3_fisheye_evaluation_report.md"
CHECKPOINT = REPO_ROOT / "checkpoints" / "ORB1d_orbslam3_fisheye_evaluation_results.json"


class Orbslam3FisheyeEvaluationStaticTest(unittest.TestCase):
    def test_report_checkpoint_paths_are_referenced(self) -> None:
        self.assertTrue(REPORT.is_file())
        self.assertTrue(CHECKPOINT.is_file())
        text = REPORT.read_text(encoding="utf-8")
        self.assertIn("checkpoints/ORB1d_orbslam3_fisheye_evaluation_results.json", text)
        self.assertIn("external_baselines/results/orbslam3_fisheye_cam0/scene01_seq03_est_tum.txt", text)

    def test_allowed_classifications_are_present(self) -> None:
        text = REPORT.read_text(encoding="utf-8") + CHECKPOINT.read_text(encoding="utf-8")
        for classification in [
            "ORB1D_EVALUATION_COMPLETE",
            "ORB1D_PARTIAL_EVALUATION",
            "ORB1D_EVALUATION_FAILED",
            "ORB1D_BLOCKED_BY_ORB1C",
            "ORB1D_ERROR",
        ]:
            self.assertIn(classification, text)

    def test_evaluator_commands_reference_all_alignments(self) -> None:
        payload = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
        commands = "\n".join(payload["evaluation"]["commands"])
        self.assertIn("--alignment none", commands)
        self.assertIn("--alignment se3", commands)
        self.assertIn("--alignment sim3", commands)

    def test_report_includes_tracking_success_rate(self) -> None:
        text = REPORT.read_text(encoding="utf-8")
        self.assertIn("tracking_success_rate", text)
        self.assertIn("273 / 454", text)

    def test_report_includes_fitted_kb8_caveat(self) -> None:
        text = REPORT.read_text(encoding="utf-8")
        self.assertIn("fitted KB8 compatibility approximation", text)
        self.assertIn("not native factory KB8 calibration", text)

    def test_report_includes_same_evaluator_caveat_against_s5(self) -> None:
        text = REPORT.read_text(encoding="utf-8")
        self.assertIn("same_evaluator_main_table_allowed=false", text)
        self.assertIn("S5 dense external trajectory export is not available", text)

    def test_no_s5_policy_manifest_split_or_official_evaluator_paths_modified(self) -> None:
        text = CHECKPOINT.read_text(encoding="utf-8") + REPORT.read_text(encoding="utf-8")
        self.assertNotIn("S5_clean_tmag_calibration_policy.json", text)
        self.assertNotIn("final_clean_candidate_manifest.json", text)
        self.assertNotIn("final_clean_candidate_split", text)
        self.assertNotIn("official evaluator", text.lower())

    def test_no_s5_locked_fake_metric_values_hardcoded(self) -> None:
        text = CHECKPOINT.read_text(encoding="utf-8") + REPORT.read_text(encoding="utf-8")
        self.assertNotIn("7.352288", text)
        self.assertNotIn("1.327343", text)
        self.assertNotIn("0.932379", text)


if __name__ == "__main__":
    unittest.main()
