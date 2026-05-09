from __future__ import annotations

import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_SCRIPT = REPO_ROOT / "external_baselines" / "runners" / "run_orbslam3_fisheye_cam0.sh"
REPORT = REPO_ROOT / "reports" / "orbslam3_fisheye_run_report.md"
CHECKPOINT = REPO_ROOT / "checkpoints" / "ORB1c_orbslam3_fisheye_run.json"
RESULT = REPO_ROOT / "external_baselines" / "results" / "orbslam3_fisheye_cam0" / "scene01_seq03_est_tum.txt"


class Orbslam3FisheyeRunStaticTest(unittest.TestCase):
    def test_run_script_exists(self) -> None:
        self.assertTrue(RUN_SCRIPT.is_file())

    def test_report_checkpoint_result_paths_are_referenced(self) -> None:
        text = RUN_SCRIPT.read_text(encoding="utf-8") + REPORT.read_text(encoding="utf-8")
        self.assertIn("external_baselines/results/orbslam3_fisheye_cam0/scene01_seq03_est_tum.txt", text)
        self.assertIn("checkpoints/ORB1c_orbslam3_fisheye_run.json", text)
        self.assertTrue(CHECKPOINT.is_file())
        self.assertTrue(RESULT.is_file())

    def test_allowed_classifications_are_present(self) -> None:
        text = CHECKPOINT.read_text(encoding="utf-8") + REPORT.read_text(encoding="utf-8")
        for classification in [
            "ORB1C_RUN_COMPLETE",
            "ORB1C_TRACKING_FAILED",
            "ORB1C_SEQUENCE_EXPORT_FAILED",
            "ORB1C_BLOCKED_BY_PRECHECK",
            "ORB1C_RUNTIME_ERROR",
            "ORB1C_ERROR",
        ]:
            self.assertIn(classification, text)

    def test_run_script_does_not_call_evaluator(self) -> None:
        text = RUN_SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn("evaluate_external_baseline_trajectory.py", text)
        self.assertNotIn("run_external_baseline_comparison", text)

    def test_run_script_does_not_write_s5_or_official_files(self) -> None:
        text = RUN_SCRIPT.read_text(encoding="utf-8")
        forbidden = [
            "S5_clean_tmag_calibration_policy.json",
            "final_clean_candidate_manifest.json",
            "final_clean_candidate_split",
            "official evaluator",
            "verify_final_s5_candidate",
        ]
        for item in forbidden:
            self.assertNotIn(item, text)

    def test_json_report_mark_no_metric_evaluation(self) -> None:
        payload = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
        self.assertFalse(payload["baseline_metrics"]["evaluation_run"])
        text = REPORT.read_text(encoding="utf-8")
        self.assertIn("baseline_metrics.evaluation_run=false", text)
        self.assertIn("It does not evaluate ATE/drift/path_ratio.", text)

    def test_yaml_fitted_kb8_caveat_is_propagated(self) -> None:
        text = REPORT.read_text(encoding="utf-8") + CHECKPOINT.read_text(encoding="utf-8")
        self.assertIn("fitted KB8 compatibility approximation", text)
        self.assertIn("not original factory KannalaBrandt8 calibration", text)

    def test_no_fake_metrics_are_hardcoded(self) -> None:
        text = "\n".join(
            [
                RUN_SCRIPT.read_text(encoding="utf-8"),
                REPORT.read_text(encoding="utf-8"),
                CHECKPOINT.read_text(encoding="utf-8"),
            ]
        )
        self.assertNotIn("7.352288", text)
        self.assertNotIn("1.327343", text)
        self.assertNotIn("0.932379", text)


if __name__ == "__main__":
    unittest.main()
