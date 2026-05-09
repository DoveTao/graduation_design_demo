from __future__ import annotations

import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT = REPO_ROOT / "reports" / "orbslam3_local_setup_report.md"
CHECKPOINT = REPO_ROOT / "checkpoints" / "ORB0_orbslam3_local_setup.json"
SCRIPT = REPO_ROOT / "scripts" / "setup_orbslam3_local.sh"


class Orbslam3LocalSetupStaticTest(unittest.TestCase):
    def test_report_checkpoint_paths_are_referenced(self) -> None:
        self.assertTrue(REPORT.is_file())
        self.assertTrue(CHECKPOINT.is_file())
        script_text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("logs/orbslam3_build.log", script_text)
        report_text = REPORT.read_text(encoding="utf-8")
        self.assertIn("checkpoints/ORB0_orbslam3_local_setup.json", report_text)
        self.assertIn("ORB0 ORB-SLAM3 Local Setup Report", report_text)

    def test_allowed_classifications_are_present(self) -> None:
        text = CHECKPOINT.read_text(encoding="utf-8") + REPORT.read_text(encoding="utf-8")
        for classification in [
            "ORB0_READY",
            "ORB0_SOURCE_UNAVAILABLE",
            "ORB0_BUILD_FAILED",
            "ORB0_ERROR",
        ]:
            self.assertIn(classification, text)

    def test_no_fake_orbslam3_metrics_are_hardcoded(self) -> None:
        combined = "\n".join(
            [
                REPORT.read_text(encoding="utf-8"),
                CHECKPOINT.read_text(encoding="utf-8"),
                SCRIPT.read_text(encoding="utf-8"),
            ]
        )
        self.assertNotIn("7.352288", combined)
        self.assertNotIn("1.327343", combined)
        self.assertNotIn("0.932379", combined)
        self.assertNotIn("scene01_seq03_est_tum.txt", combined)

    def test_setup_script_does_not_write_s5_or_evaluator_files(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        forbidden = [
            "S5_clean_tmag_calibration_policy.json",
            "final_clean_candidate_manifest.json",
            "final_clean_candidate_split",
            "official evaluator",
            "verify_final_s5_candidate",
        ]
        for item in forbidden:
            self.assertNotIn(item, text)

    def test_source_path_is_outside_tracked_project_source(self) -> None:
        payload = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
        target = Path(payload["source"]["target_path"])
        self.assertEqual(str(target), "/home/dovetao/third_party/ORB_SLAM3")
        self.assertFalse(str(target).startswith(str(REPO_ROOT)))

    def test_report_states_no_dataset_baseline_run(self) -> None:
        text = REPORT.read_text(encoding="utf-8")
        self.assertIn(
            "This setup step does not run ORB-SLAM3 on the dataset and does not produce baseline metrics.",
            text,
        )


if __name__ == "__main__":
    unittest.main()
