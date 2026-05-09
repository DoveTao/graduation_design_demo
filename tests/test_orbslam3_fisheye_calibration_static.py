from __future__ import annotations

import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
TOOL = REPO_ROOT / "tools" / "convert_cam_infos_to_orbslam3_yaml.py"
REPORT = REPO_ROOT / "reports" / "orbslam3_fisheye_calibration_conversion_report.md"
CHECKPOINT = REPO_ROOT / "checkpoints" / "ORB1b_calibration_conversion_audit.json"
YAML = REPO_ROOT / "external_baselines" / "config" / "orbslam3_fisheye_cam0_scene01_seq03.yaml"
MANIFEST = REPO_ROOT / "checkpoints" / "final_clean_candidate_manifest.json"


class Orbslam3FisheyeCalibrationStaticTest(unittest.TestCase):
    def test_conversion_script_exists(self) -> None:
        self.assertTrue(TOOL.is_file())

    def test_report_checkpoint_yaml_paths_are_referenced(self) -> None:
        text = TOOL.read_text(encoding="utf-8")
        self.assertIn("reports/orbslam3_fisheye_calibration_conversion_report.md", text)
        self.assertIn("checkpoints/ORB1b_calibration_conversion_audit.json", text)
        self.assertIn("external_baselines/config/orbslam3_fisheye_cam0_scene01_seq03.yaml", text)
        self.assertTrue(REPORT.is_file())
        self.assertTrue(CHECKPOINT.is_file())

    def test_allowed_classifications_are_present(self) -> None:
        text = TOOL.read_text(encoding="utf-8")
        for classification in [
            "ORB1B_READY_FITTED_KB8",
            "ORB1B_READY_DIRECT_KB8",
            "ORB1B_CONVERSION_UNSUPPORTED",
            "ORB1B_BLOCKED_BY_DATASET_AUDIT",
            "ORB1B_ERROR",
            "DIRECT_KB8_CONVERSION_SUPPORTED",
            "FITTED_KB8_APPROXIMATION_SUPPORTED",
            "CONVERSION_UNSUPPORTED",
        ]:
            self.assertIn(classification, text)

    def test_script_prevents_direct_mapping_coeff_copy(self) -> None:
        text = TOOL.read_text(encoding="utf-8")
        self.assertIn("Never copy mapping_coeffs directly into k1/k2/k3/k4", text)
        self.assertIn("must not be copied directly", text)

    def test_generated_yaml_warning_text_is_present_in_script(self) -> None:
        text = TOOL.read_text(encoding="utf-8")
        self.assertIn("This file is generated from dataset polynomial fisheye calibration.", text)
        self.assertIn("It is a fitted/converted approximation for ORB-SLAM3 compatibility.", text)
        self.assertIn("It must not be treated as original factory KannalaBrandt8 calibration.", text)

    def test_no_fake_orbslam3_metrics_are_hardcoded(self) -> None:
        text = TOOL.read_text(encoding="utf-8")
        self.assertNotIn("7.352288", text)
        self.assertNotIn("1.327343", text)
        self.assertNotIn("0.932379", text)
        self.assertNotIn("scene01_seq03_est_tum.txt", text)

    def test_s5_locked_metrics_policy_files_are_not_modified(self) -> None:
        text = TOOL.read_text(encoding="utf-8")
        self.assertNotIn("S5_clean_tmag_calibration_policy.json", text)
        self.assertNotIn("final_clean_candidate_manifest.json", text)
        payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(payload["final_candidate_name"], "S5_clean_tmag_calibration_policy")
        self.assertAlmostEqual(float(payload["final_metrics"]["ATE"]), 7.352288, places=6)
        self.assertAlmostEqual(float(payload["final_metrics"]["drift"]), 1.327343, places=6)
        self.assertAlmostEqual(float(payload["final_metrics"]["path_ratio"]), 0.932379, places=6)

    def test_yaml_warning_if_yaml_written(self) -> None:
        payload = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
        if payload["output_yaml"]["written"]:
            self.assertTrue(YAML.is_file())
            text = YAML.read_text(encoding="utf-8")
            self.assertIn("fitted/converted approximation", text)
            self.assertIn("must not be treated as original factory KannalaBrandt8 calibration", text)


if __name__ == "__main__":
    unittest.main()
