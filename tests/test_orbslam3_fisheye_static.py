from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
TOOL = REPO_ROOT / "tools" / "audit_raw_fisheye_dataset.py"
REPORT = REPO_ROOT / "reports" / "orbslam3_fisheye_dataset_audit.md"
CHECKPOINT = REPO_ROOT / "checkpoints" / "ORB1a_fisheye_dataset_audit.json"


def load_tool_module():
    spec = importlib.util.spec_from_file_location("audit_raw_fisheye_dataset", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Orbslam3FisheyeStaticTest(unittest.TestCase):
    def test_audit_script_exists(self) -> None:
        self.assertTrue(TOOL.is_file())

    def test_report_and_checkpoint_paths_are_referenced(self) -> None:
        text = TOOL.read_text(encoding="utf-8")
        self.assertIn("reports/orbslam3_fisheye_dataset_audit.md", text)
        self.assertIn("checkpoints/ORB1a_fisheye_dataset_audit.json", text)
        self.assertTrue(REPORT.is_file())
        self.assertTrue(CHECKPOINT.is_file())

    def test_allowed_classifications_are_present(self) -> None:
        text = TOOL.read_text(encoding="utf-8")
        for classification in [
            "RAW_FISHEYE_READY",
            "RAW_FISHEYE_MISSING",
            "CALIBRATION_PARSE_FAILED",
            "TIMESTAMP_ALIGNMENT_ISSUE",
            "IMAGE_SIZE_MISMATCH",
        ]:
            self.assertIn(classification, text)

    def test_script_has_no_hardcoded_fake_orbslam3_metrics(self) -> None:
        text = TOOL.read_text(encoding="utf-8")
        self.assertNotIn("7.352288", text)
        self.assertNotIn("1.327343", text)
        self.assertNotIn("0.932379", text)
        self.assertNotIn("scene01_seq03_est_tum.txt", text)
        self.assertNotIn("fake trajectory", text.lower())

    def test_script_does_not_modify_s5_files(self) -> None:
        text = TOOL.read_text(encoding="utf-8")
        self.assertNotIn("S5_clean_tmag_calibration_policy.json", text)
        self.assertNotIn("final_clean_candidate_manifest.json", text)
        self.assertNotIn("eval_s5_clean_policy", text)
        self.assertNotIn("verify_final_s5_candidate", text)

    def test_parser_accepts_one_synthetic_cam_infos_line(self) -> None:
        module = load_tool_module()
        line = " ".join(str(i + 0.5) for i in range(18))
        parsed = module.parse_cam_infos_line(line)
        self.assertEqual(parsed["mapping_coeffs"], [0.5, 1.5, 2.5, 3.5])
        self.assertEqual(parsed["image_size"], [4, 6])
        self.assertEqual(len(parsed["stretch_matrix"]), 4)
        self.assertEqual(len(parsed["translation"]), 3)


if __name__ == "__main__":
    unittest.main()
