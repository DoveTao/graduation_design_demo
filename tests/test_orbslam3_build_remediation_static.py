from __future__ import annotations

import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT = REPO_ROOT / "reports" / "orbslam3_build_remediation_report.md"
CHECKPOINT = REPO_ROOT / "checkpoints" / "ORB0b_orbslam3_build_remediation.json"
SCRIPT = REPO_ROOT / "scripts" / "remediate_orbslam3_build_orb0b.sh"


class Orbslam3BuildRemediationStaticTest(unittest.TestCase):
    def test_report_checkpoint_paths_are_referenced(self) -> None:
        self.assertTrue(REPORT.is_file())
        self.assertTrue(CHECKPOINT.is_file())
        text = REPORT.read_text(encoding="utf-8") + SCRIPT.read_text(encoding="utf-8")
        self.assertIn("checkpoints/ORB0b_orbslam3_build_remediation.json", text)
        self.assertIn("reports/orbslam3_build_remediation_report.md", str(REPORT.relative_to(REPO_ROOT)))

    def test_allowed_classifications_are_present(self) -> None:
        text = CHECKPOINT.read_text(encoding="utf-8") + REPORT.read_text(encoding="utf-8")
        for classification in [
            "ORB0_READY_AFTER_REMEDIATION",
            "ORB0B_CPP_STANDARD_PATCH_FAILED",
            "ORB0B_ISOLATED_PANGOLIN_FAILED",
            "ORB0B_ERROR",
        ]:
            self.assertIn(classification, text)

    def test_no_fake_orbslam3_metrics_are_hardcoded(self) -> None:
        text = "\n".join(
            [
                REPORT.read_text(encoding="utf-8"),
                CHECKPOINT.read_text(encoding="utf-8"),
                SCRIPT.read_text(encoding="utf-8"),
            ]
        )
        self.assertNotIn("7.352288", text)
        self.assertNotIn("1.327343", text)
        self.assertNotIn("0.932379", text)
        self.assertNotIn("scene01_seq03_est_tum.txt", text)

    def test_no_dataset_baseline_statement(self) -> None:
        text = REPORT.read_text(encoding="utf-8") + SCRIPT.read_text(encoding="utf-8")
        self.assertIn("It does not run ORB-SLAM3 on the dataset.", text)
        self.assertIn("It does not generate a trajectory.", text)
        self.assertIn("It does not produce baseline metrics.", text)

    def test_uses_external_orbslam3_source(self) -> None:
        payload = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
        source_path = payload["source_path"]
        self.assertEqual(source_path, "/home/dovetao/third_party/ORB_SLAM3")
        self.assertFalse(source_path.startswith(str(REPO_ROOT)))
        self.assertIn("/home/dovetao/third_party/ORB_SLAM3", SCRIPT.read_text(encoding="utf-8"))

    def test_remediation_helper_does_not_modify_s5_or_evaluator_paths(self) -> None:
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


if __name__ == "__main__":
    unittest.main()
