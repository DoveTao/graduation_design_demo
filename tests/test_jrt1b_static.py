import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
JRT1B_FILES = [
    REPO_ROOT / "checkpoints" / "JRT1b_train_cv_config.json",
    REPO_ROOT / "tools" / "jrt1_train_cv_joint_rtdir_refiner.py",
    REPO_ROOT / "tools" / "jrt1_eval_train_cv_joint_rtdir_refiner.py",
    REPO_ROOT / "scripts" / "run_jrt1b_train_cv.sh",
]


class TestJRT1BStatic(unittest.TestCase):
    def test_expected_files_exist(self):
        for rel in [
            "checkpoints/JRT1b_train_cv_config.json",
            "tools/jrt1_train_cv_joint_rtdir_refiner.py",
            "tools/jrt1_eval_train_cv_joint_rtdir_refiner.py",
            "scripts/run_jrt1b_train_cv.sh",
        ]:
            self.assertTrue((REPO_ROOT / rel).exists(), rel)

    def test_runner_guardrails_and_outputs(self):
        text = (REPO_ROOT / "scripts" / "run_jrt1b_train_cv.sh").read_text(encoding="utf-8")
        self.assertIn("set -euo pipefail", text)
        self.assertIn("verify_final_candidate.sh", text)
        self.assertIn("checkpoints/JRT1b_train_cv_results.json", text)
        self.assertIn("reports/JRT1b_train_cv_report.md", text)

    def test_config_s5_metrics_and_gate(self):
        cfg = json.loads((REPO_ROOT / "checkpoints" / "JRT1b_train_cv_config.json").read_text(encoding="utf-8"))
        self.assertTrue(cfg["gate"]["no_final_test_unless_gate_pass"])
        self.assertEqual(cfg["base_candidate"], "S5_clean_tmag_calibration_policy")
        self.assertEqual(cfg["base_policy_path"], "checkpoints/S5_clean_tmag_calibration_policy.json")
        self.assertEqual(cfg["manifest_path"], "checkpoints/final_clean_candidate_manifest.json")
        self.assertAlmostEqual(cfg["locked_s5_metrics"]["ATE"], 7.352288, places=9)
        self.assertAlmostEqual(cfg["locked_s5_metrics"]["drift"], 1.327343, places=9)
        self.assertAlmostEqual(cfg["locked_s5_metrics"]["path_ratio"], 0.932379, places=9)

    def test_no_disallowed_claims_or_stage_strings(self):
        combined = "\n".join(path.read_text(encoding="utf-8") for path in JRT1B_FILES)
        self.assertNotIn("S" + "20", combined)
        lower = combined.lower()
        self.assertNotIn("is practical-ready", lower)
        self.assertNotIn("replace " + "s5", lower)
        self.assertNotIn("replaces " + "s5", lower)
        self.assertNotIn("new final " + "candidate", lower)
        self.assertNotIn("clean " + "gain", lower)


if __name__ == "__main__":
    unittest.main()
