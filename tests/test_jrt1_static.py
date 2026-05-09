import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
JRT1_FILES = [
    REPO_ROOT / "tools" / "jrt1_build_pair_dataset.py",
    REPO_ROOT / "tools" / "jrt1_train_joint_rtdir_refiner.py",
    REPO_ROOT / "tools" / "jrt1_eval_joint_rtdir_refiner.py",
    REPO_ROOT / "scripts" / "run_jrt1a_framework_smoke.sh",
    REPO_ROOT / "checkpoints" / "JRT1_joint_rtdir_refiner_config.json",
]


class TestJRT1Static(unittest.TestCase):
    def test_expected_files_exist(self):
        for rel in [
            "tools/jrt1_build_pair_dataset.py",
            "tools/jrt1_train_joint_rtdir_refiner.py",
            "tools/jrt1_eval_joint_rtdir_refiner.py",
            "scripts/run_jrt1a_framework_smoke.sh",
            "checkpoints/JRT1_joint_rtdir_refiner_config.json",
        ]:
            self.assertTrue((REPO_ROOT / rel).exists(), rel)

    def test_runner_guardrails_and_outputs(self):
        text = (REPO_ROOT / "scripts" / "run_jrt1a_framework_smoke.sh").read_text(encoding="utf-8")
        self.assertIn("set -euo pipefail", text)
        self.assertIn("verify_final_candidate.sh", text)
        self.assertIn("checkpoints/JRT1a_framework_smoke_results.json", text)
        self.assertIn("reports/JRT1a_framework_smoke_report.md", text)

    def test_config_locked_s5_metrics_unchanged(self):
        cfg = json.loads((REPO_ROOT / "checkpoints" / "JRT1_joint_rtdir_refiner_config.json").read_text(encoding="utf-8"))
        self.assertEqual(cfg["base_candidate"], "S5_clean_tmag_calibration_policy")
        self.assertEqual(cfg["base_policy_path"], "checkpoints/S5_clean_tmag_calibration_policy.json")
        self.assertEqual(cfg["manifest_path"], "checkpoints/final_clean_candidate_manifest.json")
        self.assertAlmostEqual(cfg["locked_s5_metrics"]["ATE"], 7.352288, places=9)
        self.assertAlmostEqual(cfg["locked_s5_metrics"]["drift"], 1.327343, places=9)
        self.assertAlmostEqual(cfg["locked_s5_metrics"]["path_ratio"], 0.932379, places=9)
        self.assertTrue(cfg["gate"]["no_final_test"])
        self.assertTrue(cfg["gate"]["no_candidate_reselection"])

    def test_jrt1_files_do_not_write_locked_s5_artifacts(self):
        for path in JRT1_FILES:
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("OUT_POLICY_PATH", text)
            self.assertNotIn("final_metrics\" :", text)
            if path.name.startswith("jrt1_"):
                self.assertNotIn(".write_text(json.dumps(policy", text)
                self.assertNotIn("final_clean_candidate_manifest.json\", \"w", text)

    def test_no_disallowed_claims_or_stage_strings(self):
        combined = "\n".join(path.read_text(encoding="utf-8") for path in JRT1_FILES)
        self.assertNotIn("S" + "20", combined)
        self.assertNotIn("replace " + "s5", combined.lower())
        self.assertNotIn("replaces " + "s5", combined.lower())
        self.assertNotIn("new final " + "candidate", combined.lower())
        self.assertNotIn("clean " + "gain", combined.lower())
        self.assertNotIn("is practical-ready", combined.lower())
        self.assertNotIn("practical " + "ready", combined.lower())


if __name__ == "__main__":
    unittest.main()
