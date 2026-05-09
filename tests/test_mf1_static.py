import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
MF1_FILES = [
    REPO_ROOT / "checkpoints" / "MF1_chain_refiner_config.json",
    REPO_ROOT / "tools" / "mf1_build_chain_dataset.py",
    REPO_ROOT / "tools" / "mf1_train_chain_refiner.py",
    REPO_ROOT / "tools" / "mf1_eval_chain_refiner.py",
    REPO_ROOT / "scripts" / "run_mf1a_chain_refiner_smoke.sh",
]


class TestMF1Static(unittest.TestCase):
    def test_expected_files_exist(self):
        for path in MF1_FILES:
            self.assertTrue(path.exists(), str(path))

    def test_runner_guardrails(self):
        text = (REPO_ROOT / "scripts" / "run_mf1a_chain_refiner_smoke.sh").read_text(encoding="utf-8")
        self.assertIn("set -euo pipefail", text)
        self.assertIn("verify_final_candidate.sh", text)
        self.assertIn("checkpoints/MF1a_chain_refiner_smoke_results.json", text)
        self.assertIn("reports/MF1a_chain_refiner_smoke_report.md", text)

    def test_config_locked_s5_metrics(self):
        cfg = json.loads((REPO_ROOT / "checkpoints" / "MF1_chain_refiner_config.json").read_text(encoding="utf-8"))
        self.assertEqual(cfg["base_candidate"], "S5_clean_tmag_calibration_policy")
        self.assertAlmostEqual(cfg["locked_s5_metrics"]["ATE"], 7.352288, places=9)
        self.assertAlmostEqual(cfg["locked_s5_metrics"]["drift"], 1.327343, places=9)
        self.assertAlmostEqual(cfg["locked_s5_metrics"]["path_ratio"], 0.932379, places=9)
        self.assertTrue(cfg["gate"]["no_final_test"])

    def test_no_disallowed_claims(self):
        combined = "\n".join(path.read_text(encoding="utf-8") for path in MF1_FILES)
        lower = combined.lower()
        self.assertNotIn("clean " + "gain", lower)
        self.assertNotIn("replaces " + "s5", lower)
        self.assertNotIn("new final " + "candidate", lower)
        self.assertNotIn("practical " + "ready", lower)
        self.assertNotIn("S" + "20", combined)


if __name__ == "__main__":
    unittest.main()
