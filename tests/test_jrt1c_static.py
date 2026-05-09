import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
JRT1C_FILES = [
    REPO_ROOT / "tools" / "jrt1_eval_train_cv_trajectory_proxy.py",
    REPO_ROOT / "scripts" / "run_jrt1c_train_cv_trajectory_proxy.sh",
]


class TestJRT1CStatic(unittest.TestCase):
    def test_expected_files_exist(self):
        self.assertTrue((REPO_ROOT / "tools" / "jrt1_eval_train_cv_trajectory_proxy.py").exists())
        self.assertTrue((REPO_ROOT / "scripts" / "run_jrt1c_train_cv_trajectory_proxy.sh").exists())

    def test_runner_guardrails_and_outputs(self):
        text = (REPO_ROOT / "scripts" / "run_jrt1c_train_cv_trajectory_proxy.sh").read_text(encoding="utf-8")
        self.assertIn("set -euo pipefail", text)
        self.assertIn("verify_final_candidate.sh", text)
        self.assertIn("checkpoints/JRT1c_train_cv_trajectory_proxy_results.json", text)
        self.assertIn("reports/JRT1c_train_cv_trajectory_proxy_report.md", text)

    def test_no_final_test_and_no_replacement_claim(self):
        combined = "\n".join(path.read_text(encoding="utf-8") for path in JRT1C_FILES)
        lower = combined.lower()
        self.assertIn("no final test", lower)
        self.assertNotIn("replace " + "s5", lower)
        self.assertNotIn("replaces " + "s5", lower)
        self.assertNotIn("new final " + "candidate", lower)
        self.assertNotIn("clean " + "gain", lower)
        self.assertNotIn("is practical-ready", lower)
        self.assertNotIn("practical " + "ready", lower)
        self.assertNotIn("S" + "20", combined)

    def test_locked_s5_metrics_preserved_in_config_and_results_if_present(self):
        cfg = json.loads((REPO_ROOT / "checkpoints" / "JRT1b_train_cv_config.json").read_text(encoding="utf-8"))
        payloads = [cfg]
        result_path = REPO_ROOT / "checkpoints" / "JRT1c_train_cv_trajectory_proxy_results.json"
        if result_path.exists():
            payloads.append(json.loads(result_path.read_text(encoding="utf-8")))
        for payload in payloads:
            metrics = payload["locked_s5_metrics"]
            self.assertAlmostEqual(metrics["ATE"], 7.352288, places=9)
            self.assertAlmostEqual(metrics["drift"], 1.327343, places=9)
            self.assertAlmostEqual(metrics["path_ratio"], 0.932379, places=9)


if __name__ == "__main__":
    unittest.main()
