from __future__ import annotations

import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = REPO_ROOT / "checkpoints" / "S5_clean_tmag_calibration_policy.json"


class FinalS5PolicyStaticTest(unittest.TestCase):
    def test_s5_policy_exists_and_is_readable(self) -> None:
        self.assertTrue(POLICY_PATH.is_file())
        obj = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
        self.assertEqual(obj["name"], "S5_clean_tmag_calibration_policy")
        self.assertEqual(obj["status"], "frozen final clean policy candidate")
        self.assertTrue(obj["base_checkpoint_path"])

    def test_s5_policy_locked_metrics_unchanged(self) -> None:
        obj = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
        metrics = obj["expected_metrics"]
        self.assertEqual(metrics["ATE"], 7.352288)
        self.assertEqual(metrics["drift"], 1.327343)
        self.assertEqual(metrics["path_ratio"], 0.932379)


if __name__ == "__main__":
    unittest.main()
