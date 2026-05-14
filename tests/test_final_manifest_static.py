from __future__ import annotations

import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "checkpoints" / "final_clean_candidate_manifest.json"


class FinalManifestStaticTest(unittest.TestCase):
    def test_final_manifest_exists_and_is_readable(self) -> None:
        self.assertTrue(MANIFEST_PATH.is_file())
        obj = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        self.assertEqual(obj["final_candidate_name"], "S5_clean_tmag_calibration_policy")
        self.assertEqual(obj["policy_path"], "checkpoints/S5_clean_tmag_calibration_policy.json")

    def test_final_manifest_locked_metrics_unchanged(self) -> None:
        obj = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        metrics = obj["final_metrics"]
        self.assertEqual(metrics["ATE"], 7.352288)
        self.assertEqual(metrics["drift"], 1.327343)
        self.assertEqual(metrics["path_ratio"], 0.932379)


if __name__ == "__main__":
    unittest.main()
