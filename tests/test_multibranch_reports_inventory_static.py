import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT = REPO_ROOT / "reports" / "multibranch_reports_inventory.md"
JSON_PATH = REPO_ROOT / "checkpoints" / "REPORT2_multibranch_reports_inventory.json"


class TestMultibranchReportsInventoryStatic(unittest.TestCase):
    def test_files_exist(self):
        self.assertTrue(REPORT.exists(), str(REPORT))
        self.assertTrue(JSON_PATH.exists(), str(JSON_PATH))

    def test_report_sections_exist(self):
        text = REPORT.read_text(encoding="utf-8")
        self.assertIn("Multibranch Reports Inventory", text)
        self.assertIn("16-branch read-only inventory", text)
        self.assertIn("## Branches Audited", text)
        self.assertIn("## Cross-Branch Report Matrix", text)
        self.assertIn("no files moved = true", text)
        self.assertIn("no files deleted = true", text)
        self.assertIn("no branches merged = true", text)
        self.assertIn("no cherry-picks = true", text)

    def test_json_contract(self):
        payload = json.loads(JSON_PATH.read_text(encoding="utf-8"))
        self.assertTrue(payload["no_files_moved"])
        self.assertTrue(payload["no_files_deleted"])
        self.assertTrue(payload["no_branches_merged"])
        self.assertTrue(payload["no_cherry_picks"])
        self.assertEqual(len(payload["branches_audited"]), 16)

    def test_locked_s5_metrics_unchanged(self):
        manifest = json.loads((REPO_ROOT / "checkpoints" / "final_clean_candidate_manifest.json").read_text(encoding="utf-8"))
        self.assertAlmostEqual(manifest["final_metrics"]["ATE"], 7.352288, places=9)
        self.assertAlmostEqual(manifest["final_metrics"]["drift"], 1.327343, places=9)
        self.assertAlmostEqual(manifest["final_metrics"]["path_ratio"], 0.932379, places=9)


if __name__ == "__main__":
    unittest.main()
