import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
INDEX = REPO_ROOT / "reports" / "REPORT_INDEX.md"
AUDIT = REPO_ROOT / "reports" / "reports_inventory_audit.md"
JSON_AUDIT = REPO_ROOT / "checkpoints" / "REPORT1_reports_inventory_audit.json"


class TestReportsIndexStatic(unittest.TestCase):
    def test_files_exist(self):
        self.assertTrue(INDEX.exists(), str(INDEX))
        self.assertTrue(AUDIT.exists(), str(AUDIT))
        self.assertTrue(JSON_AUDIT.exists(), str(JSON_AUDIT))

    def test_index_sections_exist(self):
        text = INDEX.read_text(encoding="utf-8")
        self.assertIn("## Final / Thesis Reports", text)
        self.assertIn("## Ablation and Baseline Reports", text)
        self.assertIn("## Negative Experiment Reports", text)
        self.assertIn("## Archive Candidates", text)

    def test_no_files_moved_deleted_statement(self):
        combined = INDEX.read_text(encoding="utf-8") + "\n" + AUDIT.read_text(encoding="utf-8")
        lower = combined.lower()
        self.assertIn("do not move files", lower)
        self.assertIn("no files moved: `true`", lower)
        self.assertIn("no files deleted: `true`", lower)

    def test_locked_s5_metrics_unchanged(self):
        payload = json.loads(JSON_AUDIT.read_text(encoding="utf-8"))
        self.assertEqual(payload["total_reports"], 72)
        self.assertTrue(payload["no_files_moved"])
        self.assertTrue(payload["no_files_deleted"])
        manifest = json.loads((REPO_ROOT / "checkpoints" / "final_clean_candidate_manifest.json").read_text(encoding="utf-8"))
        self.assertAlmostEqual(manifest["final_metrics"]["ATE"], 7.352288, places=9)
        self.assertAlmostEqual(manifest["final_metrics"]["drift"], 1.327343, places=9)
        self.assertAlmostEqual(manifest["final_metrics"]["path_ratio"], 0.932379, places=9)


if __name__ == "__main__":
    unittest.main()
