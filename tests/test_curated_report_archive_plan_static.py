import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT = REPO_ROOT / "reports" / "curated_report_archive_plan.md"
JSON_PATH = REPO_ROOT / "checkpoints" / "REPORT3_curated_report_archive_plan.json"


class TestCuratedReportArchivePlanStatic(unittest.TestCase):
    def test_files_exist(self):
        self.assertTrue(REPORT.exists(), str(REPORT))
        self.assertTrue(JSON_PATH.exists(), str(JSON_PATH))

    def test_report_sections_exist(self):
        text = REPORT.read_text(encoding="utf-8")
        self.assertIn("Curated Report Archive Plan", text)
        self.assertIn("## Proposed Curated Archive Structure", text)
        self.assertIn("## Canonical Report Selection", text)
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
        self.assertIn("canonical_report_selection", payload)

    def test_locked_s5_metrics_unchanged(self):
        manifest = json.loads((REPO_ROOT / "checkpoints" / "final_clean_candidate_manifest.json").read_text(encoding="utf-8"))
        self.assertAlmostEqual(manifest["final_metrics"]["ATE"], 7.352288, places=9)
        self.assertAlmostEqual(manifest["final_metrics"]["drift"], 1.327343, places=9)
        self.assertAlmostEqual(manifest["final_metrics"]["path_ratio"], 0.932379, places=9)


if __name__ == "__main__":
    unittest.main()
