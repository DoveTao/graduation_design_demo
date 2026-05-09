import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
ROOT = REPO_ROOT / "reports_curated"
README = ROOT / "README.md"
MANIFEST = ROOT / "MANIFEST.json"


class TestCuratedReportsMaterializedStatic(unittest.TestCase):
    def test_expected_files_exist(self):
        self.assertTrue(README.exists(), str(README))
        self.assertTrue(MANIFEST.exists(), str(MANIFEST))
        self.assertTrue((ROOT / "final").exists())
        self.assertTrue((ROOT / "baselines").exists())
        self.assertTrue((ROOT / "diagnostics").exists())
        self.assertTrue((ROOT / "negative_experiments" / "jrt1").exists())
        self.assertTrue((ROOT / "negative_experiments" / "mf1").exists())

    def test_manifest_flags(self):
        payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertTrue(payload["no_original_reports_deleted"])
        self.assertTrue(payload["no_original_reports_moved"])
        self.assertTrue(payload["no_branches_merged"])
        self.assertTrue(payload["no_cherry_picks"])
        self.assertAlmostEqual(payload["s5_locked_metrics"]["ATE"], 7.352288, places=9)
        self.assertAlmostEqual(payload["s5_locked_metrics"]["drift"], 1.327343, places=9)
        self.assertAlmostEqual(payload["s5_locked_metrics"]["path_ratio"], 0.932379, places=9)

    def test_readme_core_notes(self):
        text = README.read_text(encoding="utf-8")
        self.assertIn("Curated Reports Archive", text)
        self.assertIn("S5 remains final clean candidate", text)
        self.assertIn("ORB-SLAM3 is an external baseline", text)
        self.assertIn("JRT1 and MF1 are negative experiments", text)


if __name__ == "__main__":
    unittest.main()
