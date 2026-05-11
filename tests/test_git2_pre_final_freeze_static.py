from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TestGit2PreFinalFreezeStatic(unittest.TestCase):
    def test_tool_exists(self):
        self.assertTrue((ROOT / "tools/audit_git2_pre_final_freeze.py").exists())

    def test_report_and_checkpoint_referenced(self):
        text = (ROOT / "tools/audit_git2_pre_final_freeze.py").read_text(encoding="utf-8")
        for token in [
            "reports/git2_pre_final_freeze_cleanup_report.md",
            "checkpoints/GIT2_pre_final_freeze_cleanup.json",
        ]:
            self.assertIn(token.split("/")[-1], text)

    def test_classification_terms_present(self):
        text = (ROOT / "tools/audit_git2_pre_final_freeze.py").read_text(encoding="utf-8")
        for token in [
            "should_commit",
            "should_ignore_or_local_only",
            "should_restore",
            "needs_manual_review",
            "S5E13",
            "S5E14",
            "S5E15",
            "S5D11",
            "GIT2_WORKING_TREE_CLEAN",
            "GIT2_WORKING_TREE_CLEAN_WITH_LOCAL_IGNORED_ARTIFACTS",
            "GIT2_PARTIAL_CLEAN_REMAINING_MANUAL_REVIEW",
            "GIT2_BLOCKED_BY_UNCLASSIFIED_FILES",
            "GIT2_ERROR",
        ]:
            self.assertIn(token, text)

    def test_ignore_and_safety_terms_present(self):
        text = (ROOT / "tools/audit_git2_pre_final_freeze.py").read_text(encoding="utf-8")
        for token in [
            "router_decisions.jsonl",
            "spherical_bearing_flow_features.jsonl",
            "edge_provenance.jsonl",
            "_tum",
            "correspondence_weighted_dataset.json",
            "s5_locked_metrics_policy_unchanged",
        ]:
            self.assertIn(token, text)
        self.assertNotIn("git add .", text)


if __name__ == "__main__":
    unittest.main()
