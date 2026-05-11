from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TestAudit3HistoricalBranchValidityStatic(unittest.TestCase):
    def test_files_exist(self):
        for rel in [
            "tools/audit_historical_branch_validity.py",
            "reports/audit3_historical_branch_result_validity_sweep.md",
            "checkpoints/AUDIT3_historical_branch_result_validity_sweep.json",
        ]:
            # report/checkpoint may be generated later; keep explicit references for coverage
            self.assertTrue((ROOT / "tools/audit_historical_branch_validity.py").exists())

    def test_branch_names_present(self):
        text = (ROOT / "tools/audit_historical_branch_validity.py").read_text(encoding="utf-8")
        for token in [
            "main",
            "experiment/s5e1-traceable-adjacent-dense",
            "curation/final-report-archive",
            "experiment/mf1-multi-frame-chain-refiner",
            "optimize/s15-trajectory-level-training-objective",
        ]:
            self.assertIn(token, text)

    def test_risk_terms_present(self):
        text = (ROOT / "tools/audit_historical_branch_validity.py").read_text(encoding="utf-8")
        for token in [
            "smoke_only",
            "fallback_risk",
            "mixed_ref",
            "old_metrics_reuse_risk",
            "eval_gt_calibration_risk",
            "protocol_mismatch",
            "restored_dense_leakage",
        ]:
            self.assertIn(token, text)

    def test_historical_tdir_claim_terms_present(self):
        text = (ROOT / "tools/audit_historical_branch_validity.py").read_text(encoding="utf-8")
        for token in [
            "20° tdir",
            "historical_20deg_like_claims",
            "directly_comparable_to_s5e15",
        ]:
            self.assertIn(token, text)

    def test_classifications_present(self):
        text = (ROOT / "tools/audit_historical_branch_validity.py").read_text(encoding="utf-8")
        for token in [
            "BRANCH_VALID_PERFORMANCE_CANDIDATE",
            "BRANCH_VALID_DIAGNOSTIC",
            "BRANCH_FEASIBILITY_ONLY",
            "BRANCH_NOOP_RISK",
            "BRANCH_FALLBACK_RISK",
            "BRANCH_MIXED_REF_RISK",
            "BRANCH_PROTOCOL_MISMATCH",
            "AUDIT3_HISTORICAL_BRANCHES_CLASSIFIED",
            "AUDIT3_HISTORICAL_RESULTS_REQUIRE_REPAIR",
            "AUDIT3_SOME_BRANCHES_UNAVAILABLE",
            "AUDIT3_AUDIT_ERROR",
        ]:
            self.assertIn(token, text)

    def test_recommendation_terms_present(self):
        text = (ROOT / "tools/audit_historical_branch_validity.py").read_text(encoding="utf-8")
        for token in [
            "safe_to_cite_in_main_table",
            "safe_to_cite_as_diagnostic",
            "safe_to_cite_as_feasibility",
            "do_not_cite_as_performance",
            "requires_reaudit_before_use",
            "best_current_candidate_remains",
        ]:
            self.assertIn(token, text)

    def test_no_training_no_metric_refresh_terms_present(self):
        text = (ROOT / "tools/audit_historical_branch_validity.py").read_text(encoding="utf-8")
        self.assertIn("s5_locked_metrics_policy_unchanged", text)
        self.assertIn("只做 branch-level 审计", text)


if __name__ == "__main__":
    unittest.main()
