from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TestAudit1ExperimentalCandidateValidityStatic(unittest.TestCase):
    def test_audit_tool_exists(self):
        self.assertTrue((ROOT / "tools/audit_experimental_candidate_validity.py").exists())

    def test_reports_and_checkpoint_referenced(self):
        text = (ROOT / "tools/audit_experimental_candidate_validity.py").read_text(encoding="utf-8")
        for token in [
            "reports/audit1_experimental_candidate_validity_and_noop_sweep.md",
            "reports/audit1_s5e15_best_candidate_validity.md",
            "checkpoints/AUDIT1_experimental_candidate_validity_and_noop_sweep.json",
        ]:
            self.assertIn(token.split("/")[-1], text)

    def test_candidate_and_authenticity_terms_present(self):
        text = (ROOT / "tools/audit_experimental_candidate_validity.py").read_text(encoding="utf-8")
        for token in [
            "S5E13_real_correspondence_signed_direction_candidate",
            "S5E14_observable_edge_direction_refinement_candidate",
            "S5E15_scale_deunderfit_antiparallel_candidate",
            "S5E16_confidence_calibrated_sign_scale_router",
            "S5E17_router_activation_threshold_repair_candidate",
            "S5E18_equirectangular_bearing_flow_candidate",
            "S5E19_rotation_compensated_multiframe_geometry_candidate",
            "training_authenticity",
            "export_authenticity",
            "evaluator_authenticity",
            "likely_noop",
            "fallback_to_previous",
            "hardcoded_previous_ref",
            "valid_as_best_self_developed_composite_candidate",
            "scale_improvement_authentic",
        ]:
            self.assertIn(token, text)

    def test_allowed_classifications_present(self):
        text = (ROOT / "tools/audit_experimental_candidate_validity.py").read_text(encoding="utf-8")
        for token in [
            "CANDIDATE_VALID_FRESH",
            "CANDIDATE_VALID_DERIVED_DECLARED",
            "CANDIDATE_SMOKE_ONLY",
            "CANDIDATE_NOOP_RISK",
            "CANDIDATE_FALLBACK_RISK",
            "CANDIDATE_EVAL_REF_MIXED",
            "CANDIDATE_LEAKAGE_RISK",
            "CANDIDATE_INSUFFICIENT_EVIDENCE",
            "AUDIT1_S5E15_VALID_BEST_CANDIDATE",
            "AUDIT1_S5E15_NEEDS_REPAIR",
            "AUDIT1_MULTIPLE_CANDIDATES_NOOP_RISK",
            "AUDIT1_EXPERIMENTAL_LINE_REQUIRES_REPAIR",
            "AUDIT1_AUDIT_ERROR",
        ]:
            self.assertIn(token, text)

    def test_no_training_no_metric_refresh_terms_present(self):
        text = (ROOT / "tools/audit_experimental_candidate_validity.py").read_text(encoding="utf-8")
        for token in [
            "candidate_validity",
            "pairwise_noop_diff",
            "s5_locked_metrics_policy_unchanged",
            "no-op",
            "smoke_only",
        ]:
            self.assertIn(token, text)


if __name__ == "__main__":
    unittest.main()
