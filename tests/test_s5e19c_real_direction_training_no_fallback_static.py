from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TestS5E19CRealDirectionTrainingNoFallbackStatic(unittest.TestCase):
    def test_s5e19c_required_files_exist(self):
        for rel in [
            "configs/s5e19c_real_direction_training_no_fallback.yaml",
            "tools/train_s5e19c_real_direction_model.py",
            "tools/export_s5e19c_no_fallback_predictions.py",
            "tools/evaluate_s5e19c_traceable_dense.py",
            "tools/audit_s5e19c_noop_guard.py",
            "tools/compare_s5e19c_s5e19_s5e15_orbslam3.py",
        ]:
            self.assertTrue((ROOT / rel).exists(), rel)

    def test_s5e19c_config_terms_present(self):
        text = (ROOT / "configs/s5e19c_real_direction_training_no_fallback.yaml").read_text(encoding="utf-8")
        for token in [
            "real_training_required: true",
            "smoke_policy_allowed: false",
            "require_optimizer_step: true",
            "require_learned_weights: true",
            "forbid_s5e15_direction_fallback: true",
            "require_nonzero_delta_tdir: true",
            "require_nonconstant_sign_score: true",
        ]:
            self.assertIn(token, text)

    def test_s5e19c_training_export_audit_terms_present(self):
        train_text = (ROOT / "tools/train_s5e19c_real_direction_model.py").read_text(encoding="utf-8")
        export_text = (ROOT / "tools/export_s5e19c_no_fallback_predictions.py").read_text(encoding="utf-8")
        audit_text = (ROOT / "tools/audit_s5e19c_noop_guard.py").read_text(encoding="utf-8")
        for token in ["optimizer_step_count", "learned_weights_saved", "torch.save"]:
            self.assertIn(token, train_text)
        for token in ["uses_s5e15_direction_fallback", "delta_tdir", "sign_score", "uses_s5e15_scale_prior_reference"]:
            self.assertIn(token, export_text)
        for token in ["delta_tdir_zero_count", "sign_score_unique_count", "fallback_to_s5e15_direction", "evaluator_hardcoded_s5e15_metrics"]:
            self.assertIn(token, audit_text)

    def test_s5e19c_classifications_present(self):
        text = (ROOT / "tools/evaluate_s5e19c_traceable_dense.py").read_text(encoding="utf-8")
        for token in [
            "S5E19C_REAL_TRAINING_IMPROVED",
            "S5E19C_REAL_TRAINING_NO_IMPROVEMENT",
            "S5E19C_NOOP_GUARD_FAILED",
            "S5E19C_TRAINING_BLOCKED",
            "S5E19C_EXPORT_BLOCKED",
            "S5E19C_ERROR",
        ]:
            self.assertIn(token, text)


if __name__ == "__main__":
    unittest.main()
