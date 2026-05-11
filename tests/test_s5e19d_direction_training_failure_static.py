from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TestS5E19DDirectionTrainingFailureStatic(unittest.TestCase):
    def test_audit_tool_exists(self):
        self.assertTrue((ROOT / "tools/audit_s5e19d_direction_training_failure.py").exists())

    def test_report_and_checkpoint_referenced(self):
        text = (ROOT / "tools/audit_s5e19d_direction_training_failure.py").read_text(encoding="utf-8")
        for token in [
            "reports/s5e19d_direction_training_failure_audit.md",
            "checkpoints/S5E19D_direction_training_failure_audit.json",
            "S5E15_scale_deunderfit_antiparallel_candidate",
            "S5E19C_real_direction_training_no_fallback_candidate",
        ]:
            self.assertIn(token.split("/")[-1] if "/" in token else token, text)

    def test_audit_terms_present(self):
        text = (ROOT / "tools/audit_s5e19d_direction_training_failure.py").read_text(encoding="utf-8")
        for token in [
            "delta_tdir_audit",
            "sign_score_audit",
            "frame_convention_audit",
            "observability_weighting_audit",
            "multiframe_audit",
            "train_eval_generalization",
            "training_path_audit",
            "S5E19D_FRAME_CONVENTION_SUSPECTED",
            "S5E19D_TRAIN_OVERFIT_EVAL_DEGRADES",
            "S5E19D_DIRECTION_LOSS_NOT_LEARNING",
            "S5E19D_SIGN_HEAD_NOT_INFORMATIVE",
            "S5E19D_DELTA_TDIR_TOO_AGGRESSIVE",
            "S5E19D_OBSERVABILITY_WEIGHTING_BAD",
            "S5E19D_MULTIFRAME_LOSS_CONFLICT",
            "S5E19D_INSUFFICIENT_TRAINING_SIGNAL",
            "S5E19D_MIXED_FAILURE_MODES",
            "S5E19D_INSUFFICIENT_ARTIFACTS",
            "S5E19D_AUDIT_ERROR",
            "audit_only",
            "metrics_refreshed",
            "S5 locked metrics/policy unchanged",
        ]:
            self.assertIn(token, text)


if __name__ == "__main__":
    unittest.main()
