from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TestGen4TrueImagePairInferenceRecoveryStatic(unittest.TestCase):
    def test_files_exist(self):
        for rel in [
            "tools/audit_gen4_true_image_pair_inference_recovery.py",
            "tools/export_gen4_360dvo_true_model_predictions.py",
        ]:
            self.assertTrue((ROOT / rel).exists(), rel)

    def test_required_terms_present(self):
        text = "\n".join(
            [
                (ROOT / "tools/audit_gen4_true_image_pair_inference_recovery.py").read_text(encoding="utf-8"),
                (ROOT / "tools/export_gen4_360dvo_true_model_predictions.py").read_text(encoding="utf-8"),
            ]
        )
        for token in [
            "GEN4_TRUE_IMAGE_PAIR_INFERENCE_READY",
            "GEN4_MODEL_CODE_FOUND_WEIGHTS_MISSING",
            "GEN4_WEIGHTS_FOUND_INPUT_PROTOCOL_UNSUPPORTED",
            "GEN4_SCENE_SPECIFIC_ONLY_CONFIRMED",
            "GEN4_NO_GENERAL_INFERENCE_PATH_FOUND",
            "GEN4_ERROR",
            "no fine-tune",
            "not fully reusable external inference model",
            "T57b_no_dt_multiscale_tmag_head_400/final.pt",
            "S5E15_scale_deunderfit_antiparallel_candidate",
            "R_pred_BA",
            "tdir_pred_B",
            "tmag_pred",
            "s5_locked_metrics_policy_unchanged",
        ]:
            self.assertIn(token, text)


if __name__ == "__main__":
    unittest.main()
