from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TestGen3360DVOExternalInferenceBridgeStatic(unittest.TestCase):
    def test_files_exist(self):
        for rel in [
            "configs/gen3_360dvo_external_inference_bridge.yaml",
            "tools/audit_gen3_s5e15_inference_availability.py",
            "tools/export_gen3_360dvo_s5e15_predictions.py",
            "tools/evaluate_gen3_360dvo_external_predictions.py",
            "tools/compare_gen3_360dvo_data2_s5e15.py",
        ]:
            self.assertTrue((ROOT / rel).exists(), rel)

    def test_required_terms_present(self):
        text = "\n".join(
            [
                (ROOT / "configs/gen3_360dvo_external_inference_bridge.yaml").read_text(encoding="utf-8"),
                (ROOT / "tools/audit_gen3_s5e15_inference_availability.py").read_text(encoding="utf-8"),
                (ROOT / "tools/export_gen3_360dvo_s5e15_predictions.py").read_text(encoding="utf-8"),
                (ROOT / "tools/evaluate_gen3_360dvo_external_predictions.py").read_text(encoding="utf-8"),
                (ROOT / "tools/compare_gen3_360dvo_data2_s5e15.py").read_text(encoding="utf-8"),
            ]
        )
        for token in [
            "no_training",
            "fine_tune: false",
            "360DVO",
            "field",
            "pair_manifest_smoke.jsonl",
            "S5E15_scale_deunderfit_antiparallel_candidate",
            "R_BA",
            "tdir_pred_B",
            "tmag_pred",
            "forbid_gt_scale_calibration",
            "GEN3_360DVO_EXTERNAL_EVAL_READY",
            "GEN3_360DVO_COMPONENT_METRICS_READY",
            "GEN3_360DVO_TRAJECTORY_EVAL_READY",
            "GEN3_INFERENCE_BRIDGE_BLOCKED",
            "GEN3_MODEL_WEIGHTS_MISSING",
            "GEN3_SCENE_SPECIFIC_EXPORT_ONLY",
            "GEN3_INPUT_PROTOCOL_UNSUPPORTED",
            "GEN3_EVAL_GT_DEPENDENCY_DETECTED",
            "GEN3_ERROR",
            "s5_locked_metrics_policy_unchanged",
        ]:
            self.assertIn(token, text)


if __name__ == "__main__":
    unittest.main()
