from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TestS5E21NoHarmObservabilityGatedRefinementStatic(unittest.TestCase):
    def test_required_files_exist(self):
        for rel in [
            "configs/s5e21_no_harm_observability_gated_refinement.yaml",
            "tools/build_s5e21_observability_gate_dataset.py",
            "tools/train_s5e21_no_harm_refinement.py",
            "tools/export_s5e21_adjacent_dense_predictions.py",
            "tools/audit_s5e21_no_harm_gate.py",
            "tools/evaluate_s5e21_traceable_dense.py",
            "tools/compare_s5e21_s5e20_s5e15_orbslam3.py",
        ]:
            self.assertTrue((ROOT / rel).exists(), rel)

    def test_config_terms_present(self):
        text = (ROOT / "configs/s5e21_no_harm_observability_gated_refinement.yaml").read_text(encoding="utf-8")
        for token in [
            "use_no_harm_gate: true",
            "use_observability_gate: true",
            "protect_s5e15_base_direction: true",
            "low_signal_gate_value: 0.0",
            "max_alpha: 0.05",
            "forbid_eval_gt_gate_tuning: true",
            "uses_orbslam3_teacher: false",
        ]:
            self.assertIn(token, text)

    def test_tool_terms_present(self):
        text = (ROOT / "tools/train_s5e21_no_harm_refinement.py").read_text(encoding="utf-8")
        for token in [
            "no_harm_margin",
            "uses_eval_gt_for_gate",
            "uses_external_router",
            "modified_train_edge_fraction",
            "delta_tdir_norm_train_mean",
        ]:
            self.assertIn(token, text)

    def test_allowed_classifications_present(self):
        text = (ROOT / "tools/evaluate_s5e21_traceable_dense.py").read_text(encoding="utf-8")
        for token in [
            "S5E21_NO_HARM_REFINEMENT_IMPROVED",
            "S5E21_SUBSET_IMPROVED_GLOBAL_NOT",
            "S5E21_NO_HARM_BUT_NO_IMPROVEMENT",
            "S5E21_NO_HARM_FAILED",
            "S5E21_GATE_DATASET_BLOCKED",
            "S5E21_EXPORT_BLOCKED",
            "S5E21_TRAINING_ERROR",
        ]:
            self.assertIn(token, text)

    def test_s5e15_reference_present(self):
        text = (ROOT / "tools/evaluate_s5e21_traceable_dense.py").read_text(encoding="utf-8")
        self.assertIn("50.3530", text)
        self.assertIn("0.1479", text)
        self.assertIn("3.9682", text)


if __name__ == "__main__":
    unittest.main()
