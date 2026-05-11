from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TestS5E20TrueKstepCompositionStatic(unittest.TestCase):
    def test_required_files_exist(self):
        for rel in [
            "configs/s5e20_true_kstep_composition_supervision.yaml",
            "tools/build_s5e20_kstep_training_pairs.py",
            "tools/train_s5e20_true_kstep_composition.py",
            "tools/export_s5e20_adjacent_dense_predictions.py",
            "tools/audit_s5e20_kstep_composition.py",
            "tools/evaluate_s5e20_traceable_dense.py",
            "tools/compare_s5e20_s5e19c_s5e15_orbslam3.py",
        ]:
            self.assertTrue((ROOT / rel).exists(), rel)

    def test_config_contains_required_terms(self):
        text = (ROOT / "configs/s5e20_true_kstep_composition_supervision.yaml").read_text(encoding="utf-8")
        for token in [
            "forbid_random_batch_proxy: true",
            "k_values: [1, 2, 3, 5]",
            "max_delta_norm_target: 0.20",
            "low_signal_tdir_weight",
            "use_sign_head: false",
            "uses_eval_gt_for_training: false",
            "uses_orbslam3_teacher: false",
            "S5E15_scale_deunderfit_antiparallel_candidate",
        ]:
            self.assertIn(token, text)

    def test_tools_reference_true_kstep_terms(self):
        text = (ROOT / "tools/train_s5e20_true_kstep_composition.py").read_text(encoding="utf-8")
        for token in [
            "uses_random_batch_proxy",
            "uses_true_contiguous_kstep_windows",
            "delta_tdir_l2_regularization",
            "signed_tdir_kstep",
            "_compose_rel",
        ]:
            self.assertIn(token, text)
        self.assertNotIn("ORB-SLAM3 trajectory label", text)

    def test_compare_and_report_paths_present(self):
        compare_text = (ROOT / "tools/compare_s5e20_s5e19c_s5e15_orbslam3.py").read_text(encoding="utf-8")
        self.assertIn("ORB-SLAM3", compare_text)
        self.assertIn("S5E19C", compare_text)
        self.assertIn("S5E15", compare_text)

    def test_allowed_classifications_present(self):
        text = (ROOT / "tools/evaluate_s5e20_traceable_dense.py").read_text(encoding="utf-8")
        for token in [
            "S5E20_TRUE_KSTEP_IMPROVED",
            "S5E20_TDIR_IMPROVED_PATH_NOT",
            "S5E20_PATH_IMPROVED_TDIR_NOT",
            "S5E20_REAL_TRAINING_NO_IMPROVEMENT",
            "S5E20_DELTA_STILL_TOO_AGGRESSIVE",
            "S5E20_KSTEP_DATASET_BLOCKED",
            "S5E20_EXPORT_BLOCKED",
            "S5E20_ERROR",
        ]:
            self.assertIn(token, text)


if __name__ == "__main__":
    unittest.main()
