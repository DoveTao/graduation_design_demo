from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TestStruct1MainlineRebuildWithGeometryTokensStatic(unittest.TestCase):
    def test_required_files_exist(self):
        for rel in [
            "configs/struct1_mainline_rebuild_with_geometry_tokens.yaml",
            "tools/train_struct1_geometry_token_pose_solver.py",
            "tools/export_struct1_adjacent_dense_predictions.py",
            "tools/evaluate_struct1_traceable_dense.py",
            "tools/audit_struct1_geometry_token_integrity.py",
            "tools/compare_struct1_arch2p_arch2_s5e15_orbslam3.py",
        ]:
            self.assertTrue((ROOT / rel).exists(), rel)

    def test_config_terms_present(self):
        text = (ROOT / "configs/struct1_mainline_rebuild_with_geometry_tokens.yaml").read_text(encoding="utf-8")
        for token in [
            "use_geometry_token_pose_solver: true",
            "use_coarse_to_fine: true",
            "use_W_ab: true",
            "use_W_ba: true",
            "bearing_a",
            "matched_bearing_b",
            "epipolar_residual",
            "geometry_aware_decoder",
            "avoid_pooled_feature_only_tdir: true",
            "used_as_router: false",
            "uses_eval_gt_for_training: false",
            "uses_orbslam3_teacher: false",
            "allow_fallback_to_s5e15_prediction: false",
        ]:
            self.assertIn(token, text)

    def test_train_export_terms_present(self):
        text = (ROOT / "tools/train_struct1_geometry_token_pose_solver.py").read_text(encoding="utf-8")
        for token in [
            "optimizer_step_count",
            "\"W_ab_used_in_pose_solver\": True",
            "\"tdir_from_geometry_tokens\": True",
            "\"fine_stage_used\": True",
            "STRUCT1_REAL_TRAINING_COMPLETE",
            "STRUCT1_SHORT_REAL_TRAINING",
            "STRUCT1_TRAINING_BLOCKED",
            "STRUCT1_TRAINING_ERROR",
        ]:
            self.assertIn(token, text)
        export_text = (ROOT / "tools/export_struct1_adjacent_dense_predictions.py").read_text(encoding="utf-8")
        for token in [
            "source_model",
            "STRUCT1",
            "uses_geometry_tokens",
            "W_ab_used",
            "tdir_from_geometry_tokens",
            "uses_eval_gt_for_prediction",
        ]:
            self.assertIn(token, export_text)

    def test_audit_and_eval_terms_present(self):
        text = (ROOT / "tools/audit_struct1_geometry_token_integrity.py").read_text(encoding="utf-8")
        for token in [
            "spherical_token_backbone_used",
            "tangent_geometric_channels_used",
            "W_ab_used_in_pose_solver",
            "tdir_from_geometry_tokens",
            "eval_GT_not_used_for_prediction",
            "ORB_teacher_not_used",
        ]:
            self.assertIn(token, text)
        eval_text = (ROOT / "tools/evaluate_struct1_traceable_dense.py").read_text(encoding="utf-8")
        for token in [
            "50.3530",
            "0.1479",
            "3.9682",
            "STRUCT1_GEOMETRY_TOKENS_IMPROVED",
            "STRUCT1_SUBSET_IMPROVED_GLOBAL_NOT",
            "STRUCT1_REAL_TRAINING_NO_IMPROVEMENT",
            "STRUCT1_INTEGRITY_GUARD_FAILED",
            "STRUCT1_TRAINING_BLOCKED",
            "STRUCT1_EXPORT_BLOCKED",
            "STRUCT1_ERROR",
        ]:
            self.assertIn(token, eval_text)


if __name__ == "__main__":
    unittest.main()
