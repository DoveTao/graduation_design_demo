from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TestArch2MainlineRotationCompensatedSoftcorrGeometryStatic(unittest.TestCase):
    def test_required_files_exist(self):
        for rel in [
            "configs/arch2_mainline_rotation_compensated_softcorr_geometry.yaml",
            "tools/train_arch2_mainline_softcorr_geometry.py",
            "tools/export_arch2_adjacent_dense_predictions.py",
            "tools/evaluate_arch2_traceable_dense.py",
            "tools/audit_arch2_pose_convention_softcorr_noop.py",
            "tools/compare_arch2_s5e21_s5e20_s5e15_orbslam3.py",
        ]:
            self.assertTrue((ROOT / rel).exists(), rel)

    def test_config_terms_present(self):
        text = (ROOT / "configs/arch2_mainline_rotation_compensated_softcorr_geometry.yaml").read_text(encoding="utf-8")
        for token in [
            "use_rotation_compensation: true",
            "use_soft_correspondence_geometry: true",
            "use_W_ab: true",
            "use_W_ba: true",
            "residual_flow: normalize(W_ab @ bearing_b) - normalize(R_coarse @ bearing_a)",
            "use_observability_loss_weighting: true",
            "uses_eval_gt_for_training: false",
            "uses_orbslam3_teacher: false",
        ]:
            self.assertIn(token, text)

    def test_reference_terms_present(self):
        text = (ROOT / "tools/evaluate_arch2_traceable_dense.py").read_text(encoding="utf-8")
        for token in [
            "S5E21",
            "50.3530",
            "0.1479",
            "3.9682",
        ]:
            self.assertIn(token, text)

    def test_allowed_classifications_present(self):
        text = (ROOT / "tools/evaluate_arch2_traceable_dense.py").read_text(encoding="utf-8")
        for token in [
            "ARCH2_SOFTCORR_GEOMETRY_IMPROVED",
            "ARCH2_SUBSET_IMPROVED_GLOBAL_NOT",
            "ARCH2_NO_HARM_BUT_NO_IMPROVEMENT",
            "ARCH2_REAL_TRAINING_NO_IMPROVEMENT",
            "ARCH2_NOOP_GUARD_FAILED",
            "ARCH2_TRAINING_BLOCKED",
            "ARCH2_EXPORT_BLOCKED",
            "ARCH2_ERROR",
        ]:
            self.assertIn(token, text)

    def test_audit_terms_present(self):
        text = (ROOT / "tools/audit_arch2_pose_convention_softcorr_noop.py").read_text(encoding="utf-8")
        for token in [
            "twc_tcw_mismatch_suspected",
            "W_ab_used_in_motion_token",
            "R_coarse_used_in_rotation_compensation",
            "old_metrics_reuse_risk",
            "batch_first_scale_anchor_risk",
        ]:
            self.assertIn(token, text)


if __name__ == "__main__":
    unittest.main()
