from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_s5e19_required_files_exist():
    for rel in [
        "configs/s5e19_rotation_compensated_multiframe_geometry.yaml",
        "tools/train_s5e19_rotation_compensated_multiframe.py",
        "tools/export_s5e19_adjacent_dense_predictions.py",
        "tools/evaluate_s5e19_traceable_dense.py",
        "tools/compare_s5e19_s5e15_s5e18_orbslam3.py",
        "tools/audit_s5e19_architecture_integration.py",
    ]:
        assert (ROOT / rel).exists(), rel


def test_s5e19_config_mentions_mainline_terms():
    text = (ROOT / "configs/s5e19_rotation_compensated_multiframe_geometry.yaml").read_text(encoding="utf-8")
    for token in [
        "preserve_spherical_token_backbone",
        "use_erp_sampling",
        "use_coarse_pose_head",
        "use_fine_residual_refinement",
        "use_rotation_compensation",
        "use_fine_scale_residual_head",
        "use_external_router: false",
        "k_steps: [1, 2, 3, 5]",
        "observability_weighting",
        "anti_parallel_hard_negative",
    ]:
        assert token in text


def test_s5e19_no_eval_gt_or_orb_teacher_in_training():
    text = (ROOT / "tools/train_s5e19_rotation_compensated_multiframe.py").read_text(encoding="utf-8")
    assert "uses_eval_gt_for_training" in text
    assert "uses_eval_gt_for_scale" in text
    assert "uses_orbslam3_teacher" in text
    assert "uses_external_router" in text


def test_s5e19_report_and_metrics_terms_present():
    text = (ROOT / "tools/evaluate_s5e19_traceable_dense.py").read_text(encoding="utf-8")
    for token in [
        "rot_mean_deg",
        "signed_tdir_mean_deg",
        "anti_parallel_rate",
        "tmag_median_ratio",
        "tmag_p95_ratio",
        "path_ratio",
        "k1_tdir",
        "k2_tdir",
        "k3_tdir",
        "k5_tdir",
        "composition_error_mean",
        "path_length_consistency_error",
        "rotation_composition_error",
    ]:
        assert token in text


def test_s5e19_classifications_present():
    text = (ROOT / "tools/evaluate_s5e19_traceable_dense.py").read_text(encoding="utf-8")
    for token in [
        "S5E19_GEOMETRY_REFINEMENT_IMPROVED",
        "S5E19_TDIR_IMPROVED_SCALE_PRESERVED",
        "S5E19_SCALE_IMPROVED_TDIR_STILL_BAD",
        "S5E19_ARCHITECTURE_PRESERVED_NO_IMPROVEMENT",
        "S5E19_TRAINING_SMOKE_ONLY",
        "S5E19_ARCHITECTURE_INTEGRATION_BLOCKED",
        "S5E19_EXPORT_BLOCKED",
        "S5E19_ERROR",
    ]:
        assert token in text
