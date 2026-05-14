from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_s5e13_required_files_exist():
    paths = [
        "configs/s5e13_real_correspondence_signed_direction.yaml",
        "tools/build_s5e13_correspondence_weighted_dataset.py",
        "tools/train_s5e13_real_correspondence_direction_candidate.py",
        "tools/export_s5e13_adjacent_dense_predictions.py",
        "tools/evaluate_s5e13_traceable_dense.py",
        "tools/compare_s5e13_s5e12_s5e9_s5e6_to_s5e2_orbslam3.py",
    ]
    for rel in paths:
        assert (ROOT / rel).exists(), rel


def test_s5e13_config_mentions_weighting_and_limits():
    text = (ROOT / "configs/s5e13_real_correspondence_signed_direction.yaml").read_text(encoding="utf-8")
    assert "signed_reliable" in text
    assert "signed_small_motion_multiplier" in text
    assert "strict essential geometry" in text.lower() or "strict_essential_geometry" in text
    assert "scene01/seq03 GT" in text


def test_s5e13_script_mentions_classifications_and_metrics():
    text = (ROOT / "tools/evaluate_s5e13_traceable_dense.py").read_text(encoding="utf-8")
    for token in [
        "S5E13_CORRESPONDENCE_TDIR_IMPROVED",
        "S5E13_OBSERVABLE_TDIR_IMPROVED_OVERALL_STILL_BAD",
        "S5E13_SCALE_PRESERVED_TDIR_NO_IMPROVEMENT",
        "anti_parallel_rate",
        "tmag_p95_ratio",
        "path_ratio",
        "observable_edges",
        "unobservable_edges",
        "small_motion_edges",
        "reliable_signed_direction_edges",
    ]:
        assert token in text


def test_s5e13_report_and_checkpoint_paths_referenced():
    cfg = (ROOT / "configs/s5e13_real_correspondence_signed_direction.yaml").read_text(encoding="utf-8")
    assert "reports/s5e13_real_correspondence_signed_direction_report.md" in cfg
    assert "reports/s5e13_s5e12_s5e9_s5e6_to_s5e2_orbslam3_comparison.md" in cfg
    assert "checkpoints/S5E13_real_correspondence_signed_direction_candidate.json" in cfg


def test_s5e13_no_gt_leakage_and_locked_policy_caveats_present():
    text = (ROOT / "tools/build_s5e13_correspondence_weighted_dataset.py").read_text(encoding="utf-8")
    assert "uses_scene01_seq03_gt_for_training" in text
    report_text = (ROOT / "tools/evaluate_s5e13_traceable_dense.py").read_text(encoding="utf-8")
    assert "official S5 locked result" in report_text or "official_s5_unchanged" in report_text
