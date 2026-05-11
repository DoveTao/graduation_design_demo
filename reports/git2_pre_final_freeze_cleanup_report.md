# GIT2 pre-final freeze cleanup report

## 执行摘要
- final_classification = `GIT2_WORKING_TREE_CLEAN_WITH_LOCAL_IGNORED_ARTIFACTS`
- branch = `experiment/s5e1-traceable-adjacent-dense`
- pushed = `True`

## 当前分支与远端
- branch: `experiment/s5e1-traceable-adjacent-dense`
- remote: `['origin\tgit@github.com:DoveTao/graduation_design_demo.git (fetch)', 'origin\tgit@github.com:DoveTao/graduation_design_demo.git (push)']`

## working tree before
{
  "clean": false,
  "modified": [
    "checkpoints/S5D11_serial_validation_and_nonselected_edge_audit.json",
    "checkpoints/S5D11_validation_concurrency_audit.json",
    "reports/s5d11_serial_validation_and_nonselected_edge_audit.md"
  ],
  "untracked": [
    "checkpoints/S5E13_correspondence_weighted_dataset_audit.json",
    "checkpoints/S5E13_real_correspondence_signed_direction_candidate.json",
    "checkpoints/S5E13_real_correspondence_signed_direction_candidate/",
    "checkpoints/S5E14_observable_edge_direction_refinement_candidate.json",
    "checkpoints/S5E14_observable_edge_direction_refinement_candidate/",
    "checkpoints/S5E15_scale_deunderfit_antiparallel_candidate.json",
    "checkpoints/S5E15_scale_deunderfit_antiparallel_candidate/",
    "checkpoints/S5E16_confidence_calibrated_sign_scale_router/router_policy.json",
    "configs/s5e13_real_correspondence_signed_direction.yaml",
    "configs/s5e14_observable_edge_direction_refinement.yaml",
    "configs/s5e15_scale_deunderfit_antiparallel.yaml",
    "external_baselines/results/s5e13_traceable_dense/",
    "external_baselines/results/s5e14_traceable_dense/",
    "external_baselines/results/s5e15_traceable_dense/",
    "external_baselines/results/s5e16_traceable_dense/router_decisions.jsonl",
    "external_baselines/results/s5e17_traceable_dense/router_decisions.jsonl",
    "external_baselines/results/s5e18_bearing_flow/spherical_bearing_flow_features.jsonl",
    "reports/s5e13_correspondence_weighted_dataset_audit.md",
    "reports/s5e13_real_correspondence_signed_direction_report.md",
    "reports/s5e13_s5e12_s5e9_s5e6_to_s5e2_orbslam3_comparison.md",
    "reports/s5e14_observable_edge_direction_refinement_report.md",
    "reports/s5e14_s5e13_s5e12_s5e9_orbslam3_comparison.md",
    "reports/s5e15_s5e14_s5e13_s5e9_orbslam3_comparison.md",
    "reports/s5e15_scale_deunderfit_antiparallel_report.md",
    "tests/test_s5e13_real_correspondence_signed_direction_static.py",
    "tests/test_s5e14_observable_edge_direction_refinement_static.py",
    "tests/test_s5e15_scale_deunderfit_antiparallel_static.py",
    "tools/audit_s5e14_s5e13_direction_errors.py",
    "tools/audit_s5e15_under_scale_and_antiparallel.py",
    "tools/build_s5e13_correspondence_weighted_dataset.py",
    "tools/compare_s5e13_s5e12_s5e9_s5e6_to_s5e2_orbslam3.py",
    "tools/compare_s5e14_s5e13_s5e12_s5e9_orbslam3.py",
    "tools/compare_s5e15_s5e14_s5e13_s5e9_orbslam3.py",
    "tools/evaluate_s5e13_traceable_dense.py",
    "tools/evaluate_s5e14_traceable_dense.py",
    "tools/evaluate_s5e15_traceable_dense.py",
    "tools/export_s5e13_adjacent_dense_predictions.py",
    "tools/export_s5e14_adjacent_dense_predictions.py",
    "tools/export_s5e15_adjacent_dense_predictions.py",
    "tools/train_s5e13_real_correspondence_direction_candidate.py",
    "tools/train_s5e14_observable_direction_refinement.py",
    "tools/train_s5e15_scale_antiparallel_refinement.py"
  ],
  "deleted": [],
  "ignored_summary_count": 113
}

## 大文件审计
- `./checkpoints/T57b_no_dt_multiscale_tmag_head_400/final.pt`
- `./logs/s5d12_dense_fill_code_search.log`
- `./logs/s5d6_final_s5_pairwise_source_search.log`

## 文件分类表
- should_commit: 36
- should_ignore_or_local_only: 6
- should_restore: 3
- needs_manual_review: 0

## S5E13-S5E15 历史残留处理
{
  "s5e13_commit_candidates": [
    "checkpoints/S5E13_correspondence_weighted_dataset_audit.json",
    "checkpoints/S5E13_real_correspondence_signed_direction_candidate.json",
    "checkpoints/S5E13_real_correspondence_signed_direction_candidate/",
    "configs/s5e13_real_correspondence_signed_direction.yaml",
    "reports/s5e13_correspondence_weighted_dataset_audit.md",
    "reports/s5e13_real_correspondence_signed_direction_report.md",
    "reports/s5e13_s5e12_s5e9_s5e6_to_s5e2_orbslam3_comparison.md"
  ],
  "s5e14_commit_candidates": [
    "checkpoints/S5E14_observable_edge_direction_refinement_candidate.json",
    "checkpoints/S5E14_observable_edge_direction_refinement_candidate/",
    "configs/s5e14_observable_edge_direction_refinement.yaml",
    "reports/s5e14_observable_edge_direction_refinement_report.md",
    "reports/s5e14_s5e13_s5e12_s5e9_orbslam3_comparison.md"
  ],
  "s5e15_commit_candidates": [
    "checkpoints/S5E15_scale_deunderfit_antiparallel_candidate.json",
    "checkpoints/S5E15_scale_deunderfit_antiparallel_candidate/",
    "configs/s5e15_scale_deunderfit_antiparallel.yaml",
    "reports/s5e15_s5e14_s5e13_s5e9_orbslam3_comparison.md",
    "reports/s5e15_scale_deunderfit_antiparallel_report.md"
  ]
}

## S5D11 validation refresh 文件处理
{
  "restored": [
    "checkpoints/S5D11_validation_concurrency_audit.json",
    "checkpoints/S5D11_serial_validation_and_nonselected_edge_audit.json",
    "reports/s5d11_serial_validation_and_nonselected_edge_audit.md"
  ],
  "logs_local_only": []
}

## .gitignore 更新
- `external_baselines/results/**/router_decisions.jsonl`
- `external_baselines/results/**/spherical_bearing_flow_features.jsonl`
- `external_baselines/results/**/correspondence_loss_weights.jsonl`
- `external_baselines/results/**/correspondence_refinement_weights.jsonl`
- `external_baselines/results/**/scale_antiparallel_refinement_weights.jsonl`
- `checkpoints/**/correspondence_weighted_dataset.json`

## commits created
- `1f54a65` `(HEAD -> experiment/s5e1-traceable-adjacent-dense) GIT2: add cleanup report and checkpoint`
- `652964f` `GIT2: update gitignore for final local artifacts`
- `58269a2` `GIT2: add pre-final freeze cleanup audit`
- `08fcac7` `S5E13-S5E15: add missing lightweight experiment summaries`

## push status
- pushed = `True`

## working tree after
{
  "clean": true,
  "remaining_files": []
}

## 剩余 local-only / ignored artifacts
- none

## caveats
- 不重新训练。
- 不刷新实验指标。
- 不生成新 trajectory。
- S5 locked metrics/policy unchanged。
