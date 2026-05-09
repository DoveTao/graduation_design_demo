# Report Index

## Purpose

This index inventories the reports visible on the current branch and labels their likely role without changing any experiment conclusion. It does not move files, delete files, or rewrite historical report bodies.

Scope note: this is a branch-local inventory. Reports that live only on other branches, such as JRT1/MF1 closeout files or ORB-branch same-evaluator reports, are intentionally not listed here unless they are present on this branch.

## Final / Thesis Reports

| path | role / experiment | status | notes |
|---|---|---|---|
| `reports/final_clean_results_table.md` | final / thesis-facing | `active_final` | likely_ref=yes |
| `reports/final_figure_index.md` | final / thesis-facing | `active_final` | likely_ref=yes |
| `reports/final_negative_results_summary.md` | final / thesis-facing | `active_final` | thesis_facing; likely_ref=yes |
| `reports/final_project_mainline_summary.md` | final / thesis-facing | `active_final` | likely_ref=yes |
| `reports/final_reproducibility_guide.md` | final / thesis-facing | `active_final` | likely_ref=yes |
| `reports/final_s1d5_english_abstract.md` | final / thesis-facing | `active_final` | likely_ref=yes |
| `reports/final_s1d5_mainline_report.md` | final / thesis-facing | `active_final` | likely_ref=yes |
| `reports/final_s1d5_results_table.csv` | final / thesis-facing | `active_final` | non_markdown_artifact; likely_ref=yes |
| `reports/final_s1d5_thesis_summary.md` | final / thesis-facing | `active_final` | likely_ref=yes |
| `reports/final_s2_fine_refinement_summary.md` | final / thesis-facing | `active_final` | likely_ref=no |
| `reports/final_s5_clean_candidate_summary.md` | final / thesis-facing | `active_final` | likely_ref=yes |
| `reports/final_s7_project_delivery_summary.md` | final / thesis-facing | `active_final` | likely_ref=no |
| `reports/final_thesis_claims_and_limitations.md` | final / thesis-facing | `active_final` | thesis_facing; likely_ref=yes |
| `reports/thesis_experiment_section_draft.md` | final / thesis-facing | `active_final` | thesis_facing; likely_ref=yes |

## Ablation and Baseline Reports

| path | role / experiment | status | notes |
|---|---|---|---|
| `reports/alignment_consistent_strong_baseline_comparison.md` | ablation / baseline summary | `keep` | diagnostic_baseline_overlap; likely_ref=yes |
| `reports/current_valid_baselines.md` | ablation / baseline summary | `keep` | partly_superseded_by_final_clean_results_table; likely_ref=yes |
| `reports/final_ablation_baseline_comparison.md` | ablation / baseline summary | `keep` | likely_ref=yes |
| `reports/thesis_three_axis_ablation.md` | ablation / baseline summary | `keep` | likely_ref=yes |

## Diagnostics and Audits

| path | role / experiment | status | notes |
|---|---|---|---|
| `reports/S5D2_dense_export_convention_audit.md` | diagnostic / audit | `diagnostic_only` | active_dense_export_audit; likely_ref=yes |
| `reports/post_s5d2_git_uncommitted_audit.md` | diagnostic / audit | `diagnostic_only` | superseded_by_REPO2_untracked_experiment_artifact_resolution; likely_ref=no |
| `reports/s5_external_export_compatibility_audit.md` | diagnostic / audit | `diagnostic_only` | older_export_compatibility_audit; likely_ref=yes |

## Negative Experiment Reports

| path | role / experiment | status | notes |
|---|---|---|---|
| `reports/S3a0_minimal_coupled_pose_head_implementation_checklist.md` | negative or diagnostic experiment line | `negative_or_diagnostic` | likely_ref=no |
| `reports/S3a_coupled_pose_head_design_plan.md` | negative or diagnostic experiment line | `negative_or_diagnostic` | likely_ref=no |
| `reports/final_post_s11_code_optimization_closure.md` | negative or diagnostic experiment line | `negative_or_diagnostic` | likely_ref=no |
| `reports/final_post_s14_code_optimization_closure.md` | negative or diagnostic experiment line | `negative_or_diagnostic` | likely_ref=no |
| `reports/final_post_s15_trajectory_training_closure.md` | negative or diagnostic experiment line | `negative_or_diagnostic` | likely_ref=no |
| `reports/final_post_s16_backbone_feasibility_closure.md` | negative or diagnostic experiment line | `negative_or_diagnostic` | likely_ref=no |
| `reports/final_post_s19_geometry_pretraining_closure.md` | negative or diagnostic experiment line | `negative_or_diagnostic` | likely_ref=no |
| `reports/final_post_s9_code_optimization_closure.md` | negative or diagnostic experiment line | `negative_or_diagnostic` | likely_ref=no |
| `reports/final_s10_chain_smoother_summary.md` | negative or diagnostic experiment line | `negative_or_diagnostic` | likely_ref=yes |
| `reports/final_s11_tmag_scale_consistency_summary.md` | negative or diagnostic experiment line | `negative_or_diagnostic` | likely_ref=yes |
| `reports/final_s12_regime_balanced_sampling_summary.md` | negative or diagnostic experiment line | `negative_or_diagnostic` | likely_ref=yes |
| `reports/final_s13_practical_usability_gap_summary.md` | negative or diagnostic experiment line | `negative_or_diagnostic` | likely_ref=yes |
| `reports/final_s14_local_window_pose_graph_summary.md` | negative or diagnostic experiment line | `negative_or_diagnostic` | likely_ref=yes |
| `reports/final_s15_trajectory_level_training_objective_summary.md` | negative or diagnostic experiment line | `negative_or_diagnostic` | likely_ref=yes |
| `reports/final_s15a_trajectory_training_smoke_failure_attribution_summary.md` | negative or diagnostic experiment line | `negative_or_diagnostic` | likely_ref=yes |
| `reports/final_s15b_pair_training_harness_stability_audit_summary.md` | negative or diagnostic experiment line | `negative_or_diagnostic` | likely_ref=yes |
| `reports/final_s15c_clean_policy_wrapped_training_harness_fix_summary.md` | negative or diagnostic experiment line | `negative_or_diagnostic` | likely_ref=yes |
| `reports/final_s15d_train_eval_forward_parity_fix_summary.md` | negative or diagnostic experiment line | `negative_or_diagnostic` | likely_ref=yes |
| `reports/final_s15e_tiny_trajectory_retest_after_harness_fix_summary.md` | negative or diagnostic experiment line | `negative_or_diagnostic` | likely_ref=yes |
| `reports/final_s16_stronger_visual_backbone_feasibility_summary.md` | negative or diagnostic experiment line | `negative_or_diagnostic` | likely_ref=yes |
| `reports/final_s16b_frozen_pretrained_backbone_probe_summary.md` | negative or diagnostic experiment line | `negative_or_diagnostic` | likely_ref=no |
| `reports/final_s17_pose_supervision_dataset_quality_summary.md` | negative or diagnostic experiment line | `negative_or_diagnostic` | likely_ref=yes |
| `reports/final_s18_split_redesign_representativeness_summary.md` | negative or diagnostic experiment line | `negative_or_diagnostic` | likely_ref=yes |
| `reports/final_s19_geometry_aware_pretraining_feasibility_summary.md` | negative or diagnostic experiment line | `negative_or_diagnostic` | likely_ref=yes |
| `reports/final_s3a_residual_head_summary.md` | negative or diagnostic experiment line | `negative_or_diagnostic` | likely_ref=no |
| `reports/final_s8_token_reliability_negative_summary.md` | negative or diagnostic experiment line | `negative_or_diagnostic` | likely_ref=no |
| `reports/final_s9_regime_only_router_summary.md` | negative or diagnostic experiment line | `negative_or_diagnostic` | likely_ref=yes |

## External Baseline / ORB-SLAM3 Reports

| path | role / experiment | status | notes |
|---|---|---|---|
| `reports/droidslam_environment_setup_report.md` | external baseline or environment | `keep` | likely_ref=no |
| `reports/external_algorithm_baseline_comparison.md` | external baseline or environment | `keep` | paired_protocol_and_results; likely_ref=yes |
| `reports/external_algorithm_baseline_protocol.md` | external baseline or environment | `keep` | paired_protocol_and_results; likely_ref=yes |
| `reports/external_baseline_environment_audit.md` | external baseline or environment | `keep` | likely_ref=no |
| `reports/orbslam3_environment_setup_report.md` | external baseline or environment | `keep` | likely_ref=no |
| `reports/pano_orb_vo_baseline_report.md` | external baseline or environment | `keep` | likely_ref=yes |
| `reports/pano_orb_vo_component_diagnostics.md` | external baseline or environment | `keep` | likely_ref=yes |

## Repository Hygiene Reports

| path | role / experiment | status | notes |
|---|---|---|---|
| `reports/REPO2_untracked_experiment_artifact_resolution.md` | repository hygiene / audit | `audit_only` | latest_repo2_resolution_audit; likely_ref=no |
| `reports/checkpoints_live_cleanup_inventory.md` | repository hygiene / audit | `audit_only` | archive_candidate; likely_ref=yes |
| `reports/checkpoints_live_cleanup_state.json` | repository hygiene / audit | `audit_only` | archive_candidate; non_markdown_artifact; likely_ref=yes |
| `reports/checkpoints_live_cleanup_summary.md` | repository hygiene / audit | `audit_only` | archive_candidate; likely_ref=yes |
| `reports/cleanup_actions_s1d5.md` | repository hygiene / audit | `audit_only` | archive_candidate; likely_ref=yes |
| `reports/cleanup_delete_candidates_s1d5.md` | repository hygiene / audit | `audit_only` | archive_candidate; likely_ref=yes |
| `reports/cleanup_inventory_s1d5.md` | repository hygiene / audit | `audit_only` | archive_candidate; likely_ref=yes |
| `reports/cleanup_summary_s1d5.md` | repository hygiene / audit | `audit_only` | archive_candidate; likely_ref=yes |
| `reports/git_branch_remote_sync_audit.md` | repository hygiene / audit | `audit_only` | likely_ref=no |
| `reports/git_commit_s1d5_summary.md` | repository hygiene / audit | `audit_only` | archive_candidate; likely_ref=yes |
| `reports/git_commit_s2b_summary.md` | repository hygiene / audit | `audit_only` | archive_candidate; likely_ref=no |
| `reports/git_staging_s1d5_precheck.md` | repository hygiene / audit | `audit_only` | archive_candidate; likely_ref=yes |
| `reports/git_staging_s1d5_summary.md` | repository hygiene / audit | `audit_only` | archive_candidate; likely_ref=yes |
| `reports/git_track_s1d5_final_files.md` | repository hygiene / audit | `audit_only` | archive_candidate; likely_ref=yes |
| `reports/gitignore_cleanup_plan.md` | repository hygiene / audit | `audit_only` | archive_candidate; likely_ref=yes |
| `reports/post_commit_workspace_cleanup_plan.md` | repository hygiene / audit | `audit_only` | archive_candidate; likely_ref=no |
| `reports/repo_remote_sync_and_hygiene_audit.md` | repository hygiene / audit | `audit_only` | likely_ref=no |

## Archive Candidates

| path | suggested future location | reason | safe_to_move_now |
|---|---|---|---|
| `reports/checkpoints_live_cleanup_inventory.md` | `reports/audits/checkpoint_cleanup/` | historical cleanup planning | `unknown` |
| `reports/checkpoints_live_cleanup_state.json` | `reports/audits/checkpoint_cleanup/` | paired state artifact for cleanup audit | `unknown` |
| `reports/checkpoints_live_cleanup_summary.md` | `reports/audits/checkpoint_cleanup/` | historical cleanup summary | `unknown` |
| `reports/cleanup_actions_s1d5.md` | `reports/audits/checkpoint_cleanup/` | transitional cleanup action log | `unknown` |
| `reports/cleanup_delete_candidates_s1d5.md` | `reports/audits/checkpoint_cleanup/` | historical delete-candidate note | `unknown` |
| `reports/cleanup_inventory_s1d5.md` | `reports/audits/checkpoint_cleanup/` | inventory-style historical note | `unknown` |
| `reports/cleanup_summary_s1d5.md` | `reports/audits/checkpoint_cleanup/` | summary of cleanup round | `unknown` |
| `reports/git_staging_s1d5_precheck.md` | `reports/audits/git_history/` | staging precheck, not thesis-facing | `unknown` |
| `reports/git_staging_s1d5_summary.md` | `reports/audits/git_history/` | staging summary, not thesis-facing | `unknown` |
| `reports/git_commit_s1d5_summary.md` | `reports/audits/git_history/` | commit bookkeeping | `unknown` |
| `reports/git_commit_s2b_summary.md` | `reports/audits/git_history/` | commit bookkeeping | `unknown` |
| `reports/post_commit_workspace_cleanup_plan.md` | `reports/audits/git_history/` | workspace cleanup planning artifact | `unknown` |
| `reports/post_s5d2_git_uncommitted_audit.md` | `reports/audits/repo_sync/` | superseded by REPO2 resolution audit | `unknown` |

## Do Not Delete

These reports are retained for reproducibility and auditability. Do not move files or delete files based on this index alone. This inventory does not move files, delete files, or invalidate any historical experiment result.
