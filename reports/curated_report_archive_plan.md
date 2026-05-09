# Curated Report Archive Plan

## Scope

planning only. no files moved, no files deleted, no branches merged, and no cherry-picks were performed.

## Inputs

- `reports/all_branch_inventory_and_classification.md`
- `checkpoints/BRANCH1_all_branch_inventory_and_classification.json`
- `reports/REPORT_INDEX.md`
- `reports/reports_inventory_audit.md`
- `checkpoints/REPORT1_reports_inventory_audit.json`
- `reports/multibranch_reports_inventory.md`
- `checkpoints/REPORT2_multibranch_reports_inventory.json`

## Proposed Curated Archive Structure

```text
reports_curated/
  README.md
  final/
    final_project_mainline_summary.md
    final_clean_results_table.md
    final_reproducibility_guide.md
    final_thesis_claims_and_limitations.md
    thesis_experiment_section_draft.md
  ablations/
    final_ablation_baseline_comparison.md
    thesis_three_axis_ablation.md
  baselines/
    pano_orb_vo_baseline_report.md
    pano_orb_vo_component_diagnostics.md
    external_algorithm_baseline_comparison.md
    orbslam3_fisheye_evaluation_report.md
    s5_orbslam3_same_evaluator_comparison.md
  diagnostics/
    S5D2_dense_export_convention_audit.md
    s5_dense_external_export_report.md
    s5_external_export_compatibility_audit.md
  negative_experiments/
    jrt1/
      JRT1_final_closeout_summary.md
      JRT1b_train_cv_report.md
      JRT1c_train_cv_trajectory_proxy_report.md
    mf1/
      MF1_final_closeout_summary.md
      MF1a_chain_refiner_smoke_report.md
      MF1b_train_cv_report.md
  audits/
    repo_remote_sync_and_hygiene_audit.md
    REPO2_untracked_experiment_artifact_resolution.md
    all_branch_inventory_and_classification.md
    multibranch_reports_inventory.md
  environment/
    external_baseline_environment_audit.md
    orbslam3_environment_setup_report.md
    droidslam_environment_setup_report.md
```

## Canonical Report Selection

| target | source branch | source path | source commit | reason | status |
|---|---|---|---|---|---|
| `reports_curated/ablations/final_ablation_baseline_comparison.md` | `polish/final-reproducibility-guardrails` | `reports/final_ablation_baseline_comparison.md` | `7cb0a2c` | final ablation summary should stay with final baseline line | `ready_to_cherry_pick_later` |
| `reports_curated/ablations/thesis_three_axis_ablation.md` | `polish/final-reproducibility-guardrails` | `reports/thesis_three_axis_ablation.md` | `7cb0a2c` | three-axis thesis ablation companion report | `ready_to_cherry_pick_later` |
| `reports_curated/audits/REPO2_untracked_experiment_artifact_resolution.md` | `experiment/s5d2-dense-export-convention-audit` | `reports/REPO2_untracked_experiment_artifact_resolution.md` | `9fcdd5d` | tracks branch-specific artifact resolution history | `ready_to_cherry_pick_later` |
| `reports_curated/audits/all_branch_inventory_and_classification.md` | `experiment/s5d2-dense-export-convention-audit` | `reports/all_branch_inventory_and_classification.md` | `9fcdd5d` | all-branch inventory that feeds later archive curation | `ready_to_cherry_pick_later` |
| `reports_curated/audits/multibranch_reports_inventory.md` | `experiment/s5d2-dense-export-convention-audit` | `reports/multibranch_reports_inventory.md` | `9fcdd5d` | cross-branch matrix for report canonicalization | `ready_to_cherry_pick_later` |
| `reports_curated/audits/repo_remote_sync_and_hygiene_audit.md` | `polish/final-reproducibility-guardrails` | `reports/repo_remote_sync_and_hygiene_audit.md` | `7cb0a2c` | pre-JRT hygiene audit with reproducibility context | `ready_to_cherry_pick_later` |
| `reports_curated/baselines/external_algorithm_baseline_comparison.md` | `polish/final-reproducibility-guardrails` | `reports/external_algorithm_baseline_comparison.md` | `7cb0a2c` | thesis-facing baseline comparison on the final/polish line | `ready_to_cherry_pick_later` |
| `reports_curated/baselines/orbslam3_fisheye_evaluation_report.md` | `experiment/orbslam3-fisheye-strong-baseline` | `reports/orbslam3_fisheye_evaluation_report.md` | `797865b` | ORB-specific artifacts and reports live on the dedicated external baseline branch | `ready_to_cherry_pick_later` |
| `reports_curated/baselines/pano_orb_vo_baseline_report.md` | `polish/final-reproducibility-guardrails` | `reports/pano_orb_vo_baseline_report.md` | `7cb0a2c` | final baseline branch keeps the stable panorama baseline story | `ready_to_cherry_pick_later` |
| `reports_curated/baselines/pano_orb_vo_component_diagnostics.md` | `polish/final-reproducibility-guardrails` | `reports/pano_orb_vo_component_diagnostics.md` | `7cb0a2c` | component-level companion for the panorama baseline | `ready_to_cherry_pick_later` |
| `reports_curated/baselines/s5_orbslam3_same_evaluator_comparison.md` | `experiment/orbslam3-fisheye-strong-baseline` | `reports/s5_orbslam3_same_evaluator_comparison.md` | `797865b` | paired comparison with ORB-SLAM3 belongs to the ORB branch | `ready_to_cherry_pick_later` |
| `reports_curated/diagnostics/S5D2_dense_export_convention_audit.md` | `experiment/s5d2-dense-export-convention-audit` | `reports/S5D2_dense_export_convention_audit.md` | `9fcdd5d` | latest branch-specific audit of dense export scope and convention | `ready_to_cherry_pick_later` |
| `reports_curated/diagnostics/s5_dense_external_export_report.md` | `experiment/orbslam3-fisheye-strong-baseline` | `reports/s5_dense_external_export_report.md` | `797865b` | paired dense export diagnostic used in same-evaluator comparison line | `ready_to_cherry_pick_later` |
| `reports_curated/diagnostics/s5_external_export_compatibility_audit.md` | `polish/final-reproducibility-guardrails` | `reports/s5_external_export_compatibility_audit.md` | `7cb0a2c` | older compatibility audit retained as supporting diagnostic history | `ready_to_cherry_pick_later` |
| `reports_curated/environment/droidslam_environment_setup_report.md` | `polish/final-reproducibility-guardrails` | `reports/droidslam_environment_setup_report.md` | `7cb0a2c` | DROID-SLAM setup feasibility context | `ready_to_cherry_pick_later` |
| `reports_curated/environment/external_baseline_environment_audit.md` | `polish/final-reproducibility-guardrails` | `reports/external_baseline_environment_audit.md` | `7cb0a2c` | baseline environment feasibility audit | `ready_to_cherry_pick_later` |
| `reports_curated/environment/orbslam3_environment_setup_report.md` | `polish/final-reproducibility-guardrails` | `reports/orbslam3_environment_setup_report.md` | `7cb0a2c` | ORB-SLAM3 setup feasibility context | `ready_to_cherry_pick_later` |
| `reports_curated/final/final_clean_results_table.md` | `polish/final-reproducibility-guardrails` | `reports/final_clean_results_table.md` | `7cb0a2c` | canonical locked final results table | `ready_to_cherry_pick_later` |
| `reports_curated/final/final_project_mainline_summary.md` | `polish/final-reproducibility-guardrails` | `reports/final_project_mainline_summary.md` | `7cb0a2c` | final mainline overview for thesis-facing archive | `ready_to_cherry_pick_later` |
| `reports_curated/final/final_reproducibility_guide.md` | `polish/final-reproducibility-guardrails` | `reports/final_reproducibility_guide.md` | `7cb0a2c` | authoritative final reproducibility branch | `ready_to_cherry_pick_later` |
| `reports_curated/final/final_thesis_claims_and_limitations.md` | `polish/final-reproducibility-guardrails` | `reports/final_thesis_claims_and_limitations.md` | `7cb0a2c` | canonical claims/limitations summary | `ready_to_cherry_pick_later` |
| `reports_curated/final/thesis_experiment_section_draft.md` | `polish/final-reproducibility-guardrails` | `reports/thesis_experiment_section_draft.md` | `7cb0a2c` | thesis draft that ties experiments together | `ready_to_cherry_pick_later` |
| `reports_curated/negative_experiments/jrt1/JRT1_final_closeout_summary.md` | `experiment/jrt1-joint-rtdir-coupled-refiner` | `reports/JRT1_final_closeout_summary.md` | `6765af3` | negative experiment conclusion belongs to its own branch | `ready_to_cherry_pick_later` |
| `reports_curated/negative_experiments/jrt1/JRT1b_train_cv_report.md` | `experiment/jrt1-joint-rtdir-coupled-refiner` | `reports/JRT1b_train_cv_report.md` | `6765af3` | stage report retained as supporting evidence under the JRT1 closeout | `ready_to_cherry_pick_later` |
| `reports_curated/negative_experiments/jrt1/JRT1c_train_cv_trajectory_proxy_report.md` | `experiment/jrt1-joint-rtdir-coupled-refiner` | `reports/JRT1c_train_cv_trajectory_proxy_report.md` | `6765af3` | trajectory-proxy stage report retained as supporting evidence under the JRT1 closeout | `ready_to_cherry_pick_later` |
| `reports_curated/negative_experiments/mf1/MF1_final_closeout_summary.md` | `experiment/mf1-multi-frame-chain-refiner` | `reports/MF1_final_closeout_summary.md` | `ea559d5` | negative experiment conclusion belongs to its own branch | `ready_to_cherry_pick_later` |
| `reports_curated/negative_experiments/mf1/MF1a_chain_refiner_smoke_report.md` | `experiment/mf1-multi-frame-chain-refiner` | `reports/MF1a_chain_refiner_smoke_report.md` | `ea559d5` | smoke harness evidence for MF1 line | `ready_to_cherry_pick_later` |
| `reports_curated/negative_experiments/mf1/MF1b_train_cv_report.md` | `experiment/mf1-multi-frame-chain-refiner` | `reports/MF1b_train_cv_report.md` | `ea559d5` | train-CV evidence for MF1 line | `ready_to_cherry_pick_later` |
| `reports_curated/negative_experiments/optimize/final_s10_chain_smoother_summary.md` | `optimize/s10-chain-pathratio-smoother` | `reports/final_s10_chain_smoother_summary.md` | `0ae9aee` | represents optimize S10 negative summary without bringing full branch payload | `needs_user_review` |
| `reports_curated/negative_experiments/optimize/final_s11_tmag_scale_consistency_summary.md` | `optimize/s11-tmag-scale-consistency-training` | `reports/final_s11_tmag_scale_consistency_summary.md` | `b00eab4` | represents optimize S11 negative summary without bringing full branch payload | `needs_user_review` |
| `reports_curated/negative_experiments/optimize/final_s12_regime_balanced_sampling_summary.md` | `optimize/s12-regime-balanced-sampling` | `reports/final_s12_regime_balanced_sampling_summary.md` | `34edd74` | represents optimize S12 negative summary without bringing full branch payload | `needs_user_review` |
| `reports_curated/negative_experiments/optimize/final_s14_local_window_pose_graph_summary.md` | `optimize/s14-local-window-pose-graph` | `reports/final_s14_local_window_pose_graph_summary.md` | `36e68fe` | represents optimize S14 negative summary without bringing full branch payload | `needs_user_review` |
| `reports_curated/negative_experiments/optimize/final_s15_trajectory_level_training_objective_summary.md` | `optimize/s15-trajectory-level-training-objective` | `reports/final_s15_trajectory_level_training_objective_summary.md` | `8a28577` | represents optimize S15 negative summary without bringing full branch payload | `needs_user_review` |
| `reports_curated/negative_experiments/optimize/final_s16_stronger_visual_backbone_feasibility_summary.md` | `optimize/s16-stronger-visual-backbone-feasibility` | `reports/final_s16_stronger_visual_backbone_feasibility_summary.md` | `8f52076` | represents optimize S16 negative summary without bringing full branch payload | `needs_user_review` |
| `reports_curated/negative_experiments/optimize/final_s17_pose_supervision_dataset_quality_summary.md` | `optimize/s17-pose-supervision-dataset-quality-audit` | `reports/final_s17_pose_supervision_dataset_quality_summary.md` | `e020e11` | represents optimize S17 negative summary without bringing full branch payload | `needs_user_review` |
| `reports_curated/negative_experiments/optimize/final_s19_geometry_aware_pretraining_feasibility_summary.md` | `optimize/s19-geometry-aware-pretraining-feasibility` | `reports/final_s19_geometry_aware_pretraining_feasibility_summary.md` | `f90d167` | represents optimize S19 negative summary without bringing full branch payload | `needs_user_review` |

## Negative Experiment Selection

### JRT1

- canonical closeout: `reports_curated/negative_experiments/jrt1/JRT1_final_closeout_summary.md`
- supporting reports: `reports_curated/negative_experiments/jrt1/JRT1b_train_cv_report.md, reports_curated/negative_experiments/jrt1/JRT1c_train_cv_trajectory_proxy_report.md`
- policy: keep stage reports as supporting evidence, but treat closeout as canonical summary

### MF1

- canonical closeout: `reports_curated/negative_experiments/mf1/MF1_final_closeout_summary.md`
- supporting reports: `reports_curated/negative_experiments/mf1/MF1a_chain_refiner_smoke_report.md, reports_curated/negative_experiments/mf1/MF1b_train_cv_report.md`
- policy: keep stage reports as supporting evidence, but treat closeout as canonical summary

## External Baseline Selection

- Pano-ORB-VO: `reports_curated/baselines/pano_orb_vo_baseline_report.md`, `reports_curated/baselines/pano_orb_vo_component_diagnostics.md`
- ORB-SLAM3: `reports_curated/baselines/orbslam3_fisheye_evaluation_report.md`
- S5 dense same-evaluator comparison: `reports_curated/baselines/s5_orbslam3_same_evaluator_comparison.md`

## Diagnostics Selection

- `reports_curated/diagnostics/S5D2_dense_export_convention_audit.md`
- `reports_curated/diagnostics/s5_dense_external_export_report.md`
- `reports_curated/diagnostics/s5_external_export_compatibility_audit.md`

## Duplicate and Divergent Report Policy

| topic / issue | handling | recommendation |
|---|---|---|
| divergent same-path reports (`6`) | keep branch-local | do not merge by filename alone; select canonical branch per topic |
| duplicate / overlapping topic groups (`14`) | choose canonical report later | use REPORT2 canonical mapping as the starting point |
| JRT1 / MF1 stage vs closeout | closeout supersedes as summary | keep stage reports as evidence only |
| S5 export diagnostics chain | S5D2 is current canonical audit | retain older export audits as supporting history |
| optimize/* final summaries | manual review needed | include only summary/negative-closure style files, not whole branches |

## Branch Handling Policy

- Do not merge experiment branches back into polish/final-reproducibility-guardrails.
- Keep JRT1 and MF1 as preserved negative-experiment branches.
- Keep experiment/orbslam3-fisheye-strong-baseline as the dedicated external baseline branch.
- Keep experiment/s5d2-dense-export-convention-audit as the dedicated diagnostic and reporting branch.
- If a unified thesis/report branch is needed later, cherry-pick curated reports and supporting checkpoints only, never an entire experiment branch.

## Recommended Next Step

If user approves: `REPORT4_curated_report_archive_materialization`.
Else: keep current branch state and use this plan as a no-op curation blueprint.

## No-Op Confirmation

- no files moved = true
- no files deleted = true
- no branches merged = true
- no cherry-picks = true

