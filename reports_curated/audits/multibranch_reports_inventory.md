# Multibranch Reports Inventory

## Scope

This is a 16-branch read-only inventory. no files moved, no files deleted, no branches merged, and no cherry-picks were performed.

## Branches Audited

| branch | group | latest commit | report_count | checkpoint_count | recommendation |
|---|---|---|---:|---:|---|
| `experiment/jrt1-joint-rtdir-coupled-refiner` | `archived_negative_experiments` | `6765af3 Add JRT1 branch remote sync audit` | `74` | `157` | `historical_archive` |
| `experiment/mf1-multi-frame-chain-refiner` | `archived_negative_experiments` | `ea559d5 Add MF1 multi-frame chain refiner artifacts` | `77` | `162` | `historical_archive` |
| `experiment/orbslam3-fisheye-strong-baseline` | `external_baseline_experiments` | `797865b Add S5 dense same-evaluator comparison artifacts` | `77` | `159` | `keep_separate` |
| `experiment/s5d2-dense-export-convention-audit` | `diagnostics` | `9574f51 Add all-branch inventory and classification` | `75` | `154` | `keep_separate` |
| `main` | `final_or_polish` | `bb8fadd Add S1d5 cleanup and staging audit reports` | `18` | `11` | `no_action` |
| `optimize/s10-chain-pathratio-smoother` | `historical_optimize_experiments` | `0ae9aee Add S10 chain-level path-ratio preserving smoother diagnostic` | `35` | `94` | `historical_archive` |
| `optimize/s11-tmag-scale-consistency-training` | `historical_optimize_experiments` | `b00eab4 Close code optimization after S10 S11 diagnostics` | `37` | `96` | `historical_archive` |
| `optimize/s12-regime-balanced-sampling` | `historical_optimize_experiments` | `34edd74 Add S13 practical usability gap analysis` | `39` | `111` | `historical_archive` |
| `optimize/s14-local-window-pose-graph` | `historical_optimize_experiments` | `36e68fe Close code optimization after S14 pose graph diagnostic` | `41` | `114` | `historical_archive` |
| `optimize/s15-trajectory-level-training-objective` | `historical_optimize_experiments` | `8a28577 Close S15 trajectory training line after stability retest` | `48` | `122` | `historical_archive` |
| `optimize/s16-stronger-visual-backbone-feasibility` | `historical_optimize_experiments` | `8f52076 Close S16 backbone feasibility after ResNet50 probe` | `51` | `126` | `historical_archive` |
| `optimize/s17-pose-supervision-dataset-quality-audit` | `historical_optimize_experiments` | `e020e11 Add S18 split representativeness evaluation` | `53` | `137` | `historical_archive` |
| `optimize/s19-geometry-aware-pretraining-feasibility` | `historical_optimize_experiments` | `f90d167 Add git branch remote sync audit` | `56` | `143` | `historical_archive` |
| `optimize/s2-fine-refinement-on-s1d5` | `historical_optimize_experiments` | `c1f4b36 Add S3 coupled pose head design plan` | `22` | `44` | `historical_archive` |
| `optimize/s3a0-coupled-pose-residual-head` | `historical_optimize_experiments` | `f232fa3 Close post-S5 optimization after S8 S9 diagnostics` | `34` | `89` | `historical_archive` |
| `polish/final-reproducibility-guardrails` | `final_or_polish` | `7cb0a2c Add repository hygiene audit before JRT1` | `69` | `151` | `canonical_for_topic` |

## Branch Summaries

### experiment/jrt1-joint-rtdir-coupled-refiner

- group: `archived_negative_experiments`
- role: negative experiment branch with closeout reports
- status/classification: `NO_STABLE_JRT1_TRAJECTORY_GAIN`
- report_count/checkpoint_count/script_count/test_count: `74/157/56/15`
- recommendation: `historical_archive`
- key reports:
  - `reports/JRT1_final_closeout_summary.md`
  - `reports/final_ablation_baseline_comparison.md`
  - `reports/final_clean_results_table.md`
  - `reports/final_figure_index.md`
  - `reports/final_negative_results_summary.md`
  - `reports/final_post_s11_code_optimization_closure.md`
- key checkpoints:
  - `checkpoints/AB1_final_ablation_baseline_comparison_results.json`
  - `checkpoints/AB2_thesis_three_axis_ablation_results.json`
  - `checkpoints/EXT1_external_algorithm_baseline_results.json`
  - `checkpoints/JRT1a_framework_smoke_results.json`
  - `checkpoints/JRT1b_train_cv_results.json`
  - `checkpoints/JRT1c_train_cv_trajectory_proxy_results.json`

### experiment/mf1-multi-frame-chain-refiner

- group: `archived_negative_experiments`
- role: negative experiment branch with closeout reports
- status/classification: `NO_STABLE_MF1_CHAIN_GAIN`
- report_count/checkpoint_count/script_count/test_count: `77/162/59/18`
- recommendation: `historical_archive`
- key reports:
  - `reports/JRT1_final_closeout_summary.md`
  - `reports/MF1_final_closeout_summary.md`
  - `reports/final_ablation_baseline_comparison.md`
  - `reports/final_clean_results_table.md`
  - `reports/final_figure_index.md`
  - `reports/final_negative_results_summary.md`
- key checkpoints:
  - `checkpoints/AB1_final_ablation_baseline_comparison_results.json`
  - `checkpoints/AB2_thesis_three_axis_ablation_results.json`
  - `checkpoints/EXT1_external_algorithm_baseline_results.json`
  - `checkpoints/JRT1a_framework_smoke_results.json`
  - `checkpoints/JRT1b_train_cv_results.json`
  - `checkpoints/JRT1c_train_cv_trajectory_proxy_results.json`

### experiment/orbslam3-fisheye-strong-baseline

- group: `external_baseline_experiments`
- role: external baseline and same-evaluator comparison branch
- status/classification: `unknown_or_not_applicable`
- report_count/checkpoint_count/script_count/test_count: `77/159/54/18`
- recommendation: `keep_separate`
- key reports:
  - `reports/final_ablation_baseline_comparison.md`
  - `reports/final_clean_results_table.md`
  - `reports/final_figure_index.md`
  - `reports/final_negative_results_summary.md`
  - `reports/final_post_s11_code_optimization_closure.md`
  - `reports/final_post_s14_code_optimization_closure.md`
- key checkpoints:
  - `checkpoints/AB1_final_ablation_baseline_comparison_results.json`
  - `checkpoints/AB2_thesis_three_axis_ablation_results.json`
  - `checkpoints/EXT1_external_algorithm_baseline_results.json`
  - `checkpoints/ORB1d_orbslam3_fisheye_evaluation_results.json`
  - `checkpoints/S5D_same_evaluator_comparison_results.json`
  - `checkpoints/SB1_pano_orb_vo_baseline_results.json`

### experiment/s5d2-dense-export-convention-audit

- group: `diagnostics`
- role: dense export diagnostic and inventory branch
- status/classification: `S5D2-EVAL-SCOPE-MISMATCH`
- report_count/checkpoint_count/script_count/test_count: `75/154/53/14`
- recommendation: `keep_separate`
- key reports:
  - `reports/final_ablation_baseline_comparison.md`
  - `reports/final_clean_results_table.md`
  - `reports/final_figure_index.md`
  - `reports/final_negative_results_summary.md`
  - `reports/final_post_s11_code_optimization_closure.md`
  - `reports/final_post_s14_code_optimization_closure.md`
- key checkpoints:
  - `checkpoints/AB1_final_ablation_baseline_comparison_results.json`
  - `checkpoints/AB2_thesis_three_axis_ablation_results.json`
  - `checkpoints/EXT1_external_algorithm_baseline_results.json`
  - `checkpoints/SB1_pano_orb_vo_baseline_results.json`
  - `checkpoints/SB2_alignment_consistent_baseline_results.json`
  - `checkpoints/S1d5_final_figures/scale_repair_summary.png`

### main

- group: `final_or_polish`
- role: reference branch
- status/classification: `historical main branch reference`
- report_count/checkpoint_count/script_count/test_count: `18/11/22/0`
- recommendation: `no_action`
- key reports:
  - `reports/final_s1d5_english_abstract.md`
  - `reports/final_s1d5_mainline_report.md`
  - `reports/final_s1d5_results_table.csv`
  - `reports/final_s1d5_thesis_summary.md`
  - `reports/git_track_s1d5_final_files.md`
  - `reports/checkpoints_live_cleanup_summary.md`
- key checkpoints:
  - `checkpoints/S1d5_final_figures/scale_repair_summary.png`
  - `checkpoints/S1d5_final_mainline_summary.md`
  - `checkpoints/S1d5_final_repro/s1d5_policy_eval_summary.json`
  - `checkpoints/S1d5_clean_dt_anchor_policy.json`
  - `checkpoints/S1d5_freeze_clean_dt_anchor_policy_export_report.md`
  - `checkpoints/S1d5_final_baseline_comparison.md`

### optimize/s10-chain-pathratio-smoother

- group: `historical_optimize_experiments`
- role: historical optimization experiment branch
- status/classification: `unknown_or_not_applicable`
- report_count/checkpoint_count/script_count/test_count: `35/94/33/0`
- recommendation: `historical_archive`
- key reports:
  - `reports/final_clean_results_table.md`
  - `reports/final_figure_index.md`
  - `reports/final_negative_results_summary.md`
  - `reports/final_post_s9_code_optimization_closure.md`
  - `reports/final_project_mainline_summary.md`
  - `reports/final_reproducibility_guide.md`
- key checkpoints:
  - `checkpoints/S1d5_final_figures/scale_repair_summary.png`
  - `checkpoints/S1d5_final_mainline_summary.md`
  - `checkpoints/S1d5_final_repro/s1d5_policy_eval_summary.json`
  - `checkpoints/S2_final_fine_refinement_summary.md`
  - `checkpoints/S2a_fine_rot_policy_audit_on_s1d5/rot_0p2/s1d5_policy_eval_summary.json`
  - `checkpoints/S2a_fine_rot_policy_audit_on_s1d5/rot_0p25/s1d5_policy_eval_summary.json`

### optimize/s11-tmag-scale-consistency-training

- group: `historical_optimize_experiments`
- role: historical optimization experiment branch
- status/classification: `unknown_or_not_applicable`
- report_count/checkpoint_count/script_count/test_count: `37/96/35/0`
- recommendation: `historical_archive`
- key reports:
  - `reports/final_clean_results_table.md`
  - `reports/final_figure_index.md`
  - `reports/final_negative_results_summary.md`
  - `reports/final_post_s11_code_optimization_closure.md`
  - `reports/final_post_s9_code_optimization_closure.md`
  - `reports/final_project_mainline_summary.md`
- key checkpoints:
  - `checkpoints/S1d5_final_figures/scale_repair_summary.png`
  - `checkpoints/S1d5_final_mainline_summary.md`
  - `checkpoints/S1d5_final_repro/s1d5_policy_eval_summary.json`
  - `checkpoints/S2_final_fine_refinement_summary.md`
  - `checkpoints/S2a_fine_rot_policy_audit_on_s1d5/rot_0p2/s1d5_policy_eval_summary.json`
  - `checkpoints/S2a_fine_rot_policy_audit_on_s1d5/rot_0p25/s1d5_policy_eval_summary.json`

### optimize/s12-regime-balanced-sampling

- group: `historical_optimize_experiments`
- role: historical optimization experiment branch
- status/classification: `unknown_or_not_applicable`
- report_count/checkpoint_count/script_count/test_count: `39/111/37/0`
- recommendation: `historical_archive`
- key reports:
  - `reports/final_clean_results_table.md`
  - `reports/final_figure_index.md`
  - `reports/final_negative_results_summary.md`
  - `reports/final_post_s11_code_optimization_closure.md`
  - `reports/final_post_s9_code_optimization_closure.md`
  - `reports/final_project_mainline_summary.md`
- key checkpoints:
  - `checkpoints/S1d5_final_figures/scale_repair_summary.png`
  - `checkpoints/S1d5_final_mainline_summary.md`
  - `checkpoints/S1d5_final_repro/s1d5_policy_eval_summary.json`
  - `checkpoints/S2_final_fine_refinement_summary.md`
  - `checkpoints/S2a_fine_rot_policy_audit_on_s1d5/rot_0p2/s1d5_policy_eval_summary.json`
  - `checkpoints/S2a_fine_rot_policy_audit_on_s1d5/rot_0p25/s1d5_policy_eval_summary.json`

### optimize/s14-local-window-pose-graph

- group: `historical_optimize_experiments`
- role: historical optimization experiment branch
- status/classification: `unknown_or_not_applicable`
- report_count/checkpoint_count/script_count/test_count: `41/114/38/0`
- recommendation: `historical_archive`
- key reports:
  - `reports/final_clean_results_table.md`
  - `reports/final_figure_index.md`
  - `reports/final_negative_results_summary.md`
  - `reports/final_post_s11_code_optimization_closure.md`
  - `reports/final_post_s14_code_optimization_closure.md`
  - `reports/final_post_s9_code_optimization_closure.md`
- key checkpoints:
  - `checkpoints/S1d5_final_figures/scale_repair_summary.png`
  - `checkpoints/S1d5_final_mainline_summary.md`
  - `checkpoints/S1d5_final_repro/s1d5_policy_eval_summary.json`
  - `checkpoints/S2_final_fine_refinement_summary.md`
  - `checkpoints/S2a_fine_rot_policy_audit_on_s1d5/rot_0p2/s1d5_policy_eval_summary.json`
  - `checkpoints/S2a_fine_rot_policy_audit_on_s1d5/rot_0p25/s1d5_policy_eval_summary.json`

### optimize/s15-trajectory-level-training-objective

- group: `historical_optimize_experiments`
- role: historical optimization experiment branch
- status/classification: `unknown_or_not_applicable`
- report_count/checkpoint_count/script_count/test_count: `48/122/40/0`
- recommendation: `historical_archive`
- key reports:
  - `reports/final_clean_results_table.md`
  - `reports/final_figure_index.md`
  - `reports/final_negative_results_summary.md`
  - `reports/final_post_s11_code_optimization_closure.md`
  - `reports/final_post_s14_code_optimization_closure.md`
  - `reports/final_post_s15_trajectory_training_closure.md`
- key checkpoints:
  - `checkpoints/S1d5_final_figures/scale_repair_summary.png`
  - `checkpoints/S1d5_final_mainline_summary.md`
  - `checkpoints/S1d5_final_repro/s1d5_policy_eval_summary.json`
  - `checkpoints/S2_final_fine_refinement_summary.md`
  - `checkpoints/S2a_fine_rot_policy_audit_on_s1d5/rot_0p2/s1d5_policy_eval_summary.json`
  - `checkpoints/S2a_fine_rot_policy_audit_on_s1d5/rot_0p25/s1d5_policy_eval_summary.json`

### optimize/s16-stronger-visual-backbone-feasibility

- group: `historical_optimize_experiments`
- role: historical optimization experiment branch
- status/classification: `unknown_or_not_applicable`
- report_count/checkpoint_count/script_count/test_count: `51/126/40/0`
- recommendation: `historical_archive`
- key reports:
  - `reports/final_clean_results_table.md`
  - `reports/final_figure_index.md`
  - `reports/final_negative_results_summary.md`
  - `reports/final_post_s11_code_optimization_closure.md`
  - `reports/final_post_s14_code_optimization_closure.md`
  - `reports/final_post_s15_trajectory_training_closure.md`
- key checkpoints:
  - `checkpoints/S1d5_final_figures/scale_repair_summary.png`
  - `checkpoints/S1d5_final_mainline_summary.md`
  - `checkpoints/S1d5_final_repro/s1d5_policy_eval_summary.json`
  - `checkpoints/S2_final_fine_refinement_summary.md`
  - `checkpoints/S2a_fine_rot_policy_audit_on_s1d5/rot_0p2/s1d5_policy_eval_summary.json`
  - `checkpoints/S2a_fine_rot_policy_audit_on_s1d5/rot_0p25/s1d5_policy_eval_summary.json`

### optimize/s17-pose-supervision-dataset-quality-audit

- group: `historical_optimize_experiments`
- role: historical optimization experiment branch
- status/classification: `unknown_or_not_applicable`
- report_count/checkpoint_count/script_count/test_count: `53/137/40/0`
- recommendation: `historical_archive`
- key reports:
  - `reports/final_clean_results_table.md`
  - `reports/final_figure_index.md`
  - `reports/final_negative_results_summary.md`
  - `reports/final_post_s11_code_optimization_closure.md`
  - `reports/final_post_s14_code_optimization_closure.md`
  - `reports/final_post_s15_trajectory_training_closure.md`
- key checkpoints:
  - `checkpoints/S1d5_final_figures/scale_repair_summary.png`
  - `checkpoints/S1d5_final_mainline_summary.md`
  - `checkpoints/S1d5_final_repro/s1d5_policy_eval_summary.json`
  - `checkpoints/S2_final_fine_refinement_summary.md`
  - `checkpoints/S2a_fine_rot_policy_audit_on_s1d5/rot_0p2/s1d5_policy_eval_summary.json`
  - `checkpoints/S2a_fine_rot_policy_audit_on_s1d5/rot_0p25/s1d5_policy_eval_summary.json`

### optimize/s19-geometry-aware-pretraining-feasibility

- group: `historical_optimize_experiments`
- role: historical optimization experiment branch
- status/classification: `unknown_or_not_applicable`
- report_count/checkpoint_count/script_count/test_count: `56/143/42/0`
- recommendation: `historical_archive`
- key reports:
  - `reports/final_clean_results_table.md`
  - `reports/final_figure_index.md`
  - `reports/final_negative_results_summary.md`
  - `reports/final_post_s11_code_optimization_closure.md`
  - `reports/final_post_s14_code_optimization_closure.md`
  - `reports/final_post_s15_trajectory_training_closure.md`
- key checkpoints:
  - `checkpoints/S1d5_final_figures/scale_repair_summary.png`
  - `checkpoints/S1d5_final_mainline_summary.md`
  - `checkpoints/S1d5_final_repro/s1d5_policy_eval_summary.json`
  - `checkpoints/S2_final_fine_refinement_summary.md`
  - `checkpoints/S2a_fine_rot_policy_audit_on_s1d5/rot_0p2/s1d5_policy_eval_summary.json`
  - `checkpoints/S2a_fine_rot_policy_audit_on_s1d5/rot_0p25/s1d5_policy_eval_summary.json`

### optimize/s2-fine-refinement-on-s1d5

- group: `historical_optimize_experiments`
- role: historical optimization experiment branch
- status/classification: `unknown_or_not_applicable`
- report_count/checkpoint_count/script_count/test_count: `22/44/23/0`
- recommendation: `historical_archive`
- key reports:
  - `reports/final_s1d5_english_abstract.md`
  - `reports/final_s1d5_mainline_report.md`
  - `reports/final_s1d5_results_table.csv`
  - `reports/final_s1d5_thesis_summary.md`
  - `reports/final_s2_fine_refinement_summary.md`
  - `reports/git_track_s1d5_final_files.md`
- key checkpoints:
  - `checkpoints/S1d5_final_figures/scale_repair_summary.png`
  - `checkpoints/S1d5_final_mainline_summary.md`
  - `checkpoints/S1d5_final_repro/s1d5_policy_eval_summary.json`
  - `checkpoints/S2_final_fine_refinement_summary.md`
  - `checkpoints/S2a_fine_rot_policy_audit_on_s1d5/rot_0p2/s1d5_policy_eval_summary.json`
  - `checkpoints/S2a_fine_rot_policy_audit_on_s1d5/rot_0p25/s1d5_policy_eval_summary.json`

### optimize/s3a0-coupled-pose-residual-head

- group: `historical_optimize_experiments`
- role: historical optimization experiment branch
- status/classification: `unknown_or_not_applicable`
- report_count/checkpoint_count/script_count/test_count: `34/89/32/0`
- recommendation: `historical_archive`
- key reports:
  - `reports/final_clean_results_table.md`
  - `reports/final_figure_index.md`
  - `reports/final_negative_results_summary.md`
  - `reports/final_post_s9_code_optimization_closure.md`
  - `reports/final_project_mainline_summary.md`
  - `reports/final_reproducibility_guide.md`
- key checkpoints:
  - `checkpoints/S1d5_final_figures/scale_repair_summary.png`
  - `checkpoints/S1d5_final_mainline_summary.md`
  - `checkpoints/S1d5_final_repro/s1d5_policy_eval_summary.json`
  - `checkpoints/S2_final_fine_refinement_summary.md`
  - `checkpoints/S2a_fine_rot_policy_audit_on_s1d5/rot_0p2/s1d5_policy_eval_summary.json`
  - `checkpoints/S2a_fine_rot_policy_audit_on_s1d5/rot_0p25/s1d5_policy_eval_summary.json`

### polish/final-reproducibility-guardrails

- group: `final_or_polish`
- role: authoritative clean baseline and final-report branch
- status/classification: `S5_clean_tmag_calibration_policy remains final clean candidate`
- report_count/checkpoint_count/script_count/test_count: `69/151/52/11`
- recommendation: `canonical_for_topic`
- key reports:
  - `reports/final_ablation_baseline_comparison.md`
  - `reports/final_clean_results_table.md`
  - `reports/final_figure_index.md`
  - `reports/final_negative_results_summary.md`
  - `reports/final_post_s11_code_optimization_closure.md`
  - `reports/final_post_s14_code_optimization_closure.md`
- key checkpoints:
  - `checkpoints/AB1_final_ablation_baseline_comparison_results.json`
  - `checkpoints/AB2_thesis_three_axis_ablation_results.json`
  - `checkpoints/EXT1_external_algorithm_baseline_results.json`
  - `checkpoints/SB1_pano_orb_vo_baseline_results.json`
  - `checkpoints/SB2_alignment_consistent_baseline_results.json`
  - `checkpoints/S1d5_final_figures/scale_repair_summary.png`

## Cross-Branch Report Matrix

| topic | branches | paths | same content? | canonical recommendation |
|---|---|---|---|---|
| `JRT1_branch_remote_sync_audit.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner` | `reports/JRT1_branch_remote_sync_audit.md` | `True` | experiment/jrt1-joint-rtdir-coupled-refiner -> reports/JRT1_branch_remote_sync_audit.md |
| `JRT1_final_closeout_summary.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner` | `reports/JRT1_final_closeout_summary.md` | `True` | experiment/jrt1-joint-rtdir-coupled-refiner -> reports/JRT1_final_closeout_summary.md |
| `JRT1a_framework_smoke_report.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner` | `reports/JRT1a_framework_smoke_report.md` | `True` | experiment/jrt1-joint-rtdir-coupled-refiner -> reports/JRT1a_framework_smoke_report.md |
| `JRT1b_train_cv_report.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner` | `reports/JRT1b_train_cv_report.md` | `True` | experiment/jrt1-joint-rtdir-coupled-refiner -> reports/JRT1b_train_cv_report.md |
| `JRT1c_train_cv_trajectory_proxy_report.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner` | `reports/JRT1c_train_cv_trajectory_proxy_report.md` | `True` | experiment/jrt1-joint-rtdir-coupled-refiner -> reports/JRT1c_train_cv_trajectory_proxy_report.md |
| `MF1_final_closeout_summary.md` | `experiment/mf1-multi-frame-chain-refiner` | `reports/MF1_final_closeout_summary.md` | `True` | experiment/mf1-multi-frame-chain-refiner -> reports/MF1_final_closeout_summary.md |
| `MF1a_chain_refiner_smoke_report.md` | `experiment/mf1-multi-frame-chain-refiner` | `reports/MF1a_chain_refiner_smoke_report.md` | `True` | experiment/mf1-multi-frame-chain-refiner -> reports/MF1a_chain_refiner_smoke_report.md |
| `MF1b_train_cv_report.md` | `experiment/mf1-multi-frame-chain-refiner` | `reports/MF1b_train_cv_report.md` | `True` | experiment/mf1-multi-frame-chain-refiner -> reports/MF1b_train_cv_report.md |
| `REPO2_untracked_experiment_artifact_resolution.md` | `experiment/s5d2-dense-export-convention-audit` | `reports/REPO2_untracked_experiment_artifact_resolution.md` | `True` | experiment/s5d2-dense-export-convention-audit -> reports/REPO2_untracked_experiment_artifact_resolution.md |
| `REPORT_INDEX.md` | `experiment/s5d2-dense-export-convention-audit` | `reports/REPORT_INDEX.md` | `True` | experiment/s5d2-dense-export-convention-audit -> reports/REPORT_INDEX.md |
| `S3a0_minimal_coupled_pose_head_implementation_checklist.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, optimize/s10-chain-pathratio-smoother, optimize/s11-tmag-scale-consistency-training, optimize/s12-regime-balanced-sampling, optimize/s14-local-window-pose-graph, optimize/s15-trajectory-level-training-objective, optimize/s16-stronger-visual-backbone-feasibility, optimize/s17-pose-supervision-dataset-quality-audit, optimize/s19-geometry-aware-pretraining-feasibility, optimize/s2-fine-refinement-on-s1d5, optimize/s3a0-coupled-pose-residual-head, polish/final-reproducibility-guardrails` | `reports/S3a0_minimal_coupled_pose_head_implementation_checklist.md` | `True` | experiment/jrt1-joint-rtdir-coupled-refiner -> reports/S3a0_minimal_coupled_pose_head_implementation_checklist.md |
| `S3a_coupled_pose_head_design_plan.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, optimize/s10-chain-pathratio-smoother, optimize/s11-tmag-scale-consistency-training, optimize/s12-regime-balanced-sampling, optimize/s14-local-window-pose-graph, optimize/s15-trajectory-level-training-objective, optimize/s16-stronger-visual-backbone-feasibility, optimize/s17-pose-supervision-dataset-quality-audit, optimize/s19-geometry-aware-pretraining-feasibility, optimize/s2-fine-refinement-on-s1d5, optimize/s3a0-coupled-pose-residual-head, polish/final-reproducibility-guardrails` | `reports/S3a_coupled_pose_head_design_plan.md` | `True` | experiment/jrt1-joint-rtdir-coupled-refiner -> reports/S3a_coupled_pose_head_design_plan.md |
| `S5D2_dense_export_convention_audit.md` | `experiment/s5d2-dense-export-convention-audit` | `reports/S5D2_dense_export_convention_audit.md` | `True` | experiment/s5d2-dense-export-convention-audit -> reports/S5D2_dense_export_convention_audit.md |
| `alignment_consistent_strong_baseline_comparison.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, polish/final-reproducibility-guardrails` | `reports/alignment_consistent_strong_baseline_comparison.md` | `True` | experiment/jrt1-joint-rtdir-coupled-refiner -> reports/alignment_consistent_strong_baseline_comparison.md |
| `all_branch_inventory_and_classification.md` | `experiment/s5d2-dense-export-convention-audit` | `reports/all_branch_inventory_and_classification.md` | `True` | experiment/s5d2-dense-export-convention-audit -> reports/all_branch_inventory_and_classification.md |
| `checkpoints_live_cleanup_inventory.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, main, optimize/s10-chain-pathratio-smoother, optimize/s11-tmag-scale-consistency-training, optimize/s12-regime-balanced-sampling, optimize/s14-local-window-pose-graph, optimize/s15-trajectory-level-training-objective, optimize/s16-stronger-visual-backbone-feasibility, optimize/s17-pose-supervision-dataset-quality-audit, optimize/s19-geometry-aware-pretraining-feasibility, optimize/s2-fine-refinement-on-s1d5, optimize/s3a0-coupled-pose-residual-head, polish/final-reproducibility-guardrails` | `reports/checkpoints_live_cleanup_inventory.md` | `True` | experiment/jrt1-joint-rtdir-coupled-refiner -> reports/checkpoints_live_cleanup_inventory.md |
| `checkpoints_live_cleanup_state.json` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, main, optimize/s10-chain-pathratio-smoother, optimize/s11-tmag-scale-consistency-training, optimize/s12-regime-balanced-sampling, optimize/s14-local-window-pose-graph, optimize/s15-trajectory-level-training-objective, optimize/s16-stronger-visual-backbone-feasibility, optimize/s17-pose-supervision-dataset-quality-audit, optimize/s19-geometry-aware-pretraining-feasibility, optimize/s2-fine-refinement-on-s1d5, optimize/s3a0-coupled-pose-residual-head, polish/final-reproducibility-guardrails` | `reports/checkpoints_live_cleanup_state.json` | `True` | experiment/jrt1-joint-rtdir-coupled-refiner -> reports/checkpoints_live_cleanup_state.json |
| `checkpoints_live_cleanup_summary.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, main, optimize/s10-chain-pathratio-smoother, optimize/s11-tmag-scale-consistency-training, optimize/s12-regime-balanced-sampling, optimize/s14-local-window-pose-graph, optimize/s15-trajectory-level-training-objective, optimize/s16-stronger-visual-backbone-feasibility, optimize/s17-pose-supervision-dataset-quality-audit, optimize/s19-geometry-aware-pretraining-feasibility, optimize/s2-fine-refinement-on-s1d5, optimize/s3a0-coupled-pose-residual-head, polish/final-reproducibility-guardrails` | `reports/checkpoints_live_cleanup_summary.md` | `True` | experiment/jrt1-joint-rtdir-coupled-refiner -> reports/checkpoints_live_cleanup_summary.md |
| `cleanup_actions_s1d5.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, main, optimize/s10-chain-pathratio-smoother, optimize/s11-tmag-scale-consistency-training, optimize/s12-regime-balanced-sampling, optimize/s14-local-window-pose-graph, optimize/s15-trajectory-level-training-objective, optimize/s16-stronger-visual-backbone-feasibility, optimize/s17-pose-supervision-dataset-quality-audit, optimize/s19-geometry-aware-pretraining-feasibility, optimize/s2-fine-refinement-on-s1d5, optimize/s3a0-coupled-pose-residual-head, polish/final-reproducibility-guardrails` | `reports/cleanup_actions_s1d5.md` | `True` | experiment/jrt1-joint-rtdir-coupled-refiner -> reports/cleanup_actions_s1d5.md |
| `cleanup_delete_candidates_s1d5.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, main, optimize/s10-chain-pathratio-smoother, optimize/s11-tmag-scale-consistency-training, optimize/s12-regime-balanced-sampling, optimize/s14-local-window-pose-graph, optimize/s15-trajectory-level-training-objective, optimize/s16-stronger-visual-backbone-feasibility, optimize/s17-pose-supervision-dataset-quality-audit, optimize/s19-geometry-aware-pretraining-feasibility, optimize/s2-fine-refinement-on-s1d5, optimize/s3a0-coupled-pose-residual-head, polish/final-reproducibility-guardrails` | `reports/cleanup_delete_candidates_s1d5.md` | `True` | experiment/jrt1-joint-rtdir-coupled-refiner -> reports/cleanup_delete_candidates_s1d5.md |
| `cleanup_inventory_s1d5.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, main, optimize/s10-chain-pathratio-smoother, optimize/s11-tmag-scale-consistency-training, optimize/s12-regime-balanced-sampling, optimize/s14-local-window-pose-graph, optimize/s15-trajectory-level-training-objective, optimize/s16-stronger-visual-backbone-feasibility, optimize/s17-pose-supervision-dataset-quality-audit, optimize/s19-geometry-aware-pretraining-feasibility, optimize/s2-fine-refinement-on-s1d5, optimize/s3a0-coupled-pose-residual-head, polish/final-reproducibility-guardrails` | `reports/cleanup_inventory_s1d5.md` | `True` | experiment/jrt1-joint-rtdir-coupled-refiner -> reports/cleanup_inventory_s1d5.md |
| `cleanup_summary_s1d5.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, main, optimize/s10-chain-pathratio-smoother, optimize/s11-tmag-scale-consistency-training, optimize/s12-regime-balanced-sampling, optimize/s14-local-window-pose-graph, optimize/s15-trajectory-level-training-objective, optimize/s16-stronger-visual-backbone-feasibility, optimize/s17-pose-supervision-dataset-quality-audit, optimize/s19-geometry-aware-pretraining-feasibility, optimize/s2-fine-refinement-on-s1d5, optimize/s3a0-coupled-pose-residual-head, polish/final-reproducibility-guardrails` | `reports/cleanup_summary_s1d5.md` | `True` | experiment/jrt1-joint-rtdir-coupled-refiner -> reports/cleanup_summary_s1d5.md |
| `current_valid_baselines.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, main, optimize/s10-chain-pathratio-smoother, optimize/s11-tmag-scale-consistency-training, optimize/s12-regime-balanced-sampling, optimize/s14-local-window-pose-graph, optimize/s15-trajectory-level-training-objective, optimize/s16-stronger-visual-backbone-feasibility, optimize/s17-pose-supervision-dataset-quality-audit, optimize/s19-geometry-aware-pretraining-feasibility, optimize/s2-fine-refinement-on-s1d5, optimize/s3a0-coupled-pose-residual-head, polish/final-reproducibility-guardrails` | `reports/current_valid_baselines.md` | `False` | experiment/jrt1-joint-rtdir-coupled-refiner -> reports/current_valid_baselines.md |
| `droidslam_environment_setup_report.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, polish/final-reproducibility-guardrails` | `reports/droidslam_environment_setup_report.md` | `True` | experiment/jrt1-joint-rtdir-coupled-refiner -> reports/droidslam_environment_setup_report.md |
| `external_algorithm_baseline_comparison.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, polish/final-reproducibility-guardrails` | `reports/external_algorithm_baseline_comparison.md` | `True` | experiment/jrt1-joint-rtdir-coupled-refiner -> reports/external_algorithm_baseline_comparison.md |
| `external_algorithm_baseline_protocol.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, polish/final-reproducibility-guardrails` | `reports/external_algorithm_baseline_protocol.md` | `True` | experiment/jrt1-joint-rtdir-coupled-refiner -> reports/external_algorithm_baseline_protocol.md |
| `external_baseline_environment_audit.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, polish/final-reproducibility-guardrails` | `reports/external_baseline_environment_audit.md` | `True` | experiment/jrt1-joint-rtdir-coupled-refiner -> reports/external_baseline_environment_audit.md |
| `final_ablation_baseline_comparison.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, polish/final-reproducibility-guardrails` | `reports/final_ablation_baseline_comparison.md` | `True` | polish/final-reproducibility-guardrails -> reports/final_ablation_baseline_comparison.md |
| `final_clean_results_table.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, optimize/s10-chain-pathratio-smoother, optimize/s11-tmag-scale-consistency-training, optimize/s12-regime-balanced-sampling, optimize/s14-local-window-pose-graph, optimize/s15-trajectory-level-training-objective, optimize/s16-stronger-visual-backbone-feasibility, optimize/s17-pose-supervision-dataset-quality-audit, optimize/s19-geometry-aware-pretraining-feasibility, optimize/s3a0-coupled-pose-residual-head, polish/final-reproducibility-guardrails` | `reports/final_clean_results_table.md` | `False` | polish/final-reproducibility-guardrails -> reports/final_clean_results_table.md |
| `final_figure_index.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, optimize/s10-chain-pathratio-smoother, optimize/s11-tmag-scale-consistency-training, optimize/s12-regime-balanced-sampling, optimize/s14-local-window-pose-graph, optimize/s15-trajectory-level-training-objective, optimize/s16-stronger-visual-backbone-feasibility, optimize/s17-pose-supervision-dataset-quality-audit, optimize/s19-geometry-aware-pretraining-feasibility, optimize/s3a0-coupled-pose-residual-head, polish/final-reproducibility-guardrails` | `reports/final_figure_index.md` | `True` | polish/final-reproducibility-guardrails -> reports/final_figure_index.md |
| `final_negative_results_summary.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, optimize/s10-chain-pathratio-smoother, optimize/s11-tmag-scale-consistency-training, optimize/s12-regime-balanced-sampling, optimize/s14-local-window-pose-graph, optimize/s15-trajectory-level-training-objective, optimize/s16-stronger-visual-backbone-feasibility, optimize/s17-pose-supervision-dataset-quality-audit, optimize/s19-geometry-aware-pretraining-feasibility, optimize/s3a0-coupled-pose-residual-head, polish/final-reproducibility-guardrails` | `reports/final_negative_results_summary.md` | `False` | polish/final-reproducibility-guardrails -> reports/final_negative_results_summary.md |
| `final_post_s11_code_optimization_closure.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, optimize/s11-tmag-scale-consistency-training, optimize/s12-regime-balanced-sampling, optimize/s14-local-window-pose-graph, optimize/s15-trajectory-level-training-objective, optimize/s16-stronger-visual-backbone-feasibility, optimize/s17-pose-supervision-dataset-quality-audit, optimize/s19-geometry-aware-pretraining-feasibility, polish/final-reproducibility-guardrails` | `reports/final_post_s11_code_optimization_closure.md` | `True` | polish/final-reproducibility-guardrails -> reports/final_post_s11_code_optimization_closure.md |
| `final_post_s14_code_optimization_closure.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, optimize/s14-local-window-pose-graph, optimize/s15-trajectory-level-training-objective, optimize/s16-stronger-visual-backbone-feasibility, optimize/s17-pose-supervision-dataset-quality-audit, optimize/s19-geometry-aware-pretraining-feasibility, polish/final-reproducibility-guardrails` | `reports/final_post_s14_code_optimization_closure.md` | `True` | polish/final-reproducibility-guardrails -> reports/final_post_s14_code_optimization_closure.md |
| `final_post_s15_trajectory_training_closure.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, optimize/s15-trajectory-level-training-objective, optimize/s16-stronger-visual-backbone-feasibility, optimize/s17-pose-supervision-dataset-quality-audit, optimize/s19-geometry-aware-pretraining-feasibility, polish/final-reproducibility-guardrails` | `reports/final_post_s15_trajectory_training_closure.md` | `True` | polish/final-reproducibility-guardrails -> reports/final_post_s15_trajectory_training_closure.md |
| `final_post_s16_backbone_feasibility_closure.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, optimize/s16-stronger-visual-backbone-feasibility, optimize/s17-pose-supervision-dataset-quality-audit, optimize/s19-geometry-aware-pretraining-feasibility, polish/final-reproducibility-guardrails` | `reports/final_post_s16_backbone_feasibility_closure.md` | `True` | polish/final-reproducibility-guardrails -> reports/final_post_s16_backbone_feasibility_closure.md |
| `final_post_s19_geometry_pretraining_closure.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, optimize/s19-geometry-aware-pretraining-feasibility, polish/final-reproducibility-guardrails` | `reports/final_post_s19_geometry_pretraining_closure.md` | `True` | polish/final-reproducibility-guardrails -> reports/final_post_s19_geometry_pretraining_closure.md |
| `final_post_s9_code_optimization_closure.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, optimize/s10-chain-pathratio-smoother, optimize/s11-tmag-scale-consistency-training, optimize/s12-regime-balanced-sampling, optimize/s14-local-window-pose-graph, optimize/s15-trajectory-level-training-objective, optimize/s16-stronger-visual-backbone-feasibility, optimize/s17-pose-supervision-dataset-quality-audit, optimize/s19-geometry-aware-pretraining-feasibility, optimize/s3a0-coupled-pose-residual-head, polish/final-reproducibility-guardrails` | `reports/final_post_s9_code_optimization_closure.md` | `True` | polish/final-reproducibility-guardrails -> reports/final_post_s9_code_optimization_closure.md |
| `final_project_mainline_summary.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, optimize/s10-chain-pathratio-smoother, optimize/s11-tmag-scale-consistency-training, optimize/s12-regime-balanced-sampling, optimize/s14-local-window-pose-graph, optimize/s15-trajectory-level-training-objective, optimize/s16-stronger-visual-backbone-feasibility, optimize/s17-pose-supervision-dataset-quality-audit, optimize/s19-geometry-aware-pretraining-feasibility, optimize/s3a0-coupled-pose-residual-head, polish/final-reproducibility-guardrails` | `reports/final_project_mainline_summary.md` | `False` | polish/final-reproducibility-guardrails -> reports/final_project_mainline_summary.md |
| `final_reproducibility_guide.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, optimize/s10-chain-pathratio-smoother, optimize/s11-tmag-scale-consistency-training, optimize/s12-regime-balanced-sampling, optimize/s14-local-window-pose-graph, optimize/s15-trajectory-level-training-objective, optimize/s16-stronger-visual-backbone-feasibility, optimize/s17-pose-supervision-dataset-quality-audit, optimize/s19-geometry-aware-pretraining-feasibility, optimize/s3a0-coupled-pose-residual-head, polish/final-reproducibility-guardrails` | `reports/final_reproducibility_guide.md` | `False` | polish/final-reproducibility-guardrails -> reports/final_reproducibility_guide.md |
| `final_s10_chain_smoother_summary.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, optimize/s10-chain-pathratio-smoother, optimize/s11-tmag-scale-consistency-training, optimize/s12-regime-balanced-sampling, optimize/s14-local-window-pose-graph, optimize/s15-trajectory-level-training-objective, optimize/s16-stronger-visual-backbone-feasibility, optimize/s17-pose-supervision-dataset-quality-audit, optimize/s19-geometry-aware-pretraining-feasibility, polish/final-reproducibility-guardrails` | `reports/final_s10_chain_smoother_summary.md` | `True` | polish/final-reproducibility-guardrails -> reports/final_s10_chain_smoother_summary.md |
| `final_s11_tmag_scale_consistency_summary.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, optimize/s11-tmag-scale-consistency-training, optimize/s12-regime-balanced-sampling, optimize/s14-local-window-pose-graph, optimize/s15-trajectory-level-training-objective, optimize/s16-stronger-visual-backbone-feasibility, optimize/s17-pose-supervision-dataset-quality-audit, optimize/s19-geometry-aware-pretraining-feasibility, polish/final-reproducibility-guardrails` | `reports/final_s11_tmag_scale_consistency_summary.md` | `True` | polish/final-reproducibility-guardrails -> reports/final_s11_tmag_scale_consistency_summary.md |
| `final_s12_regime_balanced_sampling_summary.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, optimize/s12-regime-balanced-sampling, optimize/s14-local-window-pose-graph, optimize/s15-trajectory-level-training-objective, optimize/s16-stronger-visual-backbone-feasibility, optimize/s17-pose-supervision-dataset-quality-audit, optimize/s19-geometry-aware-pretraining-feasibility, polish/final-reproducibility-guardrails` | `reports/final_s12_regime_balanced_sampling_summary.md` | `True` | polish/final-reproducibility-guardrails -> reports/final_s12_regime_balanced_sampling_summary.md |
| `final_s13_practical_usability_gap_summary.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, optimize/s12-regime-balanced-sampling, optimize/s14-local-window-pose-graph, optimize/s15-trajectory-level-training-objective, optimize/s16-stronger-visual-backbone-feasibility, optimize/s17-pose-supervision-dataset-quality-audit, optimize/s19-geometry-aware-pretraining-feasibility, polish/final-reproducibility-guardrails` | `reports/final_s13_practical_usability_gap_summary.md` | `True` | polish/final-reproducibility-guardrails -> reports/final_s13_practical_usability_gap_summary.md |
| `final_s14_local_window_pose_graph_summary.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, optimize/s14-local-window-pose-graph, optimize/s15-trajectory-level-training-objective, optimize/s16-stronger-visual-backbone-feasibility, optimize/s17-pose-supervision-dataset-quality-audit, optimize/s19-geometry-aware-pretraining-feasibility, polish/final-reproducibility-guardrails` | `reports/final_s14_local_window_pose_graph_summary.md` | `True` | polish/final-reproducibility-guardrails -> reports/final_s14_local_window_pose_graph_summary.md |
| `final_s15_trajectory_level_training_objective_summary.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, optimize/s15-trajectory-level-training-objective, optimize/s16-stronger-visual-backbone-feasibility, optimize/s17-pose-supervision-dataset-quality-audit, optimize/s19-geometry-aware-pretraining-feasibility, polish/final-reproducibility-guardrails` | `reports/final_s15_trajectory_level_training_objective_summary.md` | `True` | polish/final-reproducibility-guardrails -> reports/final_s15_trajectory_level_training_objective_summary.md |
| `final_s15a_trajectory_training_smoke_failure_attribution_summary.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, optimize/s15-trajectory-level-training-objective, optimize/s16-stronger-visual-backbone-feasibility, optimize/s17-pose-supervision-dataset-quality-audit, optimize/s19-geometry-aware-pretraining-feasibility, polish/final-reproducibility-guardrails` | `reports/final_s15a_trajectory_training_smoke_failure_attribution_summary.md` | `True` | polish/final-reproducibility-guardrails -> reports/final_s15a_trajectory_training_smoke_failure_attribution_summary.md |
| `final_s15b_pair_training_harness_stability_audit_summary.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, optimize/s15-trajectory-level-training-objective, optimize/s16-stronger-visual-backbone-feasibility, optimize/s17-pose-supervision-dataset-quality-audit, optimize/s19-geometry-aware-pretraining-feasibility, polish/final-reproducibility-guardrails` | `reports/final_s15b_pair_training_harness_stability_audit_summary.md` | `True` | polish/final-reproducibility-guardrails -> reports/final_s15b_pair_training_harness_stability_audit_summary.md |
| `final_s15c_clean_policy_wrapped_training_harness_fix_summary.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, optimize/s15-trajectory-level-training-objective, optimize/s16-stronger-visual-backbone-feasibility, optimize/s17-pose-supervision-dataset-quality-audit, optimize/s19-geometry-aware-pretraining-feasibility, polish/final-reproducibility-guardrails` | `reports/final_s15c_clean_policy_wrapped_training_harness_fix_summary.md` | `True` | polish/final-reproducibility-guardrails -> reports/final_s15c_clean_policy_wrapped_training_harness_fix_summary.md |
| `final_s15d_train_eval_forward_parity_fix_summary.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, optimize/s15-trajectory-level-training-objective, optimize/s16-stronger-visual-backbone-feasibility, optimize/s17-pose-supervision-dataset-quality-audit, optimize/s19-geometry-aware-pretraining-feasibility, polish/final-reproducibility-guardrails` | `reports/final_s15d_train_eval_forward_parity_fix_summary.md` | `True` | polish/final-reproducibility-guardrails -> reports/final_s15d_train_eval_forward_parity_fix_summary.md |
| `final_s15e_tiny_trajectory_retest_after_harness_fix_summary.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, optimize/s15-trajectory-level-training-objective, optimize/s16-stronger-visual-backbone-feasibility, optimize/s17-pose-supervision-dataset-quality-audit, optimize/s19-geometry-aware-pretraining-feasibility, polish/final-reproducibility-guardrails` | `reports/final_s15e_tiny_trajectory_retest_after_harness_fix_summary.md` | `True` | polish/final-reproducibility-guardrails -> reports/final_s15e_tiny_trajectory_retest_after_harness_fix_summary.md |
| `final_s16_stronger_visual_backbone_feasibility_summary.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, optimize/s16-stronger-visual-backbone-feasibility, optimize/s17-pose-supervision-dataset-quality-audit, optimize/s19-geometry-aware-pretraining-feasibility, polish/final-reproducibility-guardrails` | `reports/final_s16_stronger_visual_backbone_feasibility_summary.md` | `True` | polish/final-reproducibility-guardrails -> reports/final_s16_stronger_visual_backbone_feasibility_summary.md |
| `final_s16b_frozen_pretrained_backbone_probe_summary.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, optimize/s16-stronger-visual-backbone-feasibility, optimize/s17-pose-supervision-dataset-quality-audit, optimize/s19-geometry-aware-pretraining-feasibility, polish/final-reproducibility-guardrails` | `reports/final_s16b_frozen_pretrained_backbone_probe_summary.md` | `True` | polish/final-reproducibility-guardrails -> reports/final_s16b_frozen_pretrained_backbone_probe_summary.md |
| `final_s17_pose_supervision_dataset_quality_summary.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, optimize/s17-pose-supervision-dataset-quality-audit, optimize/s19-geometry-aware-pretraining-feasibility, polish/final-reproducibility-guardrails` | `reports/final_s17_pose_supervision_dataset_quality_summary.md` | `True` | polish/final-reproducibility-guardrails -> reports/final_s17_pose_supervision_dataset_quality_summary.md |
| `final_s18_split_redesign_representativeness_summary.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, optimize/s17-pose-supervision-dataset-quality-audit, optimize/s19-geometry-aware-pretraining-feasibility, polish/final-reproducibility-guardrails` | `reports/final_s18_split_redesign_representativeness_summary.md` | `True` | polish/final-reproducibility-guardrails -> reports/final_s18_split_redesign_representativeness_summary.md |
| `final_s19_geometry_aware_pretraining_feasibility_summary.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, optimize/s19-geometry-aware-pretraining-feasibility, polish/final-reproducibility-guardrails` | `reports/final_s19_geometry_aware_pretraining_feasibility_summary.md` | `True` | polish/final-reproducibility-guardrails -> reports/final_s19_geometry_aware_pretraining_feasibility_summary.md |
| `final_s1d5_english_abstract.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, main, optimize/s10-chain-pathratio-smoother, optimize/s11-tmag-scale-consistency-training, optimize/s12-regime-balanced-sampling, optimize/s14-local-window-pose-graph, optimize/s15-trajectory-level-training-objective, optimize/s16-stronger-visual-backbone-feasibility, optimize/s17-pose-supervision-dataset-quality-audit, optimize/s19-geometry-aware-pretraining-feasibility, optimize/s2-fine-refinement-on-s1d5, optimize/s3a0-coupled-pose-residual-head, polish/final-reproducibility-guardrails` | `reports/final_s1d5_english_abstract.md` | `True` | polish/final-reproducibility-guardrails -> reports/final_s1d5_english_abstract.md |
| `final_s1d5_mainline_report.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, main, optimize/s10-chain-pathratio-smoother, optimize/s11-tmag-scale-consistency-training, optimize/s12-regime-balanced-sampling, optimize/s14-local-window-pose-graph, optimize/s15-trajectory-level-training-objective, optimize/s16-stronger-visual-backbone-feasibility, optimize/s17-pose-supervision-dataset-quality-audit, optimize/s19-geometry-aware-pretraining-feasibility, optimize/s2-fine-refinement-on-s1d5, optimize/s3a0-coupled-pose-residual-head, polish/final-reproducibility-guardrails` | `reports/final_s1d5_mainline_report.md` | `True` | polish/final-reproducibility-guardrails -> reports/final_s1d5_mainline_report.md |
| `final_s1d5_results_table.csv` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, main, optimize/s10-chain-pathratio-smoother, optimize/s11-tmag-scale-consistency-training, optimize/s12-regime-balanced-sampling, optimize/s14-local-window-pose-graph, optimize/s15-trajectory-level-training-objective, optimize/s16-stronger-visual-backbone-feasibility, optimize/s17-pose-supervision-dataset-quality-audit, optimize/s19-geometry-aware-pretraining-feasibility, optimize/s2-fine-refinement-on-s1d5, optimize/s3a0-coupled-pose-residual-head, polish/final-reproducibility-guardrails` | `reports/final_s1d5_results_table.csv` | `True` | polish/final-reproducibility-guardrails -> reports/final_s1d5_results_table.csv |
| `final_s1d5_thesis_summary.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, main, optimize/s10-chain-pathratio-smoother, optimize/s11-tmag-scale-consistency-training, optimize/s12-regime-balanced-sampling, optimize/s14-local-window-pose-graph, optimize/s15-trajectory-level-training-objective, optimize/s16-stronger-visual-backbone-feasibility, optimize/s17-pose-supervision-dataset-quality-audit, optimize/s19-geometry-aware-pretraining-feasibility, optimize/s2-fine-refinement-on-s1d5, optimize/s3a0-coupled-pose-residual-head, polish/final-reproducibility-guardrails` | `reports/final_s1d5_thesis_summary.md` | `True` | polish/final-reproducibility-guardrails -> reports/final_s1d5_thesis_summary.md |
| `final_s2_fine_refinement_summary.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, optimize/s10-chain-pathratio-smoother, optimize/s11-tmag-scale-consistency-training, optimize/s12-regime-balanced-sampling, optimize/s14-local-window-pose-graph, optimize/s15-trajectory-level-training-objective, optimize/s16-stronger-visual-backbone-feasibility, optimize/s17-pose-supervision-dataset-quality-audit, optimize/s19-geometry-aware-pretraining-feasibility, optimize/s2-fine-refinement-on-s1d5, optimize/s3a0-coupled-pose-residual-head, polish/final-reproducibility-guardrails` | `reports/final_s2_fine_refinement_summary.md` | `True` | polish/final-reproducibility-guardrails -> reports/final_s2_fine_refinement_summary.md |
| `final_s3a_residual_head_summary.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, optimize/s10-chain-pathratio-smoother, optimize/s11-tmag-scale-consistency-training, optimize/s12-regime-balanced-sampling, optimize/s14-local-window-pose-graph, optimize/s15-trajectory-level-training-objective, optimize/s16-stronger-visual-backbone-feasibility, optimize/s17-pose-supervision-dataset-quality-audit, optimize/s19-geometry-aware-pretraining-feasibility, optimize/s3a0-coupled-pose-residual-head, polish/final-reproducibility-guardrails` | `reports/final_s3a_residual_head_summary.md` | `True` | polish/final-reproducibility-guardrails -> reports/final_s3a_residual_head_summary.md |
| `final_s5_clean_candidate_summary.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, optimize/s10-chain-pathratio-smoother, optimize/s11-tmag-scale-consistency-training, optimize/s12-regime-balanced-sampling, optimize/s14-local-window-pose-graph, optimize/s15-trajectory-level-training-objective, optimize/s16-stronger-visual-backbone-feasibility, optimize/s17-pose-supervision-dataset-quality-audit, optimize/s19-geometry-aware-pretraining-feasibility, optimize/s3a0-coupled-pose-residual-head, polish/final-reproducibility-guardrails` | `reports/final_s5_clean_candidate_summary.md` | `True` | polish/final-reproducibility-guardrails -> reports/final_s5_clean_candidate_summary.md |
| `final_s7_project_delivery_summary.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, optimize/s10-chain-pathratio-smoother, optimize/s11-tmag-scale-consistency-training, optimize/s12-regime-balanced-sampling, optimize/s14-local-window-pose-graph, optimize/s15-trajectory-level-training-objective, optimize/s16-stronger-visual-backbone-feasibility, optimize/s17-pose-supervision-dataset-quality-audit, optimize/s19-geometry-aware-pretraining-feasibility, optimize/s3a0-coupled-pose-residual-head, polish/final-reproducibility-guardrails` | `reports/final_s7_project_delivery_summary.md` | `True` | polish/final-reproducibility-guardrails -> reports/final_s7_project_delivery_summary.md |
| `final_s8_token_reliability_negative_summary.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, optimize/s10-chain-pathratio-smoother, optimize/s11-tmag-scale-consistency-training, optimize/s12-regime-balanced-sampling, optimize/s14-local-window-pose-graph, optimize/s15-trajectory-level-training-objective, optimize/s16-stronger-visual-backbone-feasibility, optimize/s17-pose-supervision-dataset-quality-audit, optimize/s19-geometry-aware-pretraining-feasibility, optimize/s3a0-coupled-pose-residual-head, polish/final-reproducibility-guardrails` | `reports/final_s8_token_reliability_negative_summary.md` | `True` | polish/final-reproducibility-guardrails -> reports/final_s8_token_reliability_negative_summary.md |
| `final_s9_regime_only_router_summary.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, optimize/s10-chain-pathratio-smoother, optimize/s11-tmag-scale-consistency-training, optimize/s12-regime-balanced-sampling, optimize/s14-local-window-pose-graph, optimize/s15-trajectory-level-training-objective, optimize/s16-stronger-visual-backbone-feasibility, optimize/s17-pose-supervision-dataset-quality-audit, optimize/s19-geometry-aware-pretraining-feasibility, optimize/s3a0-coupled-pose-residual-head, polish/final-reproducibility-guardrails` | `reports/final_s9_regime_only_router_summary.md` | `True` | polish/final-reproducibility-guardrails -> reports/final_s9_regime_only_router_summary.md |
| `final_thesis_claims_and_limitations.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, optimize/s10-chain-pathratio-smoother, optimize/s11-tmag-scale-consistency-training, optimize/s12-regime-balanced-sampling, optimize/s14-local-window-pose-graph, optimize/s15-trajectory-level-training-objective, optimize/s16-stronger-visual-backbone-feasibility, optimize/s17-pose-supervision-dataset-quality-audit, optimize/s19-geometry-aware-pretraining-feasibility, optimize/s3a0-coupled-pose-residual-head, polish/final-reproducibility-guardrails` | `reports/final_thesis_claims_and_limitations.md` | `False` | polish/final-reproducibility-guardrails -> reports/final_thesis_claims_and_limitations.md |
| `git_branch_remote_sync_audit.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, optimize/s19-geometry-aware-pretraining-feasibility, polish/final-reproducibility-guardrails` | `reports/git_branch_remote_sync_audit.md` | `True` | experiment/jrt1-joint-rtdir-coupled-refiner -> reports/git_branch_remote_sync_audit.md |
| `git_commit_s1d5_summary.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, main, optimize/s10-chain-pathratio-smoother, optimize/s11-tmag-scale-consistency-training, optimize/s12-regime-balanced-sampling, optimize/s14-local-window-pose-graph, optimize/s15-trajectory-level-training-objective, optimize/s16-stronger-visual-backbone-feasibility, optimize/s17-pose-supervision-dataset-quality-audit, optimize/s19-geometry-aware-pretraining-feasibility, optimize/s2-fine-refinement-on-s1d5, optimize/s3a0-coupled-pose-residual-head, polish/final-reproducibility-guardrails` | `reports/git_commit_s1d5_summary.md` | `True` | experiment/jrt1-joint-rtdir-coupled-refiner -> reports/git_commit_s1d5_summary.md |
| `git_commit_s2b_summary.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, optimize/s10-chain-pathratio-smoother, optimize/s11-tmag-scale-consistency-training, optimize/s12-regime-balanced-sampling, optimize/s14-local-window-pose-graph, optimize/s15-trajectory-level-training-objective, optimize/s16-stronger-visual-backbone-feasibility, optimize/s17-pose-supervision-dataset-quality-audit, optimize/s19-geometry-aware-pretraining-feasibility, optimize/s2-fine-refinement-on-s1d5, optimize/s3a0-coupled-pose-residual-head, polish/final-reproducibility-guardrails` | `reports/git_commit_s2b_summary.md` | `True` | experiment/jrt1-joint-rtdir-coupled-refiner -> reports/git_commit_s2b_summary.md |
| `git_staging_s1d5_precheck.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, main, optimize/s10-chain-pathratio-smoother, optimize/s11-tmag-scale-consistency-training, optimize/s12-regime-balanced-sampling, optimize/s14-local-window-pose-graph, optimize/s15-trajectory-level-training-objective, optimize/s16-stronger-visual-backbone-feasibility, optimize/s17-pose-supervision-dataset-quality-audit, optimize/s19-geometry-aware-pretraining-feasibility, optimize/s2-fine-refinement-on-s1d5, optimize/s3a0-coupled-pose-residual-head, polish/final-reproducibility-guardrails` | `reports/git_staging_s1d5_precheck.md` | `True` | experiment/jrt1-joint-rtdir-coupled-refiner -> reports/git_staging_s1d5_precheck.md |
| `git_staging_s1d5_summary.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, main, optimize/s10-chain-pathratio-smoother, optimize/s11-tmag-scale-consistency-training, optimize/s12-regime-balanced-sampling, optimize/s14-local-window-pose-graph, optimize/s15-trajectory-level-training-objective, optimize/s16-stronger-visual-backbone-feasibility, optimize/s17-pose-supervision-dataset-quality-audit, optimize/s19-geometry-aware-pretraining-feasibility, optimize/s2-fine-refinement-on-s1d5, optimize/s3a0-coupled-pose-residual-head, polish/final-reproducibility-guardrails` | `reports/git_staging_s1d5_summary.md` | `True` | experiment/jrt1-joint-rtdir-coupled-refiner -> reports/git_staging_s1d5_summary.md |
| `git_track_s1d5_final_files.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, main, optimize/s10-chain-pathratio-smoother, optimize/s11-tmag-scale-consistency-training, optimize/s12-regime-balanced-sampling, optimize/s14-local-window-pose-graph, optimize/s15-trajectory-level-training-objective, optimize/s16-stronger-visual-backbone-feasibility, optimize/s17-pose-supervision-dataset-quality-audit, optimize/s19-geometry-aware-pretraining-feasibility, optimize/s2-fine-refinement-on-s1d5, optimize/s3a0-coupled-pose-residual-head, polish/final-reproducibility-guardrails` | `reports/git_track_s1d5_final_files.md` | `True` | experiment/jrt1-joint-rtdir-coupled-refiner -> reports/git_track_s1d5_final_files.md |
| `gitignore_cleanup_plan.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, main, optimize/s10-chain-pathratio-smoother, optimize/s11-tmag-scale-consistency-training, optimize/s12-regime-balanced-sampling, optimize/s14-local-window-pose-graph, optimize/s15-trajectory-level-training-objective, optimize/s16-stronger-visual-backbone-feasibility, optimize/s17-pose-supervision-dataset-quality-audit, optimize/s19-geometry-aware-pretraining-feasibility, optimize/s2-fine-refinement-on-s1d5, optimize/s3a0-coupled-pose-residual-head, polish/final-reproducibility-guardrails` | `reports/gitignore_cleanup_plan.md` | `True` | experiment/jrt1-joint-rtdir-coupled-refiner -> reports/gitignore_cleanup_plan.md |
| `orbslam3_build_remediation_report.md` | `experiment/orbslam3-fisheye-strong-baseline` | `reports/orbslam3_build_remediation_report.md` | `True` | experiment/orbslam3-fisheye-strong-baseline -> reports/orbslam3_build_remediation_report.md |
| `orbslam3_environment_setup_report.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, polish/final-reproducibility-guardrails` | `reports/orbslam3_environment_setup_report.md` | `True` | experiment/orbslam3-fisheye-strong-baseline -> reports/orbslam3_environment_setup_report.md |
| `orbslam3_fisheye_calibration_conversion_report.md` | `experiment/orbslam3-fisheye-strong-baseline` | `reports/orbslam3_fisheye_calibration_conversion_report.md` | `True` | experiment/orbslam3-fisheye-strong-baseline -> reports/orbslam3_fisheye_calibration_conversion_report.md |
| `orbslam3_fisheye_dataset_audit.md` | `experiment/orbslam3-fisheye-strong-baseline` | `reports/orbslam3_fisheye_dataset_audit.md` | `True` | experiment/orbslam3-fisheye-strong-baseline -> reports/orbslam3_fisheye_dataset_audit.md |
| `orbslam3_fisheye_evaluation_report.md` | `experiment/orbslam3-fisheye-strong-baseline` | `reports/orbslam3_fisheye_evaluation_report.md` | `True` | experiment/orbslam3-fisheye-strong-baseline -> reports/orbslam3_fisheye_evaluation_report.md |
| `orbslam3_fisheye_run_report.md` | `experiment/orbslam3-fisheye-strong-baseline` | `reports/orbslam3_fisheye_run_report.md` | `True` | experiment/orbslam3-fisheye-strong-baseline -> reports/orbslam3_fisheye_run_report.md |
| `orbslam3_local_setup_report.md` | `experiment/orbslam3-fisheye-strong-baseline` | `reports/orbslam3_local_setup_report.md` | `True` | experiment/orbslam3-fisheye-strong-baseline -> reports/orbslam3_local_setup_report.md |
| `pano_orb_vo_baseline_report.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, polish/final-reproducibility-guardrails` | `reports/pano_orb_vo_baseline_report.md` | `True` | experiment/jrt1-joint-rtdir-coupled-refiner -> reports/pano_orb_vo_baseline_report.md |
| `pano_orb_vo_component_diagnostics.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, polish/final-reproducibility-guardrails` | `reports/pano_orb_vo_component_diagnostics.md` | `True` | experiment/jrt1-joint-rtdir-coupled-refiner -> reports/pano_orb_vo_component_diagnostics.md |
| `post_commit_workspace_cleanup_plan.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, main, optimize/s10-chain-pathratio-smoother, optimize/s11-tmag-scale-consistency-training, optimize/s12-regime-balanced-sampling, optimize/s14-local-window-pose-graph, optimize/s15-trajectory-level-training-objective, optimize/s16-stronger-visual-backbone-feasibility, optimize/s17-pose-supervision-dataset-quality-audit, optimize/s19-geometry-aware-pretraining-feasibility, optimize/s2-fine-refinement-on-s1d5, optimize/s3a0-coupled-pose-residual-head, polish/final-reproducibility-guardrails` | `reports/post_commit_workspace_cleanup_plan.md` | `True` | experiment/jrt1-joint-rtdir-coupled-refiner -> reports/post_commit_workspace_cleanup_plan.md |
| `post_s5d2_git_uncommitted_audit.md` | `experiment/s5d2-dense-export-convention-audit` | `reports/post_s5d2_git_uncommitted_audit.md` | `True` | experiment/s5d2-dense-export-convention-audit -> reports/post_s5d2_git_uncommitted_audit.md |
| `repo_remote_sync_and_hygiene_audit.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, polish/final-reproducibility-guardrails` | `reports/repo_remote_sync_and_hygiene_audit.md` | `True` | experiment/jrt1-joint-rtdir-coupled-refiner -> reports/repo_remote_sync_and_hygiene_audit.md |
| `reports_inventory_audit.md` | `experiment/s5d2-dense-export-convention-audit` | `reports/reports_inventory_audit.md` | `True` | experiment/s5d2-dense-export-convention-audit -> reports/reports_inventory_audit.md |
| `s5_dense_external_export_report.md` | `experiment/orbslam3-fisheye-strong-baseline` | `reports/s5_dense_external_export_report.md` | `True` | experiment/orbslam3-fisheye-strong-baseline -> reports/s5_dense_external_export_report.md |
| `s5_external_export_compatibility_audit.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, polish/final-reproducibility-guardrails` | `reports/s5_external_export_compatibility_audit.md` | `True` | experiment/jrt1-joint-rtdir-coupled-refiner -> reports/s5_external_export_compatibility_audit.md |
| `s5_orbslam3_same_evaluator_comparison.md` | `experiment/orbslam3-fisheye-strong-baseline` | `reports/s5_orbslam3_same_evaluator_comparison.md` | `True` | experiment/orbslam3-fisheye-strong-baseline -> reports/s5_orbslam3_same_evaluator_comparison.md |
| `thesis_experiment_section_draft.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, polish/final-reproducibility-guardrails` | `reports/thesis_experiment_section_draft.md` | `True` | experiment/jrt1-joint-rtdir-coupled-refiner -> reports/thesis_experiment_section_draft.md |
| `thesis_three_axis_ablation.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, polish/final-reproducibility-guardrails` | `reports/thesis_three_axis_ablation.md` | `True` | experiment/jrt1-joint-rtdir-coupled-refiner -> reports/thesis_three_axis_ablation.md |

## Divergent Same-Path Reports

| path | branches | content hashes | recommendation |
|---|---|---|---|
| `reports/current_valid_baselines.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, main, optimize/s10-chain-pathratio-smoother, optimize/s11-tmag-scale-consistency-training, optimize/s12-regime-balanced-sampling, optimize/s14-local-window-pose-graph, optimize/s15-trajectory-level-training-objective, optimize/s16-stronger-visual-backbone-feasibility, optimize/s17-pose-supervision-dataset-quality-audit, optimize/s19-geometry-aware-pretraining-feasibility, optimize/s2-fine-refinement-on-s1d5, optimize/s3a0-coupled-pose-residual-head, polish/final-reproducibility-guardrails` | `067b4aa3fce3, 067b4aa3fce3, 067b4aa3fce3, 067b4aa3fce3, 0c7f2b43f964, 067b4aa3fce3, 067b4aa3fce3, 067b4aa3fce3, 067b4aa3fce3, 067b4aa3fce3, 067b4aa3fce3, 067b4aa3fce3, 067b4aa3fce3, d2f147dab7b5, 067b4aa3fce3, 067b4aa3fce3` | keep branch-local; do not merge wholesale |
| `reports/final_clean_results_table.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, optimize/s10-chain-pathratio-smoother, optimize/s11-tmag-scale-consistency-training, optimize/s12-regime-balanced-sampling, optimize/s14-local-window-pose-graph, optimize/s15-trajectory-level-training-objective, optimize/s16-stronger-visual-backbone-feasibility, optimize/s17-pose-supervision-dataset-quality-audit, optimize/s19-geometry-aware-pretraining-feasibility, optimize/s3a0-coupled-pose-residual-head, polish/final-reproducibility-guardrails` | `9873bf49efc9, 9873bf49efc9, 9873bf49efc9, 9873bf49efc9, 91daf528e272, a26ca94df6bd, a26ca94df6bd, 6e4be91d5f77, 5d2a1ff66653, e6b53cae80ee, e6b53cae80ee, 9873bf49efc9, 91daf528e272, 9873bf49efc9` | keep branch-local; do not merge wholesale |
| `reports/final_negative_results_summary.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, optimize/s10-chain-pathratio-smoother, optimize/s11-tmag-scale-consistency-training, optimize/s12-regime-balanced-sampling, optimize/s14-local-window-pose-graph, optimize/s15-trajectory-level-training-objective, optimize/s16-stronger-visual-backbone-feasibility, optimize/s17-pose-supervision-dataset-quality-audit, optimize/s19-geometry-aware-pretraining-feasibility, optimize/s3a0-coupled-pose-residual-head, polish/final-reproducibility-guardrails` | `63ca4a28dccb, 63ca4a28dccb, 63ca4a28dccb, 63ca4a28dccb, 51af99adfd5e, 84aa31120068, 84aa31120068, decdb360ad01, 4f7edd397a7c, d45d4fbcdebe, d45d4fbcdebe, 63ca4a28dccb, 51af99adfd5e, 63ca4a28dccb` | keep branch-local; do not merge wholesale |
| `reports/final_project_mainline_summary.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, optimize/s10-chain-pathratio-smoother, optimize/s11-tmag-scale-consistency-training, optimize/s12-regime-balanced-sampling, optimize/s14-local-window-pose-graph, optimize/s15-trajectory-level-training-objective, optimize/s16-stronger-visual-backbone-feasibility, optimize/s17-pose-supervision-dataset-quality-audit, optimize/s19-geometry-aware-pretraining-feasibility, optimize/s3a0-coupled-pose-residual-head, polish/final-reproducibility-guardrails` | `847b23744411, 847b23744411, 847b23744411, 847b23744411, 558ac7b47f4a, 5596fe403971, 5596fe403971, 79fe2f1fe768, c93676499603, 00be5f8c52ea, 00be5f8c52ea, 847b23744411, 558ac7b47f4a, 847b23744411` | keep branch-local; do not merge wholesale |
| `reports/final_reproducibility_guide.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, optimize/s10-chain-pathratio-smoother, optimize/s11-tmag-scale-consistency-training, optimize/s12-regime-balanced-sampling, optimize/s14-local-window-pose-graph, optimize/s15-trajectory-level-training-objective, optimize/s16-stronger-visual-backbone-feasibility, optimize/s17-pose-supervision-dataset-quality-audit, optimize/s19-geometry-aware-pretraining-feasibility, optimize/s3a0-coupled-pose-residual-head, polish/final-reproducibility-guardrails` | `396abb483d16, 396abb483d16, 396abb483d16, 396abb483d16, 6d6c4f0a8420, 6d6c4f0a8420, 6d6c4f0a8420, 6d6c4f0a8420, 6d6c4f0a8420, 6d6c4f0a8420, 6d6c4f0a8420, 6d6c4f0a8420, 6d6c4f0a8420, 396abb483d16` | keep branch-local; do not merge wholesale |
| `reports/final_thesis_claims_and_limitations.md` | `experiment/jrt1-joint-rtdir-coupled-refiner, experiment/mf1-multi-frame-chain-refiner, experiment/orbslam3-fisheye-strong-baseline, experiment/s5d2-dense-export-convention-audit, optimize/s10-chain-pathratio-smoother, optimize/s11-tmag-scale-consistency-training, optimize/s12-regime-balanced-sampling, optimize/s14-local-window-pose-graph, optimize/s15-trajectory-level-training-objective, optimize/s16-stronger-visual-backbone-feasibility, optimize/s17-pose-supervision-dataset-quality-audit, optimize/s19-geometry-aware-pretraining-feasibility, optimize/s3a0-coupled-pose-residual-head, polish/final-reproducibility-guardrails` | `29fe5f867ce2, 29fe5f867ce2, 29fe5f867ce2, 29fe5f867ce2, 174d47f0b927, bd7b100cdd5a, bd7b100cdd5a, d46e7b762443, d641b0a633b3, 8493a4b028c8, 8493a4b028c8, 29fe5f867ce2, 174d47f0b927, 29fe5f867ce2` | keep branch-local; do not merge wholesale |

## Duplicate / Overlapping Topics

| topic | reports | superseded_by | recommendation |
|---|---|---|---|
| `JRT1_final_closeout_summary.md` | `reports/JRT1_final_closeout_summary.md` | `reports/JRT1_final_closeout_summary.md` | canonicalize later in REPORT3; no moves now |
| `MF1_final_closeout_summary.md` | `reports/MF1_final_closeout_summary.md` | `(none)` | canonicalize later in REPORT3; no moves now |
| `S5D2_dense_export_convention_audit.md` | `reports/S5D2_dense_export_convention_audit.md` | `(none)` | canonicalize later in REPORT3; no moves now |
| `current_valid_baselines.md` | `reports/current_valid_baselines.md` | `reports/current_valid_baselines.md` | canonicalize later in REPORT3; no moves now |
| `external_algorithm_baseline_comparison.md` | `reports/external_algorithm_baseline_comparison.md` | `reports/external_algorithm_baseline_comparison.md` | canonicalize later in REPORT3; no moves now |
| `final_ablation_baseline_comparison.md` | `reports/final_ablation_baseline_comparison.md` | `reports/final_ablation_baseline_comparison.md` | canonicalize later in REPORT3; no moves now |
| `final_clean_results_table.md` | `reports/final_clean_results_table.md` | `reports/final_clean_results_table.md` | canonicalize later in REPORT3; no moves now |
| `final_negative_results_summary.md` | `reports/final_negative_results_summary.md` | `reports/final_negative_results_summary.md` | canonicalize later in REPORT3; no moves now |
| `final_project_mainline_summary.md` | `reports/final_project_mainline_summary.md` | `reports/final_project_mainline_summary.md` | canonicalize later in REPORT3; no moves now |
| `final_reproducibility_guide.md` | `reports/final_reproducibility_guide.md` | `reports/final_reproducibility_guide.md` | canonicalize later in REPORT3; no moves now |
| `final_thesis_claims_and_limitations.md` | `reports/final_thesis_claims_and_limitations.md` | `reports/final_thesis_claims_and_limitations.md` | canonicalize later in REPORT3; no moves now |
| `s5_dense_external_export_report.md` | `reports/s5_dense_external_export_report.md` | `(none)` | canonicalize later in REPORT3; no moves now |
| `s5_orbslam3_same_evaluator_comparison.md` | `reports/s5_orbslam3_same_evaluator_comparison.md` | `(none)` | canonicalize later in REPORT3; no moves now |
| `thesis_three_axis_ablation.md` | `reports/thesis_three_axis_ablation.md` | `reports/thesis_three_axis_ablation.md` | canonicalize later in REPORT3; no moves now |

## Canonical Report Recommendations

| topic | canonical branch | canonical file | reason |
|---|---|---|---|
| final reproducibility guide | `polish/final-reproducibility-guardrails` | `reports/final_reproducibility_guide.md` | authoritative final reproducibility branch |
| external baseline comparison | `polish/final-reproducibility-guardrails` | `reports/external_algorithm_baseline_comparison.md` | thesis-facing baseline comparison on the final/polish line |
| pano-orb-vo baseline | `polish/final-reproducibility-guardrails` | `reports/pano_orb_vo_baseline_report.md` | final baseline branch keeps the stable panorama baseline story |
| orbslam3 fisheye baseline | `experiment/orbslam3-fisheye-strong-baseline` | `reports/orbslam3_fisheye_evaluation_report.md` | ORB-specific artifacts and reports live on the dedicated external baseline branch |
| s5 dense same-evaluator comparison | `experiment/orbslam3-fisheye-strong-baseline` | `reports/s5_orbslam3_same_evaluator_comparison.md` | paired comparison with ORB-SLAM3 belongs to the ORB branch |
| s5 dense export convention audit | `experiment/s5d2-dense-export-convention-audit` | `reports/S5D2_dense_export_convention_audit.md` | latest branch-specific audit of dense export scope and convention |
| JRT1 closeout | `experiment/jrt1-joint-rtdir-coupled-refiner` | `reports/JRT1_final_closeout_summary.md` | negative experiment conclusion belongs to its own branch |
| MF1 closeout | `experiment/mf1-multi-frame-chain-refiner` | `reports/MF1_final_closeout_summary.md` | negative experiment conclusion belongs to its own branch |

## Recommended Next Step

`REPORT3_curated_report_archive_plan`, but not migration yet unless explicitly approved.

## No-Op Confirmation

- no files moved = true
- no files deleted = true
- no branches merged = true
- no cherry-picks = true

