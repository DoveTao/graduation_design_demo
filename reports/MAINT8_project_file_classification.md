# MAINT8 Project File Classification

## Workspace Snapshot

- branch: `maintenance/project-structure-inventory-and-reorg-plan`
- upstream: `(not yet set on local maintenance branch)`
- dirty tracked files: `0`
- untracked local-only entries: `13`
- untracked entries seen during audit: `backup_patches/, checkpoints/FINAL360I_struct360b_final/, checkpoints/STRUCT360A_implicit_spherical_cross_attention/, checkpoints/STRUCT360B_match_free_coarse_to_fine/, checkpoints/STRUCT360C_rotation_aware_fine_refinement/, checkpoints/TRAIN360D_observability_kstep_scale/, checkpoints/TRAIN360H_sweep/, external_baselines/results/seq360a_sequence_consistency_trajectory/, external_baselines/results/seq360b_scale_smoothing_trajectory/, external_baselines/results/train360e_final360i_trajectory/, reports/MAINT8_disk_usage_depth2.txt, reports/MAINT8_file_tree_maxdepth3.txt, reports/STRUCT360C_code_preparation_no_training.md`
- classification method: import-aware manual audit, not filename-only heuristics

## Current Mainline

The current thesis mainline is the FINAL360I pair-level selection and the TRAIN360E trajectory export path built on the DSET2C manifest-native loader and the STRUCT360B coarse-to-fine model family.

- `configs/final360i_struct360b_final.yaml`
- `configs/struct360b_match_free_coarse_to_fine.yaml`
- `datasets/dset2c_manifest_dataset.py`
- `models/struct360b_match_free_coarse_to_fine.py`
- `tools/final360i_retrain_and_select.py`
- `tools/train360e_sequence_trajectory_export_and_ate_eval.py`
- `tools/train_struct360b_match_free_coarse_to_fine.py`
- `reports/FINAL360I_final_retrain_and_model_selection.md`
- `reports/FINAL360I_metrics_val.json`
- `reports/FINAL360I_metrics_test.json`
- `reports/TRAIN360E_sequence_trajectory_export_and_ATE_eval.md`
- `reports/TRAIN360E_metrics_val.json`
- `reports/TRAIN360E_metrics_test.json`
- `reports/STRUCT360B_match_free_coarse_to_fine_pose_refinement.md`
- `reports/STRUCT360B_metrics_val.json`
- `reports/STRUCT360B_metrics_test.json`

## Active Challenger Models

These files belong to active challenger lines or near-mainline refinements that are still relevant to comparisons, ablations, or follow-up work.

- `configs/struct360a_implicit_spherical_cross_attention.yaml`
- `configs/struct360c_rotation_aware_fine_refinement.yaml`
- `configs/seq360a_sequence_consistency_scale_drift.yaml`
- `configs/train360d_observability_kstep_scale.yaml`
- `configs/sweeps/train360h_sweep_space.yaml`
- `datasets/dset2c_sequence_clip_dataset.py`
- `models/struct360a_implicit_spherical_attention.py`
- `models/struct360c_rotation_aware_fine_refinement.py`
- `tools/train_struct360a_implicit_spherical_attention.py`
- `tools/train_struct360c_rotation_aware_fine_refinement.py`
- `tools/train_seq360a_sequence_consistency_scale_drift.py`
- `tools/train360d_observability_kstep_scale.py`
- `tools/train360h_hparam_sweep.py`
- `tools/summarize_train360h_sweep.py`
- `reports/STRUCT360A_implicit_spherical_cross_attention.md`
- `reports/STRUCT360A_metrics_val.json`
- `reports/STRUCT360A_metrics_test.json`
- `reports/STRUCT360C_rotation_aware_fine_refinement.md`
- `reports/STRUCT360C_metrics_val.json`
- `reports/STRUCT360C_metrics_test.json`
- `reports/STRUCT360C_eval_only_recovery.md`
- `reports/SEQ360A_sequence_consistency_scale_drift_stabilization.md`
- `reports/SEQ360A_metrics_val.json`
- `reports/SEQ360A_metrics_test.json`
- `reports/TRAIN360D_observability_kstep_scale_stabilization.md`
- `reports/TRAIN360D_metrics_val.json`
- `reports/TRAIN360D_metrics_test.json`
- `reports/TRAIN360H_hyperparameter_sweep.md`
- `reports/TRAIN360H_best_config_val.json`
- `reports/TRAIN360H_best_config_test.json`

Ambiguous note: SEQ360B artifacts are still present as checkpoints, reports, and trajectory outputs, but no `tools/train_seq360b_lightweight_scale_smoothing.py` entrypoint exists in the current repo snapshot.

## Reusable Core Modules

- `erp_sampling.py`
- `healpix_utils.py`
- `transformer_encoder.py`
- `pose_head.py`
- `losses.py`
- `train360_pose_losses.py`
- `train360d_pose_losses.py`

## Legacy MVP Or Archival

- `train_mvp.py`
- `dataset_pano_only.py`
- `dataset_rflypano.py`
- `analyze_k_stats.py`
- `geometry_refine.py`
- `depth_branch.py`

Rationale: these files reflect the earlier raw-scan / MVP-era path or optional branches that are not on the active FINAL360I / DSET2C mainline.

## Legacy Name But Runtime Used

- `config.py`
- `model.py`
- `interaction.py`

Rationale: these root-level modules look like migration candidates, but they are still directly or transitively imported by active tools, so they cannot be treated as dead code.

## External Baseline

- `tools/run_base360_hkust_360dvo_official.py`
- `tools/base360d_component_metric_alignment.py`
- `tools/base360c_official_python`
- `tools/base360c_ittnotify_shim.c`
- `reports/BASE360_HKUST_360DVO_official_baseline_eval.md`
- `reports/BASE360D_component_metric_alignment.md`
- `reports/BASE360_metrics_val.json`
- `reports/BASE360_metrics_test.json`
- `reports/BASE360D_metrics_val.json`
- `reports/BASE360D_metrics_test.json`
- `external_baselines/results/base360_hkust_360dvo_official`

## Canonical Data And Manifests

- `external_baselines/results/dset2c_360dvo_canonical/pair_manifest_train.jsonl`
- `external_baselines/results/dset2c_360dvo_canonical/pair_manifest_val.jsonl`
- `external_baselines/results/dset2c_360dvo_canonical/pair_manifest_test.jsonl`
- `external_baselines/results/dset2c_360dvo_canonical/pair_manifest_all.jsonl`
- `external_baselines/results/dset2c_360dvo_canonical/canonical_manifest_summary.json`
- `external_baselines/results/dset2c_360dvo_canonical/excluded_sequences.json`
- `external_baselines/results/dset2c_360dvo_canonical/local_tree_audit.json`
- `checkpoints/DSET2C_360DVO_dataset_hygiene.json`
- `data/`

## Reports And Metrics

- `FINAL360I*`: `7` files
- `TRAIN360*`: `22` files
- `STRUCT360*`: `17` files
- `SEQ360*`: `6` files
- `BASE360*`: `10` files
- `MAINT*`: `26` files
- `RESULTS360*`: `3` files
- `DEV360*`: `0` files
- complete file list is stored in `reports/MAINT8_project_file_classification.json` under `reports_and_metrics`

## Local Only Artifacts

- `backup_patches/`
- `checkpoints/FINAL360I_struct360b_final/`
- `checkpoints/STRUCT360A_implicit_spherical_cross_attention/`
- `checkpoints/STRUCT360B_match_free_coarse_to_fine/`
- `checkpoints/STRUCT360C_rotation_aware_fine_refinement/`
- `checkpoints/TRAIN360D_observability_kstep_scale/`
- `checkpoints/TRAIN360H_sweep/`
- `external_baselines/results/train360e_final360i_trajectory/`
- `external_baselines/results/seq360a_sequence_consistency_trajectory/`
- `external_baselines/results/seq360b_scale_smoothing_trajectory/`
- `external_baselines/results/*/pred_tum.txt`
- `external_baselines/results/*/adjacent_pair_predictions.jsonl`
- `external_baselines/results/*/unmatched_or_skipped_pairs.jsonl`

## Safe To Ignore Candidates

- `__pycache__/`
- `logs/`
- `outputs/`
- `reports_curated/`
- `backup_patches/`

## Classification Summary

- `config.py`, `model.py`, and `interaction.py` are the most important migration-sensitive roots.
- `dataset_pano_only.py` is legacy for the active thesis line, but still relevant to older tooling and audits.
- `datasets/dset2c_manifest_dataset.py` and `datasets/dset2c_sequence_clip_dataset.py` cleanly separate current pair and sequence data access paths.
- `external_baselines/results/` contains a mixture of official baselines, canonical manifests, and project-generated trajectory exports, so it should be reorganized only by documentation first.
