# MAINT8 Runtime Dependency Trace

## Workspace Snapshot

- audit date: `2026-05-14`
- branch: `maintenance/project-structure-inventory-and-reorg-plan`
- upstream: `(not yet set on local maintenance branch)`
- dirty tracked files: `0`
- untracked entries: `13`
- local-only artifact roots visible in status: `backup_patches/, checkpoints/FINAL360I_struct360b_final/, checkpoints/STRUCT360A_implicit_spherical_cross_attention/, checkpoints/STRUCT360B_match_free_coarse_to_fine/, checkpoints/STRUCT360C_rotation_aware_fine_refinement/, checkpoints/TRAIN360D_observability_kstep_scale/, checkpoints/TRAIN360H_sweep/, external_baselines/results/seq360a_sequence_consistency_trajectory/, external_baselines/results/seq360b_scale_smoothing_trajectory/, external_baselines/results/train360e_final360i_trajectory/, reports/MAINT8_disk_usage_depth2.txt, reports/MAINT8_file_tree_maxdepth3.txt, reports/STRUCT360C_code_preparation_no_training.md`
- disk free on repo filesystem: `6.9G` on `/` according to `df -h` at audit time

This report is import-aware rather than name-only. It traces the current mainline and challenger entrypoints to the local modules and paths they actually touch.

## `tools/final360i_retrain_and_select.py`

- directly imported local modules: `config, datasets.dset2c_manifest_dataset, models.struct360b_match_free_coarse_to_fine`
- likely runtime-used local files: `config.py, datasets/dset2c_manifest_dataset.py, models/struct360b_match_free_coarse_to_fine.py, model.py, interaction.py, pose_head.py, erp_sampling.py, healpix_utils.py, transformer_encoder.py`
- config files used: `configs/final360i_struct360b_final.yaml, configs/struct360b_match_free_coarse_to_fine.yaml`
- checkpoint paths referenced: `checkpoints/FINAL360I_struct360b_final/seed*/best_val.pt, checkpoints/FINAL360I_struct360b_final/seed*/final.pt`
- report paths written or referenced: `reports/FINAL360I_final_retrain_and_model_selection.md, reports/FINAL360I_model_selection_table.json, reports/FINAL360I_thesis_ready_result_paragraph.md`
- data/manifests read: `external_baselines/results/dset2c_360dvo_canonical/pair_manifest_train.jsonl, external_baselines/results/dset2c_360dvo_canonical/pair_manifest_val.jsonl, checkpoints/DSET2C_360DVO_dataset_hygiene.json`
- notes: Main FINAL360I selector remains rooted in manifest-native DSET2C loading, but still uses root-level Config and the STRUCT360B coarse-to-fine model stack.

## `tools/train360e_sequence_trajectory_export_and_ate_eval.py`

- directly imported local modules: `config, datasets.dset2c_manifest_dataset, models.struct360b_match_free_coarse_to_fine`
- likely runtime-used local files: `config.py, datasets/dset2c_manifest_dataset.py, models/struct360b_match_free_coarse_to_fine.py, model.py, interaction.py, pose_head.py, erp_sampling.py, healpix_utils.py, transformer_encoder.py`
- config files used: `configs/struct360b_match_free_coarse_to_fine.yaml`
- checkpoint paths referenced: `checkpoints/FINAL360I_struct360b_final/seed0/best_val.pt`
- report paths written or referenced: `reports/TRAIN360E_sequence_trajectory_export_and_ATE_eval.md, reports/TRAIN360E_metrics_val.json, reports/TRAIN360E_metrics_test.json`
- data/manifests read: `external_baselines/results/dset2c_360dvo_canonical/pair_manifest_val.jsonl, external_baselines/results/dset2c_360dvo_canonical/pair_manifest_test.jsonl, external_baselines/results/train360e_final360i_trajectory/**/trajectory_metrics.json`
- notes: Trajectory export is an evaluation-side consumer of the FINAL360I/STRUCT360B stack and writes local trajectory artifacts under external_baselines/results/.

## `tools/train_seq360a_sequence_consistency_scale_drift.py`

- directly imported local modules: `config, datasets.dset2c_manifest_dataset, datasets.dset2c_sequence_clip_dataset, models.struct360b_match_free_coarse_to_fine, pose_head`
- likely runtime-used local files: `config.py, datasets/dset2c_manifest_dataset.py, datasets/dset2c_sequence_clip_dataset.py, models/struct360b_match_free_coarse_to_fine.py, model.py, interaction.py, pose_head.py, erp_sampling.py, healpix_utils.py, transformer_encoder.py`
- config files used: `configs/seq360a_sequence_consistency_scale_drift.yaml`
- checkpoint paths referenced: `checkpoints/FINAL360I_struct360b_final/seed0/best_val.pt, checkpoints/SEQ360A_sequence_consistency_scale_drift/best_val.pt, checkpoints/SEQ360A_sequence_consistency_scale_drift/final.pt`
- report paths written or referenced: `reports/SEQ360A_sequence_consistency_scale_drift_stabilization.md, reports/SEQ360A_metrics_val.json, reports/SEQ360A_metrics_test.json`
- data/manifests read: `external_baselines/results/dset2c_360dvo_canonical/pair_manifest_train.jsonl, external_baselines/results/dset2c_360dvo_canonical/pair_manifest_val.jsonl, external_baselines/results/dset2c_360dvo_canonical/pair_manifest_test.jsonl, checkpoints/DSET2C_360DVO_dataset_hygiene.json`
- notes: SEQ360A is the clearest active user of the sequence clip dataset; it extends the same STRUCT360B/FINAL360I pair stack rather than replacing it.

## `tools/train_struct360c_rotation_aware_fine_refinement.py`

- directly imported local modules: `datasets.dset2c_manifest_dataset, models.struct360c_rotation_aware_fine_refinement, train360d_pose_losses`
- likely runtime-used local files: `datasets/dset2c_manifest_dataset.py, models/struct360c_rotation_aware_fine_refinement.py, models/struct360b_match_free_coarse_to_fine.py, config.py, model.py, interaction.py, pose_head.py, train360d_pose_losses.py, losses.py, erp_sampling.py, healpix_utils.py, transformer_encoder.py`
- config files used: `configs/struct360c_rotation_aware_fine_refinement.yaml`
- checkpoint paths referenced: `checkpoints/FINAL360I_struct360b_final/seed0/best_val.pt, checkpoints/STRUCT360C_rotation_aware_fine_refinement/best_val.pt, checkpoints/STRUCT360C_rotation_aware_fine_refinement/final.pt`
- report paths written or referenced: `reports/STRUCT360C_rotation_aware_fine_refinement.md, reports/STRUCT360C_metrics_val.json, reports/STRUCT360C_metrics_test.json, reports/STRUCT360C_eval_only_recovery.md`
- data/manifests read: `external_baselines/results/dset2c_360dvo_canonical/pair_manifest_train.jsonl, external_baselines/results/dset2c_360dvo_canonical/pair_manifest_val.jsonl, external_baselines/results/dset2c_360dvo_canonical/pair_manifest_test.jsonl, checkpoints/DSET2C_360DVO_dataset_hygiene.json`
- notes: STRUCT360C looks modern at the tool level, but the model is still anchored in the legacy root-level core stack through the imported STRUCT360B base class and model.py helpers.

## `tools/train_struct360b_match_free_coarse_to_fine.py`

- directly imported local modules: `config, datasets.dset2c_manifest_dataset, models.struct360b_match_free_coarse_to_fine, pose_head, train360d_pose_losses`
- likely runtime-used local files: `config.py, datasets/dset2c_manifest_dataset.py, models/struct360b_match_free_coarse_to_fine.py, model.py, interaction.py, pose_head.py, train360d_pose_losses.py, losses.py, erp_sampling.py, healpix_utils.py, transformer_encoder.py`
- config files used: `configs/struct360b_match_free_coarse_to_fine.yaml`
- checkpoint paths referenced: `checkpoints/STRUCT360B_match_free_coarse_to_fine/best_val.pt, checkpoints/STRUCT360B_match_free_coarse_to_fine/final.pt`
- report paths written or referenced: `reports/STRUCT360B_match_free_coarse_to_fine_pose_refinement.md, reports/STRUCT360B_metrics_val.json, reports/STRUCT360B_metrics_test.json`
- data/manifests read: `external_baselines/results/dset2c_360dvo_canonical/pair_manifest_train.jsonl, external_baselines/results/dset2c_360dvo_canonical/pair_manifest_val.jsonl, external_baselines/results/dset2c_360dvo_canonical/pair_manifest_test.jsonl, checkpoints/DSET2C_360DVO_dataset_hygiene.json`
- notes: STRUCT360B is the model-family root for the current pair-level mainline and still uses the same root-level support modules.

## `tools/train360b_forward_sanity.py`

- directly imported local modules: `config, datasets.dset2c_manifest_dataset, model`
- likely runtime-used local files: `config.py, datasets/dset2c_manifest_dataset.py, model.py, interaction.py, pose_head.py, erp_sampling.py, healpix_utils.py, transformer_encoder.py`
- config files used: `configs/train360_v0_manifest_sanity.yaml`
- checkpoint paths referenced: none directly referenced
- report paths written or referenced: `reports/TRAIN360B_forward_sanity.json, reports/TRAIN360B_manifest_native_dataloader_adapter_and_forward_sanity.md`
- data/manifests read: `external_baselines/results/dset2c_360dvo_canonical/pair_manifest_train.jsonl`
- notes: This is the most direct proof that root-level model.py is still actively imported by a current-era manifest-native sanity tool.

## `tools/train_seq360b_lightweight_scale_smoothing.py`

- status: `absent in this repo snapshot`
- notes: No matching tool file is present in this repo snapshot. Keep SEQ360B as an artifact family with reports/checkpoints/trajectory outputs, but mark the training entrypoint as absent.

## Direct Answers To The Audit Questions

- current mainline still imports root-level `config.py`: `yes`
- any current tool still imports root-level `model.py`: `yes`, confirmed by `tools/train360b_forward_sanity.py` and transitively through `models/struct360b_match_free_coarse_to_fine.py` and `models/struct360c_rotation_aware_fine_refinement.py`
- `interaction.py` still transitively runtime-used: `yes`
- `dataset_pano_only.py` still runtime-used by the current DSET2C manifest-native mainline: `no`
- `dataset_pano_only.py` still used elsewhere in legacy / archival tools: `yes`
- `datasets/dset2c_manifest_dataset.py` is on the active path: `yes`
- `datasets/dset2c_sequence_clip_dataset.py` is on the active path: `yes`, through `tools/train_seq360a_sequence_consistency_scale_drift.py`
- `tools/train_seq360b_lightweight_scale_smoothing.py` exists in this repo state: `no`

## Practical Interpretation

- The repo already has a modern manifest-native data path, but the runtime core is still anchored in root-level modules such as `config.py`, `model.py`, and `interaction.py`.
- A future reorganization should therefore treat those root-level modules as migration-sensitive, not as disposable legacy files.
- `dataset_pano_only.py` is legacy for the current thesis mainline, but it remains important for older tools and audits, so it should be archived carefully rather than deleted.
