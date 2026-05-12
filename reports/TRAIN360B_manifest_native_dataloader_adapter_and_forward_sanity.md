# TRAIN360B manifest-native dataloader adapter and forward sanity

## 1. Executive summary
- manifest-native adapter implemented: `True`
- train/val/test no_grad forward sanity completed: `True`
- can enter `TRAIN360_spherical_pose_baseline`: `True`
- current largest blocker: Need manifest-native evaluator/training integration beyond forward sanity.
- This task only covers dataloader + forward sanity. No training loop, optimizer, finetune, or weight saving was executed.

## 2. Manifest schema audit
### train
- sample_count = `12154`
- sequence_ids = `['bridge_night', 'canyon_line', 'city_driving', 'drone_racetrack', 'field', 'shanghai_street']`
- schema_fields = `['R_BA', 'dataset', 'image_path_a', 'image_path_b', 'k', 'pair_index', 'pair_type', 'seq_id', 'split', 't_BA_B', 'tdir_B', 'timestamp_a', 'timestamp_b', 'tmag', 'valid_pose', 'valid_timestamp']`
- skip_reasons = `{}`
### val
- sample_count = `5342`
- sequence_ids = `['downhill_biking', 'mountains']`
- schema_fields = `['R_BA', 'dataset', 'image_path_a', 'image_path_b', 'k', 'pair_index', 'pair_type', 'seq_id', 'split', 't_BA_B', 'tdir_B', 'timestamp_a', 'timestamp_b', 'tmag', 'valid_pose', 'valid_timestamp']`
- skip_reasons = `{}`
### test
- sample_count = `5062`
- sequence_ids = `['ridge_to_lake', 'snowmobile']`
- schema_fields = `['R_BA', 'dataset', 'image_path_a', 'image_path_b', 'k', 'pair_index', 'pair_type', 'seq_id', 'split', 't_BA_B', 'tdir_B', 'timestamp_a', 'timestamp_b', 'tmag', 'valid_pose', 'valid_timestamp']`
- skip_reasons = `{}`
- sequence_overlap = `{}`
- hygiene_consistency = `{'train_row_count_matches_hygiene': True, 'val_row_count_matches_hygiene': True, 'test_row_count_matches_hygiene': True, 'sequence_overlap_present': False, 'hygiene_train360_ready_after_hygiene': True}`

## 3. Dataset adapter design
- dataset class path: `datasets/dset2c_manifest_dataset.py`
- sample dict returns manifest-native tensors plus T57b-compatible aliases `IA` / `IB` / `R_gt` / `t_gt_vec` / `t_gt_dir` / `t_gt_mag`.
- image loading: RGB, float32, [0,1], bilinear resize to checkpoint `H/W`.
- pose loading: uses manifest `R_BA`, `t_BA_B`, `tdir_B`, `tmag`; computes guarded `log_tmag`.
- failure handling: invalid rows can be filtered at dataset init; runtime image failures raise explicit errors.

## 4. Model compatibility
- PanoramaRelPoseModel path: `/home/dovetao/graduation_design_demo/model.py`
- checkpoint load status: `non-strict`
- missing keys: `['coarse.mag_head.ridge_calib_raw_center', 'fine.mag_head.ridge_calib_raw_center', 'coupled_pose_head.backbone.0.weight', 'coupled_pose_head.backbone.0.bias', 'coupled_pose_head.backbone.1.weight', 'coupled_pose_head.backbone.1.bias', 'coupled_pose_head.backbone.4.weight', 'coupled_pose_head.backbone.4.bias', 'coupled_pose_head.rot_head.weight', 'coupled_pose_head.rot_head.bias', 'coupled_pose_head.tdir_head.weight', 'coupled_pose_head.tdir_head.bias', 'coupled_pose_head.gate_head.weight', 'coupled_pose_head.gate_head.bias']`
- unexpected keys: `[]`
- strict error: `RuntimeError: Error(s) in loading state_dict for PanoramaRelPoseModel:
	Missing key(s) in state_dict: "coarse.mag_head.ridge_calib_raw_center", "fine.mag_head.ridge_calib_raw_center", "coupled_pose_head.backbone.0.weight", "coupled_pose_head.backbone.0.bias", "coupled_pose_head.backbone.1.weight", "coupled_pose_head.backbone.1.bias", "coupled_pose_head.backbone.4.weight", "coupled_pose_head.backbone.4.bias", "coupled_pose_head.rot_head.weight", "coupled_pose_head.rot_head.bias", "coupled_pose_head.tdir_head.weight", "coupled_pose_head.tdir_head.bias", "coupled_pose_head.gate_head.weight", "coupled_pose_head.gate_head.bias". `
- bounded log_tmag enabled: `True`
- positive tmag enabled: `True`

## 5. Forward sanity results
### train
- input_image_shape = `[1, 3, 1024, 2048]`
- batch_size = `1`
- model_stage = `coarse_only`
- R_output_shape = `[1, 3, 3]`
- tdir_output_shape = `[1, 3]`
- tmag_output_shape = `[1]`
- log_tmag_output_shape = `[1]`
- tdir_norm_stats = `{'min': 1.0, 'median': 1.0, 'max': 1.0}`
- tmag_stats = `{'min': 0.12805239856243134, 'median': 0.12805239856243134, 'max': 0.12805239856243134}`
- nan_inf_count = `0`
### val
- input_image_shape = `[1, 3, 1024, 2048]`
- batch_size = `1`
- model_stage = `coarse_only`
- R_output_shape = `[1, 3, 3]`
- tdir_output_shape = `[1, 3]`
- tmag_output_shape = `[1]`
- log_tmag_output_shape = `[1]`
- tdir_norm_stats = `{'min': 1.0, 'median': 1.0, 'max': 1.0}`
- tmag_stats = `{'min': 0.1344945877790451, 'median': 0.1344945877790451, 'max': 0.1344945877790451}`
- nan_inf_count = `0`
### test
- input_image_shape = `[1, 3, 1024, 2048]`
- batch_size = `1`
- model_stage = `coarse_only`
- R_output_shape = `[1, 3, 3]`
- tdir_output_shape = `[1, 3]`
- tmag_output_shape = `[1]`
- log_tmag_output_shape = `[1]`
- tdir_norm_stats = `{'min': 1.0, 'median': 1.0, 'max': 1.0}`
- tmag_stats = `{'min': 0.11895821243524551, 'median': 0.11895821243524551, 'max': 0.11895821243524551}`
- nan_inf_count = `0`

## 6. TRAIN360-v0 readiness assessment
- dataloader ready: `True`
- model forward ready: `True`
- loss interface ready: `True`
- evaluator still missing: `True`
- training config still missing: `False`
- recommend entering formal baseline task: `True`

## 7. Deferred modules
- s5e12_observability_gate: `deferred_but_planned`
- rotation_compensation: `deferred`
- contiguous_kstep_composition: `deferred`
- struct_geometry_token_soft_correspondence: `deferred`
- arch360_enhancements: `deferred`

## 8. Compliance checklist
- `no_training_executed = true`
- `no_finetune_executed = true`
- `optimizer_created = false`
- `backward_called = false`
- `learned_weights_saved = false`
- `s5_locked_metrics_modified = false`
- `legacy_scene01_artifact_dependency = false`
- `direct_glob_data_360dvo_sequences = false`
- `dset2c_canonical_manifest_required = true`
- `uses_eval_gt_for_training = false`
- `uses_test_gt_for_training = false`
- `uses_orbslam3_teacher = false`
- `uses_hkust_360dvo_teacher = false`
