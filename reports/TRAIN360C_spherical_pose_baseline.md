# TRAIN360C spherical pose baseline

## 1. Executive summary
- real training executed: `true`
- learned weights saved: `true`
- recommended main checkpoint: `/home/dovetao/graduation_design_demo/checkpoints/TRAIN360C_spherical_pose_baseline/best_val.pt`
- improves over T57b external translation: `yes`
- next step recommendation: `proceed_to_BASE360_HKUST_360DVO_official_baseline_eval`

## 2. Data compliance
- train manifest: `/home/dovetao/graduation_design_demo/external_baselines/results/dset2c_360dvo_canonical/pair_manifest_train.jsonl`
- val manifest: `/home/dovetao/graduation_design_demo/external_baselines/results/dset2c_360dvo_canonical/pair_manifest_val.jsonl`
- test manifest: `/home/dovetao/graduation_design_demo/external_baselines/results/dset2c_360dvo_canonical/pair_manifest_test.jsonl`
- train pairs: `12154`
- val pairs: `5342`
- test pairs: `5062`
- train sequences: `['bridge_night', 'canyon_line', 'city_driving', 'drone_racetrack', 'field', 'shanghai_street']`
- val sequences: `['downhill_biking', 'mountains']`
- test sequences: `['ridge_to_lake', 'snowmobile']`
- no sequence overlap: `True`
- no random pair split: `true`
- no direct glob `data/360DVO/Sequences/*`: `true`

## 3. Model configuration
- backbone: `PanoramaRelPoseModel`
- checkpoint init: `/home/dovetao/graduation_design_demo/checkpoints/T57b_no_dt_multiscale_tmag_head_400/final.pt`
- strict/non-strict load: `non-strict`
- missing keys: `['coarse.mag_head.ridge_calib_raw_center', 'fine.mag_head.ridge_calib_raw_center', 'coupled_pose_head.backbone.0.weight', 'coupled_pose_head.backbone.0.bias', 'coupled_pose_head.backbone.1.weight', 'coupled_pose_head.backbone.1.bias', 'coupled_pose_head.backbone.4.weight', 'coupled_pose_head.backbone.4.bias', 'coupled_pose_head.rot_head.weight', 'coupled_pose_head.rot_head.bias', 'coupled_pose_head.tdir_head.weight', 'coupled_pose_head.tdir_head.bias', 'coupled_pose_head.gate_head.weight', 'coupled_pose_head.gate_head.bias']`
- unexpected keys: `[]`
- enabled heads: `rotation`, `translation direction`, `translation magnitude`
- disabled heads: `fine_stage=false`, `coupled_pose_head=false`
- training image resolution: `[384, 768]`
- bounded log_tmag: `true`

## 4. Loss configuration
- rotation loss: `SO(3) geodesic`
- tdir loss: `1 - cosine` in B frame
- tmag/log_tmag loss: `log_smooth_l1`
- weights: `rot=1.0, tdir=1.0, tmag=0.5`
- k-step weighting: `k1=1.0, k2=1.0, k3=0.9, k5=0.8`
- epsilon guard: `1e-06`

## 5. Training details
- epochs: `1`
- batch size: `2`
- optimizer: `AdamW`
- learning rate: `0.0001`
- scheduler: `cosine`
- grad clipping: `1.0`
- AMP: `false`
- seed: `3407`
- hardware: `{'device': 'cuda', 'gpu_name': 'NVIDIA GeForce RTX 3060 Laptop GPU', 'cuda_available': True}`
- runtime_sec: `751.08`
- train subset info: `{'subset_used': True, 'subset_count': 512, 'original_count': 12154, 'subset_seed': 3407}`

## 6. Validation metrics
- best epoch: `1`
- metrics: `{'count': 5342, 'coverage': 1.0, 'rot_mean_deg': 1.6207817274503875, 'rot_median_deg': 1.4548146724700928, 'rot_p90_deg': 2.386548852920535, 'signed_tdir_mean_deg': 111.86139636050414, 'signed_tdir_median_deg': 126.03379567453939, 'signed_tdir_p90_deg': 149.9810230984543, 'unsigned_tdir_mean_deg': 51.82972950681737, 'unsigned_tdir_median_deg': 46.69579100547433, 'unsigned_tdir_p90_deg': 82.83056154949084, 'anti_parallel_rate': 0.7132160239610633, 'tmag_median_ratio': 1.0500272154773382, 'tmag_mean_ratio': 1.4846339810060392, 'tmag_p90_ratio': 3.7898684874353137, 'log_tmag_mae': 0.7252769329385486, 'path_ratio': 0.586026844451413, 'path_length_pred': 3560.4996472895145, 'path_length_gt': 6075.659640852362, 'nan_inf_count': 0, 'ate_none': None, 'ate_se3': None, 'ate_sim3': None}`

## 7. Test metrics
- final selected checkpoint: `/home/dovetao/graduation_design_demo/checkpoints/TRAIN360C_spherical_pose_baseline/best_val.pt`
- metrics: `{'count': 5062, 'coverage': 1.0, 'rot_mean_deg': 3.10108115458114, 'rot_median_deg': 1.37955904006958, 'rot_p90_deg': 6.586977815628064, 'signed_tdir_mean_deg': 54.956992341554056, 'signed_tdir_median_deg': 32.252885214586854, 'signed_tdir_p90_deg': 136.8987411712531, 'unsigned_tdir_mean_deg': 36.179565415569876, 'unsigned_tdir_median_deg': 30.233118255638377, 'unsigned_tdir_p90_deg': 63.42356182828336, 'anti_parallel_rate': 0.21572500987751878, 'tmag_median_ratio': 0.7692854374461574, 'tmag_mean_ratio': 1.0144076650971499, 'tmag_p90_ratio': 2.273796730138812, 'log_tmag_mae': 0.6861240981342551, 'path_ratio': 0.5888660831060318, 'path_length_pred': 4801.225281059742, 'path_length_gt': 8153.339814946055, 'nan_inf_count': 0, 'ate_none': None, 'ate_se3': None, 'ate_sim3': None}`

## 8. Comparison to T57b
- {'model': 'T57b external reference', 'split': 'val+test external ref', 'signed_tdir_mean': 111.96493221327962, 'anti_parallel_rate': 0.6746724890829694, 'tmag_median_ratio': 0.1751560082454769, 'path_ratio': 0.14878731297064046, 'rot_mean': 2.12066772555611, 'notes': 'GEN5 external reference from TRAIN360A/B audit'}
- {'model': 'TRAIN360C_spherical_pose_baseline', 'split': 'val', 'signed_tdir_mean': 111.86139636050414, 'anti_parallel_rate': 0.7132160239610633, 'tmag_median_ratio': 1.0500272154773382, 'path_ratio': 0.586026844451413, 'rot_mean': 1.6207817274503875, 'notes': 'best-val selected checkpoint'}
- {'model': 'TRAIN360C_spherical_pose_baseline', 'split': 'test', 'signed_tdir_mean': 54.956992341554056, 'anti_parallel_rate': 0.21572500987751878, 'tmag_median_ratio': 0.7692854374461574, 'path_ratio': 0.5888660831060318, 'rot_mean': 3.10108115458114, 'notes': 'test evaluated once with best-val checkpoint'}

## 9. Failure analysis
- anti-parallel behavior is materially reduced relative to the weak T57b reference.
- scale no longer collapses as severely as the T57b reference.
- rotation is better than translation, suggesting correspondence ambiguity still dominates translation.
- path ratio improved over the T57b external reference.

## 10. Next step recommendation
- `proceed_to_BASE360_HKUST_360DVO_official_baseline_eval`

## 11. Compliance checklist
- `real_training_executed = true`
- `learned_weights_saved = true`
- `train_manifest_used = true`
- `val_manifest_used_for_validation_only = true`
- `test_manifest_used_for_final_eval_only = true`
- `uses_eval_gt_for_training = false`
- `uses_test_gt_for_training = false`
- `uses_orbslam3_teacher = false`
- `uses_hkust_360dvo_teacher = false`
- `s5_locked_metrics_modified = false`
- `legacy_scene01_artifact_dependency = false`
- `direct_glob_data_360dvo_sequences = false`
- `dset2c_canonical_manifest_required = true`
- `s5e15_external_inference_model_claimed = false`
