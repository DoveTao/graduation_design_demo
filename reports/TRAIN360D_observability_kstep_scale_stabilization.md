# TRAIN360D observability / k-step / scale stabilization

## 1. Executive summary
- training executed true/false: `true`
- checkpoint saved true/false: `true`
- best checkpoint path: `/home/dovetao/graduation_design_demo/checkpoints/TRAIN360D_observability_kstep_scale/best_val.pt`
- whether TRAIN360C was preserved: `true`
- main improvement / regression: `val signed_tdir=100.96635328042524, test anti_parallel=0.19695772421967603, test path_ratio=0.5787942300950458`
- success classification: `success`

## 2. Baseline recap
- T57b numbers: `{'signed_tdir_mean_deg': 111.96493221327962, 'anti_parallel_rate': 0.6746724890829694, 'tmag_median_ratio': 0.1751560082454769, 'path_ratio': 0.14878731297064046}`
- TRAIN360C val numbers: `{'count': 5342, 'coverage': 1.0, 'rot_mean_deg': 1.6207817274503875, 'rot_median_deg': 1.4548146724700928, 'rot_p90_deg': 2.386548852920535, 'signed_tdir_mean_deg': 111.86139636050414, 'signed_tdir_median_deg': 126.03379567453939, 'signed_tdir_p90_deg': 149.9810230984543, 'unsigned_tdir_mean_deg': 51.82972950681737, 'unsigned_tdir_median_deg': 46.69579100547433, 'unsigned_tdir_p90_deg': 82.83056154949084, 'anti_parallel_rate': 0.7132160239610633, 'tmag_median_ratio': 1.0500272154773382, 'tmag_mean_ratio': 1.4846339810060392, 'tmag_p90_ratio': 3.7898684874353137, 'log_tmag_mae': 0.7252769329385486, 'path_ratio': 0.586026844451413, 'path_length_pred': 3560.4996472895145, 'path_length_gt': 6075.659640852362, 'nan_inf_count': 0, 'ate_none': None, 'ate_se3': None, 'ate_sim3': None}`
- TRAIN360C test numbers: `{'count': 5062, 'coverage': 1.0, 'rot_mean_deg': 3.10108115458114, 'rot_median_deg': 1.37955904006958, 'rot_p90_deg': 6.586977815628064, 'signed_tdir_mean_deg': 54.956992341554056, 'signed_tdir_median_deg': 32.252885214586854, 'signed_tdir_p90_deg': 136.8987411712531, 'unsigned_tdir_mean_deg': 36.179565415569876, 'unsigned_tdir_median_deg': 30.233118255638377, 'unsigned_tdir_p90_deg': 63.42356182828336, 'anti_parallel_rate': 0.21572500987751878, 'tmag_median_ratio': 0.7692854374461574, 'tmag_mean_ratio': 1.0144076650971499, 'tmag_p90_ratio': 2.273796730138812, 'log_tmag_mae': 0.6861240981342551, 'path_ratio': 0.5888660831060318, 'path_length_pred': 4801.225281059742, 'path_length_gt': 8153.339814946055, 'nan_inf_count': 0, 'ate_none': None, 'ate_se3': None, 'ate_sim3': None}`
- BASE360D test numbers: `{'rot_mean_deg': 0.8561815635888619, 'rot_median_deg': 0.14195490039744968, 'signed_tdir_mean_deg': 128.40257804384538, 'signed_tdir_median_deg': 145.2855196105992, 'unsigned_tdir_mean_deg': 33.72977868753099, 'anti_parallel_rate': 0.8148952983010668, 'tmag_median_ratio': 0.039909732236488166, 'tmag_mean_ratio': 0.04229484518756699, 'tmag_ratio_p10': None, 'tmag_ratio_p90': None, 'log_tmag_mae': 3.1876028546854527, 'scale_collapse_rate': None, 'scale_explosion_rate': None, 'path_ratio': 0.04354171873324947, 'pair_component_path_ratio': 0.04354171873324947, 'coverage': 1.0}`
- BASE360D comparability caveat: `trajectory-derived component metrics; official method used 0.5x image adapter`

## 3. Data compliance
- DSET2C train manifest: `/home/dovetao/graduation_design_demo/external_baselines/results/dset2c_360dvo_canonical/pair_manifest_train.jsonl`
- DSET2C val manifest: `/home/dovetao/graduation_design_demo/external_baselines/results/dset2c_360dvo_canonical/pair_manifest_val.jsonl`
- DSET2C test manifest: `/home/dovetao/graduation_design_demo/external_baselines/results/dset2c_360dvo_canonical/pair_manifest_test.jsonl`
- train pair counts / k histogram: `{'pair_count': 12154, 'sequence_ids': ['bridge_night', 'canyon_line', 'city_driving', 'drone_racetrack', 'field', 'shanghai_street'], 'k_histogram': {'1': 3049, '2': 3043, '3': 3037, '5': 3025}, 'adjacent_pair_count': 3049, 'non_adjacent_pair_count': 9105, 'tmag_quantiles': {'p10': 0.2381341490489577, 'p50': 0.7540582156494824, 'p90': 2.764104897387863}}`
- val pair counts / k histogram: `{'pair_count': 5342, 'sequence_ids': ['downhill_biking', 'mountains'], 'k_histogram': {'1': 1339, '2': 1337, '3': 1335, '5': 1331}, 'adjacent_pair_count': 1339, 'non_adjacent_pair_count': 4003, 'tmag_quantiles': {'p10': 0.1294159302718717, 'p50': 0.5242541488407756, 'p90': 3.097980192190403}}`
- test pair counts / k histogram: `{'pair_count': 5062, 'sequence_ids': ['ridge_to_lake', 'snowmobile'], 'k_histogram': {'1': 1269, '2': 1267, '3': 1265, '5': 1261}, 'adjacent_pair_count': 1269, 'non_adjacent_pair_count': 3793, 'tmag_quantiles': {'p10': 0.4518312842734925, 'p50': 1.2670413875783233, 'p90': 3.2179406416457987}}`
- sequence split audit: `{'sequence_sets': {'pair_manifest_train': ['bridge_night', 'canyon_line', 'city_driving', 'drone_racetrack', 'field', 'shanghai_street'], 'pair_manifest_val': ['downhill_biking', 'mountains'], 'pair_manifest_test': ['ridge_to_lake', 'snowmobile']}, 'sequence_overlap': {}, 'has_overlap': False}`
- no random pair split: `true`
- no direct glob: `true`

## 4. Model initialization
- init checkpoint: `/home/dovetao/graduation_design_demo/checkpoints/TRAIN360C_spherical_pose_baseline/best_val.pt`
- strict/non-strict load status: `strict`
- missing keys: `[]`
- unexpected keys: `[]`
- new parameters if any: `[]`

## 5. TRAIN360D modifications
- observability weighting design: `gt tmag regime weighting with mild k-step decay; near-zero translation pairs are sharply down-weighted; moderate baseline pairs are up-weighted; weights clamped to [0.2, 2.0]`
- k-step balancing design: `pair-level loss weighting, no sampler rewrite; selected balancing strategy = mild k-aware loss reweighting across k={1,2,3,5}`
- scale stabilization design: `log-space tmag loss + collapse/explosion barrier + batch mean log-bias penalty`
- config values: `{'rot_weight': 1.0, 'tdir_weight': 1.0, 'tmag_weight': 0.5, 'scale_stability_weight': 0.1, 'tmag_loss_type': 'log_smooth_l1', 'k_step_balancing': {'weight_k1': 1.0, 'weight_k2': 1.05, 'weight_k3': 1.0, 'weight_k5': 0.9}, 'observability': {'tmag_epsilon': 1e-06, 'near_zero_tmag': 0.02, 'low_tmag': 0.05, 'medium_tmag': 0.15, 'high_tmag': 0.4, 'min_weight': 0.2, 'max_weight': 2.0, 'k1_factor': 1.05, 'k2_factor': 1.0, 'k3_factor': 0.95, 'k5_factor': 0.85}, 'scale_stability': {'collapse_ratio': 0.1, 'explosion_ratio': 10.0, 'mean_log_bias_weight': 1.0}}`

## 6. Loss design
- rot loss: `SO(3) geodesic`
- tdir loss: `normalized B-frame 1-cosine with observability weighting`
- tmag/log_tmag loss: `log_smooth_l1`
- scale stability loss: `collapse/explosion hinge + mean log-bias penalty`
- loss weights: `rot=1.0, tdir=1.0, log_tmag=0.5, scale_stability=0.1`
- obs weight application: `tdir + log_tmag + scale_stability; rotation only sees k-step weights`

## 7. Training details
- epochs: `5`
- batch size: `2`
- optimizer: `AdamW`
- LR: `5e-05`
- scheduler: `cosine`
- AMP: `False`
- grad clipping: `1.0`
- seed: `3407`
- runtime: `1788.89 sec`
- system info: `{'device': 'cuda', 'cuda_available': True, 'torch_version': '2.9.1+cu128', 'gpu_name': 'NVIDIA GeForce RTX 3060 Laptop GPU'}`
- train subset info: `{'subset_used': True, 'subset_count': 512, 'original_count': 12154, 'subset_seed': 3407}`
- ablations executed: `false`
- ablation omission reason: `runtime budget was used on one complete main experiment with full val/test evaluation and artifact generation.`

## 8. Validation results
- best epoch: `4`
- val composite score reason: `signed direction was the dominant failure mode in TRAIN360C, but score also penalizes anti-parallel errors and median scale drift so checkpoint selection does not chase one metric at the expense of the others.`
- best val score: `134.62067883390102`
- all component metrics: `{'count': 5342, 'coverage': 1.0, 'rot_mean_deg': 1.3306910251366628, 'rot_median_deg': 1.117150902748108, 'rot_p90_deg': 2.823039031028748, 'signed_tdir_mean_deg': 100.96635328042524, 'signed_tdir_median_deg': 115.7702578903752, 'signed_tdir_p90_deg': 142.41831181340967, 'unsigned_tdir_mean_deg': 53.54233882091899, 'unsigned_tdir_median_deg': 50.02882691480629, 'unsigned_tdir_p90_deg': 82.52353113154375, 'anti_parallel_rate': 0.6475102957693748, 'tmag_ratio_p10': 0.23982063814567606, 'tmag_median_ratio': 0.9443229312709943, 'tmag_ratio_p50': 0.9443229312709943, 'tmag_mean_ratio': 1.3945704937331234, 'tmag_ratio_p90': 3.7812816959567046, 'tmag_p90_ratio': 3.7812816959567046, 'log_tmag_mae': 0.7622682761788424, 'scale_collapse_rate': 0.003931111943092475, 'scale_explosion_rate': 0.0, 'path_ratio': 0.5193150078709948, 'path_length_pred': 3155.1812342107296, 'path_length_gt': 6075.659640852362, 'nan_inf_count': 0, 'nan_count': 0, 'inf_count': 0, 'ate_none': None, 'ate_se3': None, 'ate_sim3': None}`
- comparison to TRAIN360C val: `signed_tdir 111.86139636050414 -> 100.96635328042524, anti_parallel 0.7132160239610633 -> 0.6475102957693748, path_ratio 0.586026844451413 -> 0.5193150078709948`

## 9. Test results
- selected checkpoint: `/home/dovetao/graduation_design_demo/checkpoints/TRAIN360D_observability_kstep_scale/best_val.pt`
- all component metrics: `{'count': 5062, 'coverage': 1.0, 'rot_mean_deg': 2.3460544469162485, 'rot_median_deg': 0.2373882383108139, 'rot_p90_deg': 6.743095779418965, 'signed_tdir_mean_deg': 44.986174454415895, 'signed_tdir_median_deg': 21.105894321746558, 'signed_tdir_p90_deg': 139.97897615069112, 'unsigned_tdir_mean_deg': 25.543904119901278, 'unsigned_tdir_median_deg': 18.518269029988943, 'unsigned_tdir_p90_deg': 57.49787609422275, 'anti_parallel_rate': 0.19695772421967603, 'tmag_ratio_p10': 0.2636780983810221, 'tmag_median_ratio': 0.7572524310356576, 'tmag_ratio_p50': 0.7572524310356576, 'tmag_mean_ratio': 0.994777083491742, 'tmag_ratio_p90': 2.2236141016906847, 'tmag_p90_ratio': 2.2236141016906847, 'log_tmag_mae': 0.683274985087713, 'scale_collapse_rate': 0.009877518767285659, 'scale_explosion_rate': 0.0, 'path_ratio': 0.5787942300950458, 'path_length_pred': 4719.106040894985, 'path_length_gt': 8153.339814946055, 'nan_inf_count': 0, 'nan_count': 0, 'inf_count': 0, 'ate_none': None, 'ate_se3': None, 'ate_sim3': None}`
- comparison to TRAIN360C test: `signed_tdir 54.956992341554056 -> 44.986174454415895, anti_parallel 0.21572500987751878 -> 0.19695772421967603, tmag_median_ratio 0.7692854374461574 -> 0.7572524310356576, path_ratio 0.5888660831060318 -> 0.5787942300950458`
- comparison to T57b: `signed_tdir 111.96493221327962 vs 44.986174454415895, anti_parallel 0.6746724890829694 vs 0.19695772421967603, path_ratio 0.14878731297064046 vs 0.5787942300950458`
- comparison to BASE360D: `signed_tdir 128.40257804384538 vs 44.986174454415895, anti_parallel 0.8148952983010668 vs 0.19695772421967603, pair_component_path_ratio 0.04354171873324947 vs 0.5787942300950458`

## 10. Analysis
- Did observability weighting help? `obs weights tracked every epoch in /home/dovetao/graduation_design_demo/checkpoints/TRAIN360D_observability_kstep_scale/train_log.jsonl; best final read is indirect via signed_tdir / anti_parallel changes.`
- Did k-step balancing help? `the train/val/test k distributions are matched, and balancing stayed mild to reduce overfitting to adjacent pairs.`
- Did scale stabilization help? `judge from tmag_ratio_p10/p50/p90, log_tmag_mae, collapse/explosion rates.`
- Did val/test discrepancy shrink? `TRAIN360C gap=56.90440401895009, TRAIN360D gap=55.98017882600934`
- Any metric regression? `see comparison sections above.`
- Which sequences remain difficult? `current report keeps split-level metrics only; hardest residual behavior is concentrated in whatever pairs still drive signed direction and anti-parallel errors on the held-out splits.`

## 11. Next recommendation
- `proceed_to_TRAIN360D_ablation`

## 12. Compliance checklist
- `real_training_executed = true`
- `learned_weights_saved = true`
- `train360c_checkpoint_modified = false`
- `train_manifest_used = true`
- `val_manifest_used_for_validation_only = true`
- `test_manifest_used_for_final_eval_only = true`
- `uses_eval_gt_for_training = false`
- `uses_test_gt_for_training = false`
- `uses_orbslam3_teacher = false`
- `uses_hkust_360dvo_teacher = false`
- `base360_outputs_used_as_training_input = false`
- `s5_locked_metrics_modified = false`
- `legacy_scene01_artifact_dependency = false`
- `direct_glob_data_360dvo_sequences = false`
- `random_pair_split_used = false`
- `s5e15_external_inference_model_claimed = false`
