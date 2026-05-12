# TRAIN360H hyperparameter sweep

## 1. Executive summary
- sweep executed true/false: `true`
- number of runs: `16`
- best run_id: `TRAIN360H_F01_TRAIN360H_S1_best_seed1`
- best checkpoint path: `/home/dovetao/graduation_design_demo/checkpoints/TRAIN360H_sweep/TRAIN360H_F01_TRAIN360H_S1_best_seed1/best_val.pt`
- whether it improves over TRAIN360D: `partial`
- whether it improves over TRAIN360C: `yes`
- success classification: `partial_success`

## 2. Baseline recap
- T57b: `{'signed_tdir_mean_deg': 111.96493221327962, 'anti_parallel_rate': 0.6746724890829694, 'tmag_median_ratio': 0.1751560082454769, 'path_ratio': 0.14878731297064046}`
- TRAIN360C: `{'count': 5062, 'coverage': 1.0, 'rot_mean_deg': 3.10108115458114, 'rot_median_deg': 1.37955904006958, 'rot_p90_deg': 6.586977815628064, 'signed_tdir_mean_deg': 54.956992341554056, 'signed_tdir_median_deg': 32.252885214586854, 'signed_tdir_p90_deg': 136.8987411712531, 'unsigned_tdir_mean_deg': 36.179565415569876, 'unsigned_tdir_median_deg': 30.233118255638377, 'unsigned_tdir_p90_deg': 63.42356182828336, 'anti_parallel_rate': 0.21572500987751878, 'tmag_median_ratio': 0.7692854374461574, 'tmag_mean_ratio': 1.0144076650971499, 'tmag_p90_ratio': 2.273796730138812, 'log_tmag_mae': 0.6861240981342551, 'path_ratio': 0.5888660831060318, 'path_length_pred': 4801.225281059742, 'path_length_gt': 8153.339814946055, 'nan_inf_count': 0, 'ate_none': None, 'ate_se3': None, 'ate_sim3': None}`
- TRAIN360D: `{'count': 5062, 'coverage': 1.0, 'rot_mean_deg': 2.3460544469162485, 'rot_median_deg': 0.2373882383108139, 'rot_p90_deg': 6.743095779418965, 'signed_tdir_mean_deg': 44.986174454415895, 'signed_tdir_median_deg': 21.105894321746558, 'signed_tdir_p90_deg': 139.97897615069112, 'unsigned_tdir_mean_deg': 25.543904119901278, 'unsigned_tdir_median_deg': 18.518269029988943, 'unsigned_tdir_p90_deg': 57.49787609422275, 'anti_parallel_rate': 0.19695772421967603, 'tmag_ratio_p10': 0.2636780983810221, 'tmag_median_ratio': 0.7572524310356576, 'tmag_ratio_p50': 0.7572524310356576, 'tmag_mean_ratio': 0.994777083491742, 'tmag_ratio_p90': 2.2236141016906847, 'tmag_p90_ratio': 2.2236141016906847, 'log_tmag_mae': 0.683274985087713, 'scale_collapse_rate': 0.009877518767285659, 'scale_explosion_rate': 0.0, 'path_ratio': 0.5787942300950458, 'path_length_pred': 4719.106040894985, 'path_length_gt': 8153.339814946055, 'nan_inf_count': 0, 'nan_count': 0, 'inf_count': 0, 'ate_none': None, 'ate_se3': None, 'ate_sim3': None}`
- BASE360D: `{'metric_pair_count': 5062, 'total_pair_count': 5062, 'pair_coverage': 1.0, 'pose_coverage': 0.9196816208393632, 'manifest_pose_coverage': 1.0, 'gt_tum_pose_count': 1382, 'matched_pose_count': 1271, 'manifest_unique_ts_count': 719, 'matched_manifest_pose_count': 719, 'trajectory_path_ratio': 0.043618000510806804, 'pair_component_path_ratio': 0.04354171873324947, 'pred_pair_path_length': 355.01042904915164, 'gt_pair_path_length': 8153.339817016857, 'rot_mean_deg': 0.8561815635888619, 'rot_median_deg': 0.14195490039744968, 'signed_tdir_mean_deg': 128.40257804384538, 'signed_tdir_median_deg': 145.2855196105992, 'unsigned_tdir_mean_deg': 33.72977868753099, 'anti_parallel_rate': 0.8148952983010668, 'tmag_median_ratio': 0.039909732236488166, 'tmag_mean_ratio': 0.04229484518756699, 'log_tmag_mae': 3.1876028546854527, 'tdir_valid_pair_count': 5062, 'near_zero_gt_tmag_count': 0, 'nan_inf_count': 0}`

## 3. Ablation findings
| run_id | val_score | val_signed_tdir | val_anti_parallel | val_tmag_ratio | val_path_ratio | test_signed_tdir | test_path_ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TRAIN360H_A4_obs_kstep_no_scale_seed0 | 109.62795545593225 | 45.357834520303136 | 0.0 | 2.0287990385136654 | 1.7461766782996997 | 55.77945159211351 | 0.7713805482615894 |
| TRAIN360H_A0_reproduce_D_seed0 | 109.78978151589976 | 44.956484798332696 | 0.0 | 2.0848518949947197 | 1.7942905706546781 | 55.46502585619085 | 0.775810812400399 |
| TRAIN360H_A1_obs_only_seed0 | 110.32036663006657 | 45.469595021448555 | 0.0 | 2.0886753945673613 | 1.797082306854836 | 55.96411464259915 | 0.7775654981873537 |
| TRAIN360H_A3_scale_only_seed0 | 112.04095744749083 | 45.187159926184115 | 0.0 | 2.306794527369367 | 1.9837194254554353 | 56.52648010781024 | 0.7942237998828358 |
- which component helped direction: `see lowest ablation val/test signed_tdir and anti_parallel rows.`
- which component helped scale: `see ablation rows with stronger tmag/path metrics.`
- which component hurt: `rows with clear val score regression or path collapse.`

## 4. Search space
- lr: `coarse and refined around 3e-5 / 5e-5 / 1e-4 behavior with local adjustments around the best coarse candidate`
- loss weights: `rot / tdir / log_tmag / scale_stability searched in coarse and refined stages`
- obs range: `min/max and moderate boost searched in coarse stage`
- k-step weighting: `explicit and adjacent-vs-nonadjacent decays searched in coarse stage`
- score weights: `selection_score varied across coarse candidates, final choice remained val-only`
- schedule: `cosine / step / none variants in coarse stage`
- seed strategy: `best refined config re-run on extra seeds before final test`

## 5. Leaderboard
- csv: `/home/dovetao/graduation_design_demo/reports/TRAIN360H_sweep_leaderboard.csv`
- json: `/home/dovetao/graduation_design_demo/reports/TRAIN360H_sweep_leaderboard.json`

## 6. Best config
- full hyperparams: `{'task_name': 'TRAIN360H_F01_TRAIN360H_S1_best_seed1', 'prechecks': {'expected_branch': 'experiment/train360d-observability-kstep-scale'}, 'inputs': {'hygiene_json': 'checkpoints/DSET2C_360DVO_dataset_hygiene.json', 'train_manifest': 'external_baselines/results/dset2c_360dvo_canonical/pair_manifest_train.jsonl', 'val_manifest': 'external_baselines/results/dset2c_360dvo_canonical/pair_manifest_val.jsonl', 'test_manifest': 'external_baselines/results/dset2c_360dvo_canonical/pair_manifest_test.jsonl', 'dataset_adapter': 'datasets/dset2c_manifest_dataset.py', 'init_checkpoint': 'checkpoints/TRAIN360C_spherical_pose_baseline/best_val.pt', 'train360c_best_checkpoint': 'checkpoints/TRAIN360C_spherical_pose_baseline/best_val.pt', 'train360c_final_checkpoint': 'checkpoints/TRAIN360C_spherical_pose_baseline/final.pt', 'train360c_val_metrics': 'reports/TRAIN360C_metrics_val.json', 'train360c_test_metrics': 'reports/TRAIN360C_metrics_test.json', 'base360d_val_metrics': 'reports/BASE360D_metrics_val.json', 'base360d_test_metrics': 'reports/BASE360D_metrics_test.json'}, 'outputs': {'checkpoint_dir': 'checkpoints/TRAIN360H_sweep/TRAIN360H_F01_TRAIN360H_S1_best_seed1', 'report_path': 'checkpoints/TRAIN360H_sweep/TRAIN360H_F01_TRAIN360H_S1_best_seed1/run_report.md', 'val_metrics_path': 'checkpoints/TRAIN360H_sweep/TRAIN360H_F01_TRAIN360H_S1_best_seed1/metrics_val.json', 'test_metrics_path': 'checkpoints/TRAIN360H_sweep/TRAIN360H_F01_TRAIN360H_S1_best_seed1/metrics_test.json', 'comparison_summary_path': 'checkpoints/TRAIN360H_sweep/TRAIN360H_F01_TRAIN360H_S1_best_seed1/comparison_summary.md', 'next_recommendation': 'proceed_to_TRAIN360I_final_retrain_with_best_hparams'}, 'data': {'image_hw': [384, 768], 'tmag_epsilon': '1e-06', 'train_batch_size': 2, 'eval_batch_size': 8, 'num_workers': 4, 'train_subset_max': 48, 'shuffle_train': True, 'require_paths': True, 'skip_invalid': True}, 'model': {'strict_load_attempt': True, 'enable_fine_stage': False, 'enable_coupled_pose_head': False, 'enable_depth_fusion': False, 'use_cuda_if_available': True}, 'loss': {'rot_weight': 1.0, 'tdir_weight': 2.0, 'tmag_weight': 0.75, 'scale_stability_weight': 0.05, 'tmag_loss_type': 'log_smooth_l1', 'k_step_balancing': {'weight_k1': 1.0, 'weight_k2': 1.05, 'weight_k3': 1.0, 'weight_k5': 0.9, 'mode': 'adjacent_vs_nonadjacent', 'adjacent_weight': 1.25, 'non_adjacent_weight': 1.0, 'decay': 'none', 'min_weight': 0.2}, 'observability': {'tmag_epsilon': '1e-06', 'near_zero_tmag': 0.005, 'low_tmag': 0.05, 'medium_tmag': 0.15, 'high_tmag': 0.4, 'min_weight': 0.5, 'max_weight': 1.5, 'k1_factor': 1.05, 'k2_factor': 1.0, 'k3_factor': 0.95, 'k5_factor': 0.85, 'moderate_baseline_boost': 1.25, 'medium_weight': 1.25, 'high_weight': 1.35}, 'scale_stability': {'collapse_ratio': 0.1, 'explosion_ratio': 10.0, 'mean_log_bias_weight': 1.0}, 'components': {}}, 'training': {'epochs': 3, 'optimizer': 'AdamW', 'lr': '3e-05', 'weight_decay': 0.0001, 'grad_clip_norm': 1.0, 'amp': False, 'seed': 1, 'scheduler': 'none', 'min_lr': '1e-06'}, 'evaluation': {'max_val_batches': None, 'max_test_batches': None, 'run_test': False, 'selection_score': {'anti_parallel_weight': 30.0, 'tmag_ratio_weight': 20.0, 'path_ratio_weight': 10.0, 'rot_weight': 0.0}}, 'metadata': {'phase': 'final_eval', 'run_id': 'TRAIN360H_F01_TRAIN360H_S1_best_seed1', 'selection_policy': 'final_candidate_eval_only'}}`
- reason selected: `best val score after staged val-only selection and seed robustness, before any final test comparison.`
- best val epoch: `3`
- final test metrics: `{'count': 5062, 'coverage': 1.0, 'rot_mean_deg': 3.1616807945222782, 'rot_median_deg': 1.4338136315345764, 'rot_p90_deg': 7.089140510559084, 'signed_tdir_mean_deg': 53.73375854133177, 'signed_tdir_median_deg': 32.94996166945526, 'signed_tdir_p90_deg': 138.49944561811802, 'unsigned_tdir_mean_deg': 34.57963931476379, 'unsigned_tdir_median_deg': 29.76936998575061, 'unsigned_tdir_p90_deg': 62.74972042000828, 'anti_parallel_rate': 0.20209403397866457, 'tmag_ratio_p10': 0.2765550670704756, 'tmag_median_ratio': 0.8503917514492008, 'tmag_ratio_p50': 0.8503917514492008, 'tmag_mean_ratio': 1.1252581257212217, 'tmag_ratio_p90': 2.5432877639863536, 'tmag_p90_ratio': 2.5432877639863536, 'log_tmag_mae': 0.6818262532323421, 'scale_collapse_rate': 0.014816278150928487, 'scale_explosion_rate': 0.0, 'path_ratio': 0.6493755813576354, 'path_length_pred': 5294.57978233695, 'path_length_gt': 8153.339814946055, 'nan_inf_count': 0, 'nan_count': 0, 'inf_count': 0, 'ate_none': None, 'ate_se3': None, 'ate_sim3': None}`

## 7. Comparison table
| model | signed_tdir_mean | anti_parallel_rate | tmag_median_ratio | path_ratio | coverage | notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| T57b | 111.96493221327962 | 0.6746724890829694 | 0.1751560082454769 | 0.14878731297064046 | n/a | legacy external |
| TRAIN360C | 54.956992341554056 | 0.21572500987751878 | 0.7692854374461574 | 0.5888660831060318 | 1.0 | locked baseline |
| TRAIN360D | 44.986174454415895 | 0.19695772421967603 | 0.7572524310356576 | 0.5787942300950458 | 1.0 | current best before sweep |
| TRAIN360H best | 53.73375854133177 | 0.20209403397866457 | 0.8503917514492008 | 0.6493755813576354 | 1.0 | staged sweep result |
| BASE360D | 128.40257804384538 | 0.8148952983010668 | 0.039909732236488166 | 0.04354171873324947 | 1.0 | trajectory-derived component metric |

## 8. Failure analysis
- runs that collapsed scale: `inspect leaderboard rows with low tmag_median_ratio or path_ratio.`
- runs that improved direction but hurt tmag: `inspect leaderboard rows with lower signed_tdir and weaker tmag/path.`
- runs that overfit val: `compare final_eval test metrics against low val_score candidates.`
- unstable settings: `none of the retained runs should report NaN/Inf; any skipped or failed runs would appear in checkpoint-local reports.`
- NaN/Inf if any: `see per-run metrics.`

## 9. Next recommendation
- `proceed_to_TRAIN360I_final_retrain_with_best_hparams`

## 10. Compliance checklist
- `real_training_executed = true`
- `learned_weights_saved = true`
- `train360c_checkpoint_modified = false`
- `train360d_checkpoint_modified = false`
- `test_used_for_hparam_selection = false`
- `train_manifest_used = true`
- `val_manifest_used_for_selection_only = true`
- `test_manifest_used_for_final_eval_only = true`
- `uses_eval_gt_for_training = false`
- `uses_test_gt_for_training = false`
- `uses_hkust_360dvo_teacher = false`
- `uses_orbslam3_teacher = false`
- `base360_outputs_used_as_training_input = false`
- `s5_locked_metrics_modified = false`
- `random_pair_split_used = false`
- `direct_glob_data_360dvo_sequences = false`
- `large_checkpoints_committed_to_git = false`
