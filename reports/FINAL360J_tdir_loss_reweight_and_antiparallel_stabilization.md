# FINAL360J_tdir_loss_reweight_and_antiparallel_stabilization

## 1. Executive summary
- training executed: `true`
- init checkpoint: `/home/dovetao/graduation_design_demo/checkpoints/FINAL360I_struct360b_final/seed0/best_val.pt`
- selected checkpoint: `/home/dovetao/graduation_design_demo/checkpoints/FINAL360J_tdir_loss_reweight/best_val.pt`
- classification: `regression`
- final recommendation: `keep_FINAL360I_as_main_and_archive_FINAL360J`

## 2. DATA360A motivation
- DATA360A showed that SEQ360B mainly improves scale/path, not tdir.
- difficult regimes include small tmag and large rotation buckets.
- anti-parallel concentration justified adding a direct anti-parallel penalty.

## 3. Setup
- branch: `research/final360j-tdir-loss-reweight`
- config: `configs/final360j_tdir_loss_reweight.yaml`
- checkpoint init: `/home/dovetao/graduation_design_demo/checkpoints/FINAL360I_struct360b_final/seed0/best_val.pt`
- manifests: `external_baselines/results/dset2c_360dvo_canonical/pair_manifest_train.jsonl`, `external_baselines/results/dset2c_360dvo_canonical/pair_manifest_val.jsonl`, `external_baselines/results/dset2c_360dvo_canonical/pair_manifest_test.jsonl`
- epochs: `3`
- seed: `0`
- architecture changed: `false`

## 4. Loss design
- lambda_rot: `1.0`
- lambda_tdir: `1.8`
- lambda_tmag: `0.6`
- lambda_antiparallel: `0.3`
- anti-parallel penalty: `relu(margin - cos)^2`, margin=`0.0`

## 5. Model selection protocol
- val used for selection, test not used for selection
- best epoch: `2`
- best val score: `164.97172419955692`

## 6. Validation results
- val signed_tdir_mean: `107.55588077601128`
- val anti_parallel_rate: `0.7008611007113441`
- val rot_mean: `1.2369855547143207`
- val tmag_median_ratio: `0.8958737091333241`
- val path_ratio: `0.49638513656246225`

## 7. Test results
- FINAL360I vs FINAL360J: `worse`
- FINAL360I test metrics: `{'count': 5062, 'coverage': 1.0, 'rot_mean_deg': 2.330108616583517, 'rot_median_deg': 0.20558422803878784, 'rot_p90_deg': 6.713612318038943, 'signed_tdir_mean_deg': 45.264702006380205, 'signed_tdir_median_deg': 22.704615219747595, 'signed_tdir_p90_deg': 140.27662565571296, 'unsigned_tdir_mean_deg': 25.937378929700255, 'unsigned_tdir_median_deg': 19.432276054381838, 'unsigned_tdir_p90_deg': 58.16424864012134, 'anti_parallel_rate': 0.20169893322797314, 'tmag_ratio_p10': 0.2966194117283973, 'tmag_median_ratio': 0.8344251368086006, 'tmag_ratio_p50': 0.8344251368086006, 'tmag_mean_ratio': 1.1036085340991515, 'tmag_ratio_p90': 2.4225009459219193, 'tmag_p90_ratio': 2.4225009459219193, 'log_tmag_mae': 0.6559796168700758, 'scale_collapse_rate': 0.0019755037534571317, 'scale_explosion_rate': 0.0, 'path_ratio': 0.6403519796204528, 'path_length_pred': 5221.007291018963, 'path_length_gt': 8153.339814946055, 'nan_inf_count': 0, 'nan_count': 0, 'inf_count': 0, 'ate_none': None, 'ate_se3': None, 'ate_sim3': None}`
- FINAL360J test metrics: `{'count': 5062, 'coverage': 1.0, 'rot_mean_deg': 2.3749489163555255, 'rot_median_deg': 0.29341986775398254, 'rot_p90_deg': 6.734423923492441, 'signed_tdir_mean_deg': 46.236052001935235, 'signed_tdir_median_deg': 23.738133652515195, 'signed_tdir_p90_deg': 141.424592870657, 'unsigned_tdir_mean_deg': 26.86426735655429, 'unsigned_tdir_median_deg': 20.33691765201405, 'unsigned_tdir_p90_deg': 60.7425059654726, 'anti_parallel_rate': 0.2056499407348874, 'tmag_ratio_p10': 0.25264710580098654, 'tmag_median_ratio': 0.7251413215270102, 'tmag_ratio_p50': 0.7251413215270102, 'tmag_mean_ratio': 0.9552756907106262, 'tmag_ratio_p90': 2.1235026880614556, 'tmag_p90_ratio': 2.1235026880614556, 'log_tmag_mae': 0.6971472370983129, 'scale_collapse_rate': 0.013433425523508494, 'scale_explosion_rate': 0.0, 'path_ratio': 0.5542818158986929, 'path_length_pred': 4519.247998267412, 'path_length_gt': 8153.339814946055, 'nan_inf_count': 0, 'nan_count': 0, 'inf_count': 0}`

## 8. Bucket diagnostic
- small tmag signed_tdir: FINAL360I `36.10397205250734` -> FINAL360J `36.63745542777961`
- large rotation signed_tdir: FINAL360I `92.22654070199621` -> FINAL360J `91.9892156214478`
- very_large tmag anti_parallel `0.2551342812006319` -> `0.27093206951026855`, small_rotation anti_parallel `0.19390862944162437` -> `0.1985786802030457`
- small tmag improved: `False`
- large rotation improved: `True`
- anti-parallel reduced in target buckets: `False`

## 9. Comparison to SEQ360B
- SEQ360B pair metrics: `{'count': 1269, 'coverage': 1.0, 'rot_mean_deg': 0.8862303512771328, 'signed_tdir_mean_deg': 45.32544604265589, 'anti_parallel_rate': 0.20015760441292357, 'tmag_median_ratio': 1.6978484631707425, 'path_ratio': 1.3504765883324317, 'path_length_pred': 1006.1652189642191, 'path_length_gt': 745.0445477226908, 'nan_inf_count': 0, 'note': 'SEQ360B pair metrics are adjacent trajectory rows with scale-smoothed tmag; R/tdir come from frozen FINAL360I.'}`
- SEQ360B should still be interpreted as a scale/path variant, not a tdir fix.
- BASE360D component caveat: trajectory-derived metrics `{'metric_pair_count': 5062, 'total_pair_count': 5062, 'pair_coverage': 1.0, 'pose_coverage': 0.9196816208393632, 'manifest_pose_coverage': 1.0, 'gt_tum_pose_count': 1382, 'matched_pose_count': 1271, 'manifest_unique_ts_count': 719, 'matched_manifest_pose_count': 719, 'trajectory_path_ratio': 0.043618000510806804, 'pair_component_path_ratio': 0.04354171873324947, 'pred_pair_path_length': 355.01042904915164, 'gt_pair_path_length': 8153.339817016857, 'rot_mean_deg': 0.8561815635888619, 'rot_median_deg': 0.14195490039744968, 'signed_tdir_mean_deg': 128.40257804384538, 'signed_tdir_median_deg': 145.2855196105992, 'unsigned_tdir_mean_deg': 33.72977868753099, 'anti_parallel_rate': 0.8148952983010668, 'tmag_median_ratio': 0.039909732236488166, 'tmag_mean_ratio': 0.04229484518756699, 'log_tmag_mae': 3.1876028546854527, 'tdir_valid_pair_count': 5062, 'near_zero_gt_tmag_count': 0, 'nan_inf_count': 0}`

## 10. Caveats
- resource-limited single seed: `True`
- no sequence-level training in FINAL360J
- no mature VO claim from this run

## 11. Next recommendation
- `keep_FINAL360I_as_main_and_archive_FINAL360J`

## 12. Compliance checklist
- `training_executed = true`
- `test_used_for_selection = false`
- `architecture_changed = false`
- `explicit_matching_used = false`
- `ransac_used = false`
- `pnp_used = false`
- `ba_used = false`
- `hkust_teacher_used = false`
- `orbslam_teacher_used = false`
- `metrics_modified = false`
- `checkpoints_modified_original = false`
- `large_checkpoints_committed = false`
- `raw_data_committed = false`
