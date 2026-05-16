# FINAL360K_conservative_observability_weighting

## 1. Executive summary
- training executed: `true`
- init checkpoint: `/home/dovetao/graduation_design_demo/checkpoints/FINAL360I_struct360b_final/seed0/best_val.pt`
- selected checkpoint: `/home/dovetao/graduation_design_demo/checkpoints/FINAL360K_conservative_observability_weighting/best_val.pt`
- classification: `regression`
- final recommendation: `keep_FINAL360I_as_main_and_archive_FINAL360K`

## 2. DATA360A motivation
- DATA360A showed that FINAL360I still has a pair-level tdir bottleneck.
- small tmag and large rotation are difficult regimes.
- SEQ360B mainly improves scale/path rather than tdir or anti-parallel.

## 3. FINAL360J negative lesson
- FINAL360J used global tdir upweight plus anti-parallel penalty and regressed.
- FINAL360K therefore avoids strong global tdir amplification.
- FINAL360K only reweights per-sample tdir loss conservatively.

## 4. Setup
- branch: `research/final360k-conservative-observability-weighting`
- config: `configs/final360k_conservative_observability_weighting.yaml`
- checkpoint init: `/home/dovetao/graduation_design_demo/checkpoints/FINAL360I_struct360b_final/seed0/best_val.pt`
- epochs: `3`
- seed: `0`
- architecture changed: `false`
- translation head changed: `false`

## 5. Weighting design
- threshold source: `train_split_quantiles`
- tmag cut values: `[0.2381341490489577, 0.4074078840785943, 0.7540582156494824, 1.5112110698965902]`
- rot buckets deg: `{'small_max': 15.0, 'medium_max': 45.0, 'large_max': 90.0}`
- tmag weight table: `{'very_small': 0.45, 'small': 0.6, 'medium': 1.0, 'large': 1.05, 'very_large': 0.95}`
- rot weight table: `{'small_rotation': 0.95, 'medium_rotation': 1.0, 'large_rotation': 1.1, 'extreme_rotation': 1.05}`
- clamp range: `[0.35, 1.2]`
- train bucket counts: `{'very_small': 1216, 'small': 1823, 'medium': 3038, 'large': 3038, 'very_large': 3039}`
- weight distribution: `{'mean': 0.8227758546127006, 'median': 0.9024999737739563, 'p10': 0.42750000953674316, 'p90': 0.9975000023841858, 'max': 1.047374963760376, 'tmag_bucket_counts': {'very_small': 102, 'medium': 210, 'small': 105, 'very_large': 162, 'large': 189}, 'rot_bucket_counts': {'small_rotation': 756, 'medium_rotation': 12}}`

## 6. Model selection protocol
- val used for selection, test not used for selection
- best epoch: `2`
- best val score: `170.1110495393671`

## 7. Validation results
- val metrics: `{'count': 5342, 'coverage': 1.0, 'rot_mean_deg': 1.2467787049875327, 'rot_median_deg': 1.0347647070884705, 'rot_p90_deg': 2.6100851297378567, 'signed_tdir_mean_deg': 108.2088296716403, 'signed_tdir_median_deg': 122.78853679454974, 'signed_tdir_p90_deg': 149.24986313449597, 'unsigned_tdir_mean_deg': 51.056882481793224, 'unsigned_tdir_median_deg': 48.301157175613994, 'unsigned_tdir_p90_deg': 80.3526766879212, 'anti_parallel_rate': 0.7029202545862973, 'tmag_ratio_p10': 0.2589250994393299, 'tmag_median_ratio': 0.9663240689208481, 'tmag_ratio_p50': 0.9663240689208481, 'tmag_mean_ratio': 1.4142654076898242, 'tmag_ratio_p90': 3.79619399504142, 'tmag_p90_ratio': 3.79619399504142, 'log_tmag_mae': 0.7408507615290457, 'scale_collapse_rate': 0.001497566454511419, 'scale_explosion_rate': 0.0, 'path_ratio': 0.5416282376328893, 'path_length_pred': 3290.7488237321377, 'path_length_gt': 6075.659640852362, 'nan_inf_count': 0, 'nan_count': 0, 'inf_count': 0}`

## 8. Test results
- FINAL360I test metrics: `{'count': 5062, 'coverage': 1.0, 'rot_mean_deg': 2.330108616583517, 'rot_median_deg': 0.20558422803878784, 'rot_p90_deg': 6.713612318038943, 'signed_tdir_mean_deg': 45.264702006380205, 'signed_tdir_median_deg': 22.704615219747595, 'signed_tdir_p90_deg': 140.27662565571296, 'unsigned_tdir_mean_deg': 25.937378929700255, 'unsigned_tdir_median_deg': 19.432276054381838, 'unsigned_tdir_p90_deg': 58.16424864012134, 'anti_parallel_rate': 0.20169893322797314, 'tmag_ratio_p10': 0.2966194117283973, 'tmag_median_ratio': 0.8344251368086006, 'tmag_ratio_p50': 0.8344251368086006, 'tmag_mean_ratio': 1.1036085340991515, 'tmag_ratio_p90': 2.4225009459219193, 'tmag_p90_ratio': 2.4225009459219193, 'log_tmag_mae': 0.6559796168700758, 'scale_collapse_rate': 0.0019755037534571317, 'scale_explosion_rate': 0.0, 'path_ratio': 0.6403519796204528, 'path_length_pred': 5221.007291018963, 'path_length_gt': 8153.339814946055, 'nan_inf_count': 0, 'nan_count': 0, 'inf_count': 0, 'ate_none': None, 'ate_se3': None, 'ate_sim3': None}`
- FINAL360J reference metrics: `{'count': 5062, 'coverage': 1.0, 'rot_mean_deg': 2.3749489163555255, 'rot_median_deg': 0.29341986775398254, 'rot_p90_deg': 6.734423923492441, 'signed_tdir_mean_deg': 46.236052001935235, 'signed_tdir_median_deg': 23.738133652515195, 'signed_tdir_p90_deg': 141.424592870657, 'unsigned_tdir_mean_deg': 26.86426735655429, 'unsigned_tdir_median_deg': 20.33691765201405, 'unsigned_tdir_p90_deg': 60.7425059654726, 'anti_parallel_rate': 0.2056499407348874, 'tmag_ratio_p10': 0.25264710580098654, 'tmag_median_ratio': 0.7251413215270102, 'tmag_ratio_p50': 0.7251413215270102, 'tmag_mean_ratio': 0.9552756907106262, 'tmag_ratio_p90': 2.1235026880614556, 'tmag_p90_ratio': 2.1235026880614556, 'log_tmag_mae': 0.6971472370983129, 'scale_collapse_rate': 0.013433425523508494, 'scale_explosion_rate': 0.0, 'path_ratio': 0.5542818158986929, 'path_length_pred': 4519.247998267412, 'path_length_gt': 8153.339814946055, 'nan_inf_count': 0, 'nan_count': 0, 'inf_count': 0}` from `git_show_remote_branch`
- FINAL360K test metrics: `{'count': 5062, 'coverage': 1.0, 'rot_mean_deg': 2.365085963529238, 'rot_median_deg': 0.2755363881587982, 'rot_p90_deg': 6.7278229236602956, 'signed_tdir_mean_deg': 46.427316444531634, 'signed_tdir_median_deg': 23.667925516404715, 'signed_tdir_p90_deg': 141.20827364151276, 'unsigned_tdir_mean_deg': 26.895435337713252, 'unsigned_tdir_median_deg': 20.493393031164537, 'unsigned_tdir_p90_deg': 60.906862564808094, 'anti_parallel_rate': 0.2070327933623074, 'tmag_ratio_p10': 0.27977618648522834, 'tmag_median_ratio': 0.7988435549316202, 'tmag_ratio_p50': 0.7988435549316202, 'tmag_mean_ratio': 1.0552585491934794, 'tmag_ratio_p90': 2.335529179579015, 'tmag_p90_ratio': 2.335529179579015, 'log_tmag_mae': 0.6693975793279173, 'scale_collapse_rate': 0.0061240616357171075, 'scale_explosion_rate': 0.0, 'path_ratio': 0.6122365134360579, 'path_length_pred': 4991.772341161966, 'path_length_gt': 8153.339814946055, 'nan_inf_count': 0, 'nan_count': 0, 'inf_count': 0}`
- compared to FINAL360I: `worse`
- compared to FINAL360J: `comparable`

## 9. Bucket diagnostic
- small tmag signed_tdir: FINAL360I `36.10397205250734` -> FINAL360K `36.70295390816469` ; FINAL360J `36.63745542777961`
- large rotation signed_tdir: FINAL360I `92.22654070199621` -> FINAL360K `91.93039624241678` ; FINAL360J `91.9892156214478`
- very_large tmag anti_parallel `0.2551342812006319` -> `0.27488151658767773`, small_rotation anti_parallel `0.19390862944162437` -> `0.1997969543147208` ; FINAL360J refs `0.27093206951026855` / `0.1985786802030457`
- bucket flags: `{'small_tmag_improved': False, 'large_rotation_improved': True, 'anti_parallel_reduced_in_target_buckets': False, 'scale_path_protected_vs_final360i': True, 'better_than_final360j_regression': False}`

## 10. Comparison to SEQ360B
- SEQ360B pair metrics: `{'count': 1269, 'coverage': 1.0, 'rot_mean_deg': 0.8862303512771328, 'signed_tdir_mean_deg': 45.32544604265589, 'anti_parallel_rate': 0.20015760441292357, 'tmag_median_ratio': 1.6978484631707425, 'path_ratio': 1.3504765883324317, 'path_length_pred': 1006.1652189642191, 'path_length_gt': 745.0445477226908, 'nan_inf_count': 0, 'note': 'SEQ360B pair metrics are adjacent trajectory rows with scale-smoothed tmag; R/tdir come from frozen FINAL360I.'}`
- FINAL360K should be interpreted as a pair-level tdir stabilization attempt rather than a scale/path correction variant.

## 11. Caveats
- resource-limited single seed: `True`
- no sequence-level training in FINAL360K
- no mature VO claim from this run

## 12. Next recommendation
- `keep_FINAL360I_as_main_and_archive_FINAL360K`

## 13. Compliance checklist
- `training_executed = true`
- `test_used_for_selection = false`
- `architecture_changed = false`
- `translation_head_changed = false`
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
