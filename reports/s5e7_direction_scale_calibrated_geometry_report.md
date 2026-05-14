# S5E7 direction-scale calibrated geometry report

## 执行摘要
S5E7 的目标是同时改进方向误差与尺度高分位误差。`final_classification = S5E7_GUARD_DEPENDENT`。

## training status
{'attempted': True, 'classification': 'S5E7_TRAINING_SMOKE_ONLY', 'config': 'configs/s5e7_direction_scale_calibrated_geometry.yaml', 'checkpoint_dir': 'checkpoints/S5E7_direction_scale_calibrated_geometry_candidate', 'uses_scene01_seq03_for_training': False, 'model_type': 'direction_scale_calibrated_geometry', 'train_magnitude_prior': {'count': 786, 'median': 0.011908447102386463, 'p90': 0.2523947547524407, 'p95': 0.3691697395709953, 'max': 0.7818686910217085, 'clip_lo': 0.0020031222709819612, 'clip_hi': 0.3691697395709953}, 'bucket_priors': {'small_motion': {'count': 197, 'median': 0.0029765752132260696, 'p90': 0.004115805934363924, 'p95': 0.004452786086953767, 'max': 0.004801041553661831, 'clip_lo': 0.0013608692870236614, 'clip_hi': 0.004452786086953767}, 'normal_motion': {'count': 392, 'median': 0.011908447102386463, 'p90': 0.02716551357981745, 'p95': 0.03155805742315243, 'max': 0.03816320217172562, 'clip_lo': 0.005123810287804876, 'clip_hi': 0.03155805742315243}, 'large_motion': {'count': 197, 'median': 0.2105054393596516, 'p90': 0.4370886400382462, 'p95': 0.5229383635602999, 'max': 0.7818686910217085, 'clip_lo': 0.047578320811733917, 'clip_hi': 0.5229383635602999}}, 'best_checkpoint': 'checkpoints/S5E7_direction_scale_calibrated_geometry_candidate/s5e7_direction_scale_best.pt', 'losses': {'so3_geodesic': True, 'signed_tdir': True, 'tdir_abs_aux': True, 'anti_parallel_penalty': True, 'scale_residual_smooth_l1': True, 'robust_tmag_log': True, 'tmag_p95_penalty': True, 'path_length_consistency': True, 'short_window_direction_consistency': True, 'smoothness': True}, 'notes': ['S5E7 以方向学习为主目标，尺度只学习 bounded log-scale residual。', 'guard 只作为 fallback，不应被解释为真实几何改善。', 'scene01/seq03 GT 未用于训练。']}

## architecture summary
S5E7 使用 direction-scale factorization：`pred_t = pred_dir_unit * pred_tmag`，其中 `pred_tmag = prior_tmag * exp(clamped_log_scale_delta)`。

## train magnitude prior
{'count': 786, 'median': 0.011908447102386463, 'p90': 0.2523947547524407, 'p95': 0.3691697395709953, 'max': 0.7818686910217085, 'clip_lo': 0.0020031222709819612, 'clip_hi': 0.3691697395709953}

## motion bucket audit
{'small_motion_count': 114, 'small_motion_fraction': 0.25165562913907286, 'small_motion_tdir_mean': 62.10567659359031, 'small_motion_tmag_p95': 23.970107657128626, 'small_motion_path_fraction': 0.3885490653380592, 'small_motion_amplification': True}

## raw prediction metrics
{'rot_mean_deg': 0.9134395040767528, 'rot_median_deg': 0.797895569319923, 'rot_p90_deg': 1.3218302317546538, 'tdir_mean_deg': 53.62212702240343, 'tdir_median_deg': 45.740217831169936, 'tdir_p90_deg': 99.89252754388716, 'tdir_abs_mean_deg': 48.649060747032244, 'tdir_abs_median_deg': 45.56136877956687, 'tdir_abs_p90_deg': 80.07196258057124, 'tdir_mean_cosine': 0.5346236097269021, 'anti_parallel_rate': 0.1545253863134658, 'severe_wrong_sign_rate': 0.024282560706401765, 'direction_abs_good_but_signed_bad_rate': 0.002207505518763797, 'tmag_median_ratio': 11.757181188423342, 'tmag_mean_ratio': 72.8050340911994, 'tmag_p90_ratio': 46.88244268239341, 'tmag_p95_ratio': 94.28468533280407, 'tmag_p99_ratio': 2400.3853580354053, 'tmag_max_ratio': 7032.979339182292, 'path_ratio': 6.690489734209743, 'top20_tmag_path_fraction': 0.7273926587159092, 'long_run_path_fraction': 0.6882022160309632}

## guarded prediction metrics
{'rot_mean_deg': 0.9134395040767528, 'rot_median_deg': 0.797895569319923, 'rot_p90_deg': 1.3218302317546538, 'tdir_mean_deg': 53.62212702240343, 'tdir_median_deg': 45.740217831169936, 'tdir_p90_deg': 99.89252754388714, 'tdir_abs_mean_deg': 48.649060747032244, 'tdir_abs_median_deg': 45.56136877956687, 'tdir_abs_p90_deg': 80.07196258057124, 'tdir_mean_cosine': 0.5346236097269021, 'anti_parallel_rate': 0.1545253863134658, 'severe_wrong_sign_rate': 0.024282560706401765, 'direction_abs_good_but_signed_bad_rate': 0.002207505518763797, 'tmag_median_ratio': 0.8315272040683075, 'tmag_mean_ratio': 4.629716064443876, 'tmag_p90_ratio': 13.432147863947785, 'tmag_p95_ratio': 16.590740550562295, 'tmag_p99_ratio': 25.26837223664378, 'tmag_max_ratio': 74.29763973087576, 'path_ratio': 0.3237847877095814, 'top20_tmag_path_fraction': 0.07529940377427248, 'long_run_path_fraction': 0.015089004833897902}

## component metrics
{'rot_mean_deg': 0.9134395040767528, 'rot_median_deg': 0.797895569319923, 'rot_p90_deg': 1.3218302317546538, 'tdir_mean_deg': 53.62212702240343, 'tdir_median_deg': 45.740217831169936, 'tdir_p90_deg': 99.89252754388714, 'tdir_abs_mean_deg': 48.649060747032244, 'tdir_abs_median_deg': 45.56136877956687, 'tdir_abs_p90_deg': 80.07196258057124, 'tdir_mean_cosine': 0.5346236097269021, 'anti_parallel_rate': 0.1545253863134658, 'severe_wrong_sign_rate': 0.024282560706401765, 'direction_abs_good_but_signed_bad_rate': 0.002207505518763797, 'tmag_median_ratio': 0.8315272040683075, 'tmag_mean_ratio': 4.629716064443876, 'tmag_p90_ratio': 13.432147863947785, 'tmag_p95_ratio': 16.590740550562295, 'tmag_p99_ratio': 25.26837223664378, 'tmag_max_ratio': 74.29763973087576, 'path_ratio': 0.3237847877095814, 'top20_tmag_path_fraction': 0.07529940377427248, 'long_run_path_fraction': 0.015089004833897902}

## external evaluator results
{'none': {'ate': 9.787743578157508, 'drift': 0.13164512659644492, 'path_ratio': 0.3237847876348187, 'status': 'ok', 'num_matched_poses': 454, 'tracking_success_rate': 1.0}, 'se3': {'ate': 3.7889242467975253, 'drift': 0.13150609572822933, 'path_ratio': 0.3237847876348187, 'status': 'ok', 'num_matched_poses': 454, 'tracking_success_rate': 1.0}, 'sim3': {'ate': 3.765359444110737, 'drift': 0.1308366294241329, 'path_ratio': 0.3237847876348187, 'status': 'ok', 'num_matched_poses': 454, 'tracking_success_rate': 1.0}}

## comparison vs S5E6/S5E5/S5E4/S5E3/S5E2
{'rot_preserved': True, 'signed_tdir_improved_vs_s5e6': False, 'tdir_abs_improved_vs_s5e6': False, 'anti_parallel_improved_vs_s5e6': False, 'tmag_p95_improved_vs_s5e6': True, 'path_ratio_improved_vs_s5e6': True, 'sim3_ate_improved_vs_s5e6': True, 'raw_vs_guarded': {'raw_vs_guarded_tdir_gap': 0.0, 'raw_vs_guarded_tdir_abs_gap': 0.0, 'raw_vs_guarded_tmag_p95_gap': 77.69394478224177, 'raw_vs_guarded_path_ratio_gap': 6.366704946500162}}

## comparison vs ORB-SLAM3
{'coverage': 'S5E7 454/454 vs ORB-SLAM3 273/454', 'rot_gap': 'unavailable', 'tdir_gap': 'unavailable', 'tmag_gap': 'unavailable', 'se3_ate_gap': 3.4803798361050435, 'sim3_ate_gap': 3.541067278124113, 'path_ratio': 'S5E7=0.3237847877095814; ORB-SLAM3=0.2998258665660257', 'summary': 'ORB-SLAM3 没有可比的 edge-level 指标存档，因此这里只比较轨迹层面的覆盖率与 ATE/path_ratio。'}

## blockers
['translation direction 仍未形成明确改善。', '主要改善来自 fallback guard，而不是 raw direction/scale 学习质量。']

## validation results
{'verify_final_candidate': 'PASS', 'project_health_check': 'PASS', 's6_eval_only': 'PASS', 'unittest': 'PASS'}

## caveats
- S5E7 是 experimental candidate。
- 不替代 official S5 locked result。
- guard-only improvement 必须单独标记，不能当作真实几何学习成功。
