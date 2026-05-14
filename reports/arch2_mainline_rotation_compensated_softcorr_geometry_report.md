# ARCH2 mainline rotation compensated softcorr geometry report

## 为什么从 S5E21 进入 ARCH2
S5E21 证明了 no-harm gate 可行，但 global 提升仍不足，因此 ARCH2 转向把 soft correspondence、rotation compensation、observability weighting、S5E15 scale guard 和 true k-step 统一进主干。

## ARCH2 主结构说明
ARCH2 以 spherical token / ERP 主线为前提，增加 rotation-compensated soft correspondence geometry 和保守的 scale guard。

## pose convention 整理
{'tdir_pred_frame': 'B', 'tdir_gt_frame': 'B', 'tdir_B_explicit': True, 'tdir_A_explicit': True, 'twc_tcw_mismatch_suspected': False, 'forbid_ambiguous_t_dir_out': True}

## softcorr geometry audit
{'W_ab_used_in_motion_token': True, 'W_ba_used_in_cycle_consistency': True, 'R_coarse_used_in_rotation_compensation': True, 'residual_flow_nonzero_mean': 0.08336428687733004, 'softcorr_entropy_mean': 2.109631528391217}

## training status
{'real_training_executed': True, 'optimizer_step_count': 120, 'learned_weights_saved': True, 'smoke_policy_only': False, 'uses_eval_gt_for_training': False, 'uses_eval_gt_for_gate': False, 'uses_orbslam3_teacher': False, 'uses_rotation_compensated_motion_tokens': True, 'uses_soft_correspondence_geometry': True, 'uses_observability_loss_weighting': True, 'uses_true_kstep': True, 'dt_factor_per_sample': True, 'classification': 'ARCH2_REAL_TRAINING_COMPLETE'}

## component metrics
{'rot_mean_deg': 0.9134370913408972, 'rot_median_deg': 0.7979626758525834, 'rot_p90_deg': 1.32180424399533, 'signed_tdir_mean_deg': 50.352550059420615, 'signed_tdir_median_deg': 41.229479433819755, 'signed_tdir_p90_deg': 101.80693648658053, 'tdir_abs_mean_deg': 43.15991862651722, 'tdir_abs_median_deg': 39.8747502828935, 'tdir_abs_p90_deg': 78.63250397972163, 'tdir_mean_cosine': 0.559105992955047, 'anti_parallel_rate': 0.1479028697571744, 'severe_wrong_sign_rate': 0.05518763796909492, 'direction_abs_good_but_signed_bad_rate': 0.024282560706401765, 'tmag_median_ratio': 1.171006033440564, 'tmag_p95_ratio': 4.309723516706491, 'path_ratio': 0.14508782048566882}

## external evaluator
{'none': {'ate': 9.364114790564496, 'drift': 0.12861851473170366, 'path_ratio': 0.14508783032428438}, 'se3': {'ate': 3.9871073257608174, 'drift': 0.12983505169265336, 'path_ratio': 0.14508783032428438}, 'sim3': {'ate': 3.9682545773053644, 'drift': 0.13012644224846695, 'path_ratio': 0.14508783032428438}}

## compare to S5E15/S5E20/S5E21/ORB-SLAM3
{'tdir_improved': True, 'anti_parallel_reduced': False, 'path_ratio_improved': False, 'sim3_ate_improved_or_not_worse': False, 'high_confidence_subset_improved': True, 'overall_improved': False}

## 是否 promote ARCH2
final_classification=ARCH2_SUBSET_IMPROVED_GLOBAL_NOT

## caveats
- ARCH2 是 experimental candidate。
- 不替代 official locked result。
- 不使用 eval GT calibration。
