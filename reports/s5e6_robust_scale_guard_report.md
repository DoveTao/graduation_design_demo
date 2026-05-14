# S5E6 robust scale guard report

## 执行摘要
S5E6 使用 train-split magnitude prior、alpha 校正和 p95 clamp，目标是修复 S5E5 的 path explosion，同时尽量保持 S5E4 的 signed direction。`final_classification = S5E6_GEOMETRY_IMPROVED`。

## S5E5 path explosion failure audit
{'experiment': 'S5E6_robust_scale_guard_and_direction_metric_gate_candidate', 'top10_tmag_path_fraction': 0.6566711958597958, 'top20_tmag_path_fraction': 0.7207316749796695, 'top50_tmag_path_fraction': 0.7602983486767192, 'tmag_ratio_p90': 48.023883417026354, 'tmag_ratio_p95': 95.44378091660477, 'tmag_ratio_p99': 2338.4565043777293, 'tmag_ratio_max': 6869.16668913065, 'corr_tmag_gt_step': -0.06865734800555673, 'corr_tmag_signed_tdir': -0.10804135675993931, 'corr_tmag_tdir_abs': -0.12180872265214915, 'longest_run_over_p95': 13, 'long_run_path_fraction': 0.6817449643122911, 'small_motion_high_tmag_rate': 0.04194260485651214, 'outlier_dominated': True, 'long_run_dominated': True, 'small_motion_amplification': True, 'tmag_head_unbounded': True, 'recommended_guard': 'train_split_p95_clamp_with_median_logmag_correction'}

## robust scale guard 设计
S5E6 保留 S5E4 的 direction source，用 S5E5 raw log-magnitude 经过 train-split alpha correction 和 p95 clamp 输出 bounded magnitude。

## train-split magnitude prior 说明
{'median': 0.011908447102386463, 'p90': 0.2523947547524407, 'p95': 0.3691697395709953, 'max': 0.7818686910217085, 'alpha': 0.9497516241198531, 'clip_hi': 0.3691697395709953, 'clip_lo': 0.0020031222709819612}

## 为什么不能只看 tmag median
S5E5 已经说明 median 可以看起来还行，但 p95/p99/max 和 path_ratio 仍然会把整条轨迹拖爆。

## rot / signed tdir / tdir_abs / tmag / path_ratio gating 说明
S5E6 明确同时 gate rot、signed tdir、tdir_abs、anti-parallel、tmag 高分位和 path_ratio。

## training 结果
{'attempted': True, 'classification': 'S5E6_TRAINING_SMOKE_ONLY', 'config': 'configs/s5e6_robust_scale_guard_direction_gate.yaml', 'checkpoint_dir': 'checkpoints/S5E6_robust_scale_guard_direction_gate_candidate', 'uses_scene01_seq03_for_training': False, 'model_type': 'robust_scale_guard_direction_gate', 'train_magnitude_prior': {'median': 0.011908447102386463, 'p90': 0.2523947547524407, 'p95': 0.3691697395709953, 'max': 0.7818686910217085, 'alpha': 0.9497516241198531, 'clip_hi': 0.3691697395709953, 'clip_lo': 0.0020031222709819612}, 'losses': {'so3_geodesic': True, 'signed_tdir': True, 'tdir_abs_aux': True, 'anti_parallel_penalty': True, 'robust_tmag_log': True, 'tmag_p95_penalty': True, 'path_length_consistency': True, 'short_window_direction_consistency': True, 'smoothness': True}}

## traceable dense export coverage
{'available': True, 'trajectory_path': 'external_baselines/results/s5e6_traceable_dense/scene01_seq03_s5e6_traceable_dense_tum.txt', 'edge_provenance': 'external_baselines/results/s5e6_traceable_dense/edge_provenance.jsonl', 'num_poses': 454, 'num_edges': 453, 'direct_adjacent_prediction_edges': 453, 'coverage': 1.0, 'all_edges_traceable': True}

## component metrics，重点讨论 tmag 高分位、tdir、rot
{'rot_mean_deg': 0.9134395040767528, 'rot_median_deg': 0.797895569319923, 'rot_p90_deg': 1.3218302317546538, 'tdir_mean_deg': 51.47429479273258, 'tdir_median_deg': 42.61639148028055, 'tdir_p90_deg': 98.61703755727075, 'tdir_abs_mean_deg': 46.07968707914154, 'tdir_abs_median_deg': 42.3027471139478, 'tdir_abs_p90_deg': 78.37357550448448, 'tdir_mean_cosine': 0.5573840375058775, 'anti_parallel_rate': 0.1368653421633554, 'severe_wrong_sign_rate': 0.03532008830022075, 'direction_abs_good_but_signed_bad_rate': 0.002207505518763797, 'tmag_median_ratio': 11.352940388835835, 'tmag_mean_ratio': 17.20555072598954, 'tmag_p90_ratio': 45.26083001234384, 'tmag_p95_ratio': 63.490479510847564, 'tmag_p99_ratio': 101.61933582688962, 'tmag_max_ratio': 227.1712152926769, 'path_ratio': 1.880843438039364, 'top20_tmag_path_fraction': 0.11254258567185237, 'long_run_path_fraction': 0.022793399074077753}

## external evaluator none/se3/sim3
{'none': {'ate': 24.922442378420534, 'drift': 0.1862204219628343, 'path_ratio': 1.8808434381448762, 'status': 'ok', 'num_matched_poses': 454, 'tracking_success_rate': 1.0}, 'se3': {'ate': 12.016350787220915, 'drift': 0.1910960633024985, 'path_ratio': 1.8808434381448762, 'status': 'ok', 'num_matched_poses': 454, 'tracking_success_rate': 1.0}, 'sim3': {'ate': 4.012221895981248, 'drift': 0.1297471512654005, 'path_ratio': 1.8808434381448762, 'status': 'ok', 'num_matched_poses': 454, 'tracking_success_rate': 1.0}}

## 与 S5E5 / S5E4 / S5E3 / S5E2 比较
{'rot_preserved': True, 'signed_tdir_preserved_or_improved': True, 'tdir_abs_preserved_or_improved': True, 'anti_parallel_rate_preserved_or_reduced': True, 'tmag_median_improved': True, 'tmag_p95_improved': True, 'path_ratio_improved_vs_s5e4': True, 'path_ratio_improved_vs_s5e5': True, 'sim3_ate_improved_vs_s5e4': True, 'overall_geometry_improved': True}

## 与 ORB-SLAM3 比较
{'coverage_advantage': True, 'rot_close_to_orbslam3': True, 'tdir_gap_remaining': True, 'tmag_gap_remaining': True, 'aligned_ate_gap_to_orbslam3': 11.707806376528433, 'summary': 'S5E6 focuses on fixing scale outliers and path explosion while preserving S5E4 direction behavior.'}

## 是否更接近 ORB-SLAM3
如果 path explosion 真被压住，S5E6 会比 S5E5 更接近 ORB-SLAM3；但它仍然不会替代 official S5。

## rot / tdir / tmag 哪个仍是主要差距
rot 已稳定；tdir 和高分位 tmag/path_ratio 仍是主差距。

## 下一步建议
下一步可以在 S5E7 探索更强的 magnitude prior 或更稳的 learned magnitude head，但仍需保持 traceability 和 no-leakage。

## caveats
- S5E6 是 experimental candidate。
- 不替代 official S5 locked result。
- S5 locked metrics/policy unchanged。
- ORB-SLAM3 是 external strong baseline。
