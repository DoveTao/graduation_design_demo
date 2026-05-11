# S5E5 temporal visual backbone report

## 执行摘要
S5E5 引入真正的 ordered image pair temporal visual backbone，并与 S5E4/S5E3 numeric prior 融合。`final_classification = S5E5_TMAG_IMPROVED_TDIR_STILL_BAD`。

## S5E4 failure recap
S5E4 修复了 signed tdir 的反向问题，但 `tdir_abs` 回退到 S5E2 水平，`tmag` 与 `path_ratio` 也没有继续改善，说明仅靠 signed prior 还不够。

## temporal visual backbone 设计
S5E5 使用 ordered image pair CNN，输入 `Ii, Ij, Ij-Ii, abs(Ij-Ii)`，并与 S5E4/S5E3 numeric prior 融合。

## 为什么需要 ordered image pair，而不是 symmetric statistics
symmetric statistics 会抹掉时间方向；ordered image pair 保留 `i->j` 的方向性，有助于同时优化 signed tdir 和 tdir_abs。

## rot / signed tdir / tdir_abs / tmag gating 说明
S5E5 不以 ATE 单独判定成功；rot、signed tdir、tdir_abs、anti_parallel、tmag、path_ratio 都必须同时报告。

## training 结果
{'attempted': True, 'classification': 'S5E5_TRAINING_SMOKE_ONLY', 'config': 'configs/s5e5_temporal_visual_backbone_geometry.yaml', 'checkpoint_dir': 'checkpoints/S5E5_temporal_visual_backbone_geometry_candidate', 'uses_scene01_seq03_for_training': False, 'model_type': 'temporal_visual_backbone', 'losses': {'so3_geodesic': True, 'signed_tdir': True, 'tdir_abs_aux': True, 'anti_parallel_penalty': True, 'tmag_log': True, 'path_length_consistency': True, 'short_window_direction_consistency': True, 'smoothness': True}, 'best_checkpoint': 'checkpoints/S5E5_temporal_visual_backbone_geometry_candidate/s5e5_temporal_visual_best.pt'}

## traceable dense export coverage
{'available': True, 'trajectory_path': 'external_baselines/results/s5e5_traceable_dense/scene01_seq03_s5e5_traceable_dense_tum.txt', 'edge_provenance': 'external_baselines/results/s5e5_traceable_dense/edge_provenance.jsonl', 'num_poses': 454, 'num_edges': 453, 'direct_adjacent_prediction_edges': 453, 'coverage': 1.0, 'all_edges_traceable': True}

## component metrics，重点讨论 rot 和 tdir
{'experiment': 'S5E5_temporal_visual_backbone_geometry_candidate', 'available': True, 'trajectory_path': 'external_baselines/results/s5e5_traceable_dense/scene01_seq03_s5e5_traceable_dense_tum.txt', 'edge_provenance': 'external_baselines/results/s5e5_traceable_dense/edge_provenance.jsonl', 'num_poses': 454, 'num_edges': 453, 'direct_adjacent_prediction_edges': 453, 'coverage': 1.0, 'all_edges_traceable': True, 'rot_mean_deg': 0.9134395040767528, 'rot_median_deg': 0.797895569319923, 'rot_p90_deg': 1.3218302317546538, 'tdir_mean_deg': 56.14019055030064, 'tdir_median_deg': 53.842707330263345, 'tdir_p90_deg': 92.30096311683484, 'tdir_abs_mean_deg': 52.096460882587564, 'tdir_abs_median_deg': 52.765473602918775, 'tdir_abs_p90_deg': 78.79367192164725, 'tdir_mean_cosine': 0.5120819299839849, 'anti_parallel_rate': 0.10816777041942605, 'severe_wrong_sign_rate': 0.008830022075055188, 'direction_abs_good_but_signed_bad_rate': 0.006622516556291391, 'tmag_median_ratio': 11.953588693727822, 'tmag_mean_ratio': 71.64852808991282, 'tmag_p90_ratio': 48.023883417026354, 'tmag_p95_ratio': 95.44378091660477, 'tmag_max_ratio': 6869.16668913065, 'path_ratio': 6.591091747656769}

## external evaluator none/se3/sim3
{'none': {'ate': 113.44308639694538, 'drift': 2.1011291922148745, 'path_ratio': 6.591091680514336, 'status': 'ok', 'num_matched_poses': 454, 'tracking_success_rate': 1.0}, 'se3': {'ate': 66.13733880826825, 'drift': 2.104015006750345, 'path_ratio': 6.591091680514336, 'status': 'ok', 'num_matched_poses': 454, 'tracking_success_rate': 1.0}, 'sim3': {'ate': 4.111550831251635, 'drift': 0.13129335402424674, 'path_ratio': 6.591091680514336, 'status': 'ok', 'num_matched_poses': 454, 'tracking_success_rate': 1.0}}

## 与 S5E4 / S5E3 / S5E2 比较
{'rot_preserved': True, 'signed_tdir_improved': False, 'tdir_abs_improved': False, 'anti_parallel_rate_reduced_or_preserved': True, 'tmag_improved': True, 'path_ratio_improved': False, 'sim3_ate_improved': False, 'overall_geometry_improved': False}

## 与 ORB-SLAM3 比较
{'coverage_advantage': True, 'rot_close_to_orbslam3': True, 'tdir_gap_remaining': True, 'tmag_gap_remaining': True, 'aligned_ate_gap_to_orbslam3': 65.82879439757578, 'summary': 'S5E5 tests whether temporal visual features improve direction and magnitude jointly while keeping full coverage.'}

## 是否更接近 ORB-SLAM3
如果 signed tdir、tdir_abs、tmag 和 path_ratio 同时改善，才算真正更接近 ORB-SLAM3；否则只是局部修补。

## rot / tdir / tmag 哪个仍是主要差距
rot 已较稳定；tdir 和 tmag/path_ratio 通常仍是决定 trajectory 误差的主差距。

## 下一步建议
下一步可以把 temporal visual backbone 做得更深，或进入 S5E6 的 ORB-aware distillation，但那需要新的实验边界。

## caveats
- S5E5 是 experimental candidate。
- 不替代 official S5 locked result。
- S5 locked metrics/policy unchanged。
- ORB-SLAM3 是 external strong baseline。
