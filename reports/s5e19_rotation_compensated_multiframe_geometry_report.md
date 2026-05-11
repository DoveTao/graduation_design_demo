# S5E19 rotation compensated multiframe geometry report

## 执行摘要
S5E19 最终分类：`S5E19_TRAINING_SMOKE_ONLY`。

## 为什么 S5E16-S5E18 后转向主模型结构改动
- router / hand-crafted signal 路线已经验证收益有限，因此本轮回到球面 token + coarse-to-fine + geometry constraint 主线。

## 结构是否保留主线
{'uses_spherical_erp_token_encoder': True, 'uses_coarse_pose_head': True, 'uses_fine_residual_refinement': True, 'uses_rotation_compensated_direction_head': True, 'uses_fine_scale_residual_head': True, 'uses_multiframe_geometry_loss': True, 'uses_observability_weighted_loss': True, 'uses_anti_parallel_hard_negative_loss': True, 'uses_external_router': False}

## rotation-compensated direction head 设计
通过 coarse rotation 把相邻边方向变到世界系做短窗口一致性，再回投到当前局部坐标系。

## fine scale residual head 与 S5E15 scale prior
默认沿用 S5E15 的 scale/path 稳定基础，只做小幅 multiframe 邻域尺度平滑。

## short-window multi-frame geometry loss
{'k1_tdir': 50.35304148245505, 'composition_error_mean': 50.35304148245505, 'path_length_consistency_error': 0.05318809915997954, 'rotation_composition_error': 0.9134395044083475, 'k2_tdir': 48.066943689366354, 'k3_tdir': 47.774063258873134, 'k5_tdir': 47.298629760079265}

## observability-weighted geometry loss
本轮仍保留 observability 权重，但没有把 external router 作为主结构。

## anti-parallel hard negative loss
以 smoke-only 方式保留该 head/constraint 的结构接入，不夸大其训练效果。

## training status
{'attempted': True, 'classification': 'S5E19_TRAINING_SMOKE_ONLY', 'uses_eval_gt_for_training': False, 'uses_eval_gt_for_scale': False, 'uses_orbslam3_teacher': False, 'best_checkpoint': 'checkpoints/S5E19_rotation_compensated_multiframe_geometry_candidate/s5e19_smoke_policy.json'}

## traceable dense export
{'available': True, 'trajectory_path': 'external_baselines/results/s5e19_traceable_dense/scene01_seq03_s5e19_traceable_dense_tum.txt', 'edge_provenance': 'external_baselines/results/s5e19_traceable_dense/edge_provenance.jsonl', 'num_poses': 454, 'num_edges': 453, 'coverage': 1.0, 'all_edges_traceable': True}

## component metrics
{'rot_mean_deg': 0.9134395040767528, 'tdir_mean_deg': 50.353041365033235, 'tdir_abs_mean_deg': 43.16064629037026, 'tdir_mean_cosine': 0.5591010842073055, 'anti_parallel_rate': 0.1479028697571744, 'tmag_median_ratio': 1.26853220360632, 'tmag_p95_ratio': 4.668572667067984, 'path_ratio': 0.15674640990008198}

## external evaluator none/se3/sim3
{'none': {'ate': 9.328785504501196, 'drift': 0.12857538356234996, 'path_ratio': 0.15674641015640298}, 'se3': {'ate': 3.9809703358574935, 'drift': 0.12989184276204593, 'path_ratio': 0.15674641015640298}, 'sim3': {'ate': 3.9682105001578156, 'drift': 0.13014348493234643, 'path_ratio': 0.15674641015640298}}

## 与 S5E15 / S5E18 / ORB-SLAM3 比较
{'rot_preserved': True, 'tdir_improved': False, 'anti_parallel_reduced': False, 'tmag_median_in_range': True, 'path_ratio_improved_without_explosion': False, 'sim3_ate_improved': False, 'overall_geometry_improved': False}

## 是否继续后续实验
{'can_continue_to_s5e20': False, 'main_remaining_blocker': 'tdir_signal_insufficient', 'recommended_next_stage': 'stop_or_revisit_mainline_training_capacity'}

## caveats
- S5E19 是 experimental candidate。
- 不替代 official locked result。
- 没有使用 eval GT calibration。
- 没有使用 ORB-SLAM3 teacher。
- smoke training 不得夸大。
