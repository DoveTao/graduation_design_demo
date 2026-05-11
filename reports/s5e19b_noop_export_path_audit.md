# S5E19B no-op export path audit

## 执行摘要
最终分类：`S5E19B_MIXED_NOOP_AND_SMOKE`。

## 为什么 S5E19 指标几乎等于 S5E15
这次审计主要区分：真实无提升，还是 training smoke-only、export fallback、evaluator 复用旧路径导致的表面不变。

## trajectory diff
{'num_common_timestamps': 454, 'max_translation_difference': 0.009740779281393427, 'mean_translation_difference': 0.005958948542946845, 'max_quaternion_angular_difference_deg': 0.006397075879760356, 'mean_quaternion_angular_difference_deg': 0.0017373999368403676, 'identical_pose_count': 1, 'near_identical_pose_count': 1, 'trajectory_exact_copy': False, 'trajectory_near_noop': False}

## edge metrics diff
{'mean_abs_diff_rot': 0.0, 'mean_abs_diff_tdir': 1.4832873707806563e-14, 'anti_parallel_flag_changed_count': 0, 'tmag_ratio_mean_abs_diff': 0.006022508578509348, 'changed_edge_count': 453, 'unchanged_edge_count': 0, 'edge_metrics_noop': False}

## provenance delta audit
{'delta_tdir_zero_count': 453, 'delta_tdir_norm_mean': 0.0, 'delta_log_tmag_zero_count': 0, 'delta_log_tmag_abs_mean': 0.006779431247576175, 'sign_score_unique_count': 1, 'final_equals_coarse_count': 453, 'final_equals_s5e15_count': 0, 'refinement_head_effective': False, 'scale_head_effective': True, 'sign_head_effective': False}

## training path audit
{'real_training_executed': False, 'optimizer_step_present': False, 'learned_weights_saved': False, 'smoke_policy_only': True, 'losses_actually_computed': True}

## export path audit
{'loads_s5e19_checkpoint': True, 'loads_s5e15_checkpoint': True, 'fallback_to_s5e15': True, 'applies_delta_tdir': True, 'applies_delta_log_tmag': True, 'applies_sign_score': True, 'likely_noop_export': True}

## evaluator path audit
{'reads_s5e19_trajectory': True, 'reads_s5e15_trajectory': True, 'hardcoded_s5e15_metrics': True, 'likely_old_metrics_reuse': True}

## final diagnosis
{'s5e19_effectively_noop': False, 'smoke_policy_only': True, 'export_fallback_to_s5e15': True, 'evaluator_reused_old_metrics': True, 'true_model_no_improvement': False}

## 是否应该继续 S5E20
如果这里确认是 smoke/no-op/fallback，先修训练或导出路径；不要把 S5E19 当真实性能实验继续往后推。
