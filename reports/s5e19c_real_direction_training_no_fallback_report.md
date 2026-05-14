# S5E19C real direction training no-fallback report

## S5E19B 问题回顾
S5E19B 已经确认：S5E19 是 smoke-only + export fallback + evaluator 混入 S5E15 reference。

## 本轮如何避免 smoke-only
{'real_training_executed': True, 'optimizer_step_count': 180, 'learned_weights_saved': True, 'smoke_policy_only': False}

## 本轮如何禁止 S5E15 direction fallback
{'delta_tdir_zero_count': 0, 'delta_tdir_norm_mean': 0.6029416022053354, 'sign_score_unique_count': 453, 'final_equals_s5e15_direction_count': 0, 'fallback_to_s5e15_direction': False, 'learned_weights_loaded': True, 'export_used_learned_weights': True, 'evaluator_hardcoded_s5e15_metrics': False}

## 真实训练证据
{'real_training_executed': True, 'optimizer_step_count': 180, 'learned_weights_saved': True, 'smoke_policy_only': False}

## delta_tdir / sign_score 证据
{'delta_tdir_zero_count': 0, 'delta_tdir_norm_mean': 0.6029416022053354, 'sign_score_unique_count': 453, 'final_equals_s5e15_direction_count': 0, 'fallback_to_s5e15_direction': False, 'learned_weights_loaded': True, 'export_used_learned_weights': True, 'evaluator_hardcoded_s5e15_metrics': False}

## component metrics
{'rot_mean_deg': 0.9134395040767528, 'rot_median_deg': 0.797895569319923, 'rot_p90_deg': 1.3218302317546538, 'signed_tdir_mean_deg': 56.17528018993624, 'signed_tdir_median_deg': 50.11420328607101, 'signed_tdir_p90_deg': 101.58184252183032, 'tdir_abs_mean_deg': 50.59318455427033, 'tdir_abs_median_deg': 49.42133642078738, 'tdir_abs_p90_deg': 82.0810282129006, 'tdir_mean_cosine': 0.5021087114234821, 'anti_parallel_rate': 0.16556291390728478, 'severe_wrong_sign_rate': 0.02207505518763797, 'direction_abs_good_but_signed_bad_rate': 0.002207505518763797, 'tmag_median_ratio': 0.9879222977344607, 'tmag_p95_ratio': 3.6359100915472986, 'path_ratio': 0.12240164100086463}

## external evaluator
{'none': {'ate': 9.63427551263089, 'drift': 0.12951260142971552, 'path_ratio': 0.1224016409735106}, 'se3': {'ate': 4.016260541940681, 'drift': 0.13046622862205653, 'path_ratio': 0.1224016409735106}, 'sim3': {'ate': 3.9958080586973814, 'drift': 0.13109965044136235, 'path_ratio': 0.1224016409735106}}

## 是否优于 S5E15
{'signed_tdir_improved': False, 'anti_parallel_improved': False, 'path_ratio_improved': False, 'sim3_ate_improved': False}

## caveats
- S5E19C 是 experimental candidate。
- 不替代 official locked result。
- 不使用 eval GT calibration。
- 不使用 ORB-SLAM3 trajectory 作为训练 label。
