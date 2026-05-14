# S5E9 scale-unit, small-motion, signed-direction fix report

## 最终分类
S5E9_SCALE_RAW_IMPROVED

## 训练状态
{'attempted': True, 'classification': 'S5E9_SCALE_TRAINING_SMOKE', 'checkpoint': 'checkpoints/S5E9_scale_unit_small_motion_signed_direction_fix_candidate/s5e9_scale_best.pt', 'num_train_pairs': 707, 'num_val_pairs': 79, 'uses_scene01_seq03_for_training': False, 'history': [{'epoch': 0, 'train_stats': {'scale_loss': 0.020062705501914024, 'overscale_rate': 0.0, 'small_motion_weight': 1.0}, 'val_p95_ratio': 0.9204014241695404}, {'epoch': 1, 'train_stats': {'scale_loss': 0.012424352578818798, 'overscale_rate': 0.0, 'small_motion_weight': 1.0}, 'val_p95_ratio': 1.018950378894806}, {'epoch': 2, 'train_stats': {'scale_loss': 0.01533674169331789, 'overscale_rate': 0.0, 'small_motion_weight': 1.0}, 'val_p95_ratio': 0.967458164691925}], 'split_summary': {'split_by': 'scene_seq', 'train_ratio': 0.8, 'split_seed': 3407, 'train_sequence_keys': [['scene01', 'seq01'], ['scene01', 'seq02']], 'test_sequence_keys': [['scene01', 'seq03']], 'all_sequence_keys': [['scene01', 'seq01'], ['scene01', 'seq02'], ['scene01', 'seq03']]}, 'notes': ['S5E9 scale candidate 使用 tight bounded log-scale residual。', 'small-motion 使用更低权重。'], 'validation_snapshot': {'verify_final_candidate': 'PASS', 'project_health_check': 'PASS', 's6_eval_only': 'PASS', 'unittest': 'PASS'}}

## 与 S5E8 的关系
S5E9 重点检查 raw scale explosion 到底来自字段/单位链路问题，还是模型真的学坏了。

## scale pipeline audit
{'experiment': 'S5E9_scale_unit_small_motion_signed_direction_fix', 'gt_tmag': {'median': 0.007506645044218322, 'p90': 0.21251344154401194, 'p95': 0.33083908526191264, 'max': 0.740239318259982, 'p99': 0.5865205778651041}, 'prior_tmag': {'median': 0.0035446469453934365, 'p90': 0.0035446469453934365, 'p95': 0.0035446469453934365, 'max': 0.0035446469453934365}, 'pred_log_scale_delta': {'median': -0.16614298522472382, 'p90': -0.15576401650905608, 'p95': -0.15331655442714692, 'max': -0.1508728712797165, 'p99': -0.15136206448078154}, 'pred_tmag_pre_export': {'median': 0.0030020505670619196, 'p90': 0.0030333710144452076, 'p95': 0.0030408041687753702, 'max': 0.003048244013285073}, 'pred_tmag_exported': {'median': 0.0030020505670619196, 'p90': 0.0030333710144452076, 'p95': 0.0030408041687753702, 'max': 0.003048244013285073}, 'pred_tmag_evaluator': {'median': 0.003002050321691656, 'p90': 0.003033370524897047, 'p95': 0.0030408043962829425, 'max': 0.0030482443497502286}, 'export_eval_tmag_consistency': True, 'unit_mismatch_suspected': False, 'raw_tmag_to_gt_tmag_median_ratio': 0.39908889274408793, 'raw_tmag_to_gt_tmag_p95_ratio': 1.6560806175064955, 'summary': 'S5E9 首先检查 raw scale explosion 是不是来自单位、导出或字段使用错误。'}

## 是否发现 unit/export/eval mismatch
{'unit_mismatch_suspected': False, 'export_eval_tmag_consistency': True}

## tight bounded scale residual 结果
{'attempted': True, 'classification': 'S5E9_SCALE_TRAINING_SMOKE', 'checkpoint': 'checkpoints/S5E9_scale_unit_small_motion_signed_direction_fix_candidate/s5e9_scale_best.pt', 'num_train_pairs': 707, 'num_val_pairs': 79, 'uses_scene01_seq03_for_training': False, 'history': [{'epoch': 0, 'train_stats': {'scale_loss': 0.020062705501914024, 'overscale_rate': 0.0, 'small_motion_weight': 1.0}, 'val_p95_ratio': 0.9204014241695404}, {'epoch': 1, 'train_stats': {'scale_loss': 0.012424352578818798, 'overscale_rate': 0.0, 'small_motion_weight': 1.0}, 'val_p95_ratio': 1.018950378894806}, {'epoch': 2, 'train_stats': {'scale_loss': 0.01533674169331789, 'overscale_rate': 0.0, 'small_motion_weight': 1.0}, 'val_p95_ratio': 0.967458164691925}], 'split_summary': {'split_by': 'scene_seq', 'train_ratio': 0.8, 'split_seed': 3407, 'train_sequence_keys': [['scene01', 'seq01'], ['scene01', 'seq02']], 'test_sequence_keys': [['scene01', 'seq03']], 'all_sequence_keys': [['scene01', 'seq01'], ['scene01', 'seq02'], ['scene01', 'seq03']]}, 'notes': ['S5E9 scale candidate 使用 tight bounded log-scale residual。', 'small-motion 使用更低权重。'], 'validation_snapshot': {'verify_final_candidate': 'PASS', 'project_health_check': 'PASS', 's6_eval_only': 'PASS', 'unittest': 'PASS'}}

## small-motion / near-static bucket audit
{'small_motion_count': 114, 'small_motion_fraction': 0.25165562913907286, 'small_motion_gt_tmag_median': 0.002354864324871946, 'small_motion_gt_tmag_p95': 0.00338865012809901, 'small_motion_pred_tmag_median': 1.2797779040911372, 'small_motion_pred_tmag_p95': 2.3509597644570133, 'small_motion_tdir_mean': 59.47774316599026, 'small_motion_tdir_p90': 89.4427684272278, 'small_motion_signed_tdir_mean': 59.47774316599026, 'small_motion_signed_tdir_p90': 89.4427684272278, 'small_motion_anti_parallel_rate': 0.09649122807017543, 'small_motion_path_fraction': 0.2516702570922094, 'small_motion_error_contribution': 1.1086224487587648, 'small_motion_amplification': True}

## pair-order / signed direction diagnostic
{'pair_order_dir_flip_cosine_mean': 0.19340398164093905, 'pair_order_dir_flip_success_rate': 0.24061810154525387, 'pair_order_scale_symmetry_error': 0.0, 'pair_order_failure_rate': 0.7593818984547461, 'signed_direction_order_observable': False}

## scale-first candidate metrics
{'epoch': 2, 'train_stats': {'scale_loss': 0.01533674169331789, 'overscale_rate': 0.0, 'small_motion_weight': 1.0}, 'val_p95_ratio': 0.967458164691925}

## signed-direction candidate metrics
{'signed_tdir_mean': 105.84170259403277, 'signed_tdir_median': 149.87103271484375, 'signed_tdir_p90': 170.54112854003907, 'tdir_abs_mean': 20.78894360306897, 'tdir_abs_median': 17.21059226989746, 'tdir_abs_p90': 33.23591079711916, 'anti_parallel_rate': 0.620253164556962, 'severe_wrong_sign_rate': 0.5949367088607594, 'pair_order_dir_flip_cosine_mean': -0.9104710885241062, 'pair_order_dir_flip_success_rate': 0.0}

## raw prediction metrics
{'rot_mean_deg': 0.9134395040767528, 'rot_median_deg': 0.797895569319923, 'rot_p90_deg': 1.3218302317546538, 'tdir_mean_deg': 53.65013421168108, 'tdir_median_deg': 45.04897762906787, 'tdir_p90_deg': 93.59975390353512, 'tdir_abs_mean_deg': 48.92733872256603, 'tdir_abs_median_deg': 45.028682781700056, 'tdir_abs_p90_deg': 79.88821517251566, 'tdir_mean_cosine': 0.5411489354197707, 'anti_parallel_rate': 0.10596026490066225, 'severe_wrong_sign_rate': 0.033112582781456956, 'direction_abs_good_but_signed_bad_rate': 0.002207505518763797, 'tmag_median_ratio': 0.39908888763230593, 'tmag_mean_ratio': 0.5858264357524848, 'tmag_p90_ratio': 1.3689528691665809, 'tmag_p95_ratio': 1.6560804642823275, 'tmag_p99_ratio': 2.419937837464527, 'tmag_max_ratio': 7.078435706785601, 'path_ratio': 0.05272511562778335, 'top20_tmag_path_fraction': 0.04416932337552042, 'long_run_path_fraction': 0.008823800464620356}

## guarded prediction metrics
{'rot_mean_deg': 0.9134395040767528, 'rot_median_deg': 0.797895569319923, 'rot_p90_deg': 1.3218302317546538, 'tdir_mean_deg': 53.65013421168108, 'tdir_median_deg': 45.04897762906787, 'tdir_p90_deg': 93.59975390353512, 'tdir_abs_mean_deg': 48.92733872256603, 'tdir_abs_median_deg': 45.028682781700056, 'tdir_abs_p90_deg': 79.88821517251566, 'tdir_mean_cosine': 0.5411489354197707, 'anti_parallel_rate': 0.10596026490066225, 'severe_wrong_sign_rate': 0.033112582781456956, 'direction_abs_good_but_signed_bad_rate': 0.002207505518763797, 'tmag_median_ratio': 0.39908888763230593, 'tmag_mean_ratio': 0.5858264357524848, 'tmag_p90_ratio': 1.3689528691665809, 'tmag_p95_ratio': 1.6560804642823275, 'tmag_p99_ratio': 2.419937837464527, 'tmag_max_ratio': 7.078435706785601, 'path_ratio': 0.05272511562778335, 'top20_tmag_path_fraction': 0.04416932337552042, 'long_run_path_fraction': 0.008823800464620356}

## raw vs guarded gap
{'tdir_gap': 0.0, 'tdir_abs_gap': 0.0, 'tmag_p95_gap': 0.0, 'tmag_max_gap': 0.0, 'path_ratio_gap': 0.0, 'sim3_ATE_gap': 0.0}

## external evaluator results
{'guarded': {'none': {'ate': 9.659702088912063, 'drift': 0.1292975007021164, 'path_ratio': 0.05272511562873076}, 'se3': {'ate': 4.063931954977213, 'drift': 0.12965925694214364, 'path_ratio': 0.05272511562873076}, 'sim3': {'ate': 3.928931860681025, 'drift': 0.13038959631402616, 'path_ratio': 0.05272511562873076}}, 'raw': {'none': {'ate': 9.659702088912063, 'drift': 0.1292975007021164, 'path_ratio': 0.05272511562873076}, 'se3': {'ate': 4.063931954977213, 'drift': 0.12965925694214364, 'path_ratio': 0.05272511562873076}, 'sim3': {'ate': 3.928931860681025, 'drift': 0.13038959631402616, 'path_ratio': 0.05272511562873076}}}

## comparison vs S5E8/S5E7/S5E6/S5E5/S5E4/S5E3/S5E2
{'raw_tmag_p95_improved_vs_s5e8': True, 'raw_tmag_max_improved_vs_s5e8': True, 'raw_path_ratio_improved_vs_s5e8': True, 'raw_tdir_abs_mean_improved_vs_s5e8': False, 'anti_parallel_not_regressed': True, 'guard_dependency': False}

## comparison vs ORB-SLAM3
{'coverage': 'S5E9 454/454 vs ORB-SLAM3 273/454', 'edge_level_gap': 'unavailable', 'se3_ate_gap': 3.7553875442847313, 'sim3_ate_gap': 3.704639694694401, 'path_ratio': 'S5E9=0.05272511562778335; ORB-SLAM3=0.2998258665660257'}

## blockers
['raw signed direction 没有稳定优于 S5E8。', 'small-motion amplification 仍存在。']

## validation results
{'verify_final_candidate': 'PASS', 'project_health_check': 'PASS', 's6_eval_only': 'PASS', 'unittest': 'PASS'}

## caveats
- S5E9 是实验性结果，不替换 S5 locked。
- 只有 raw scale/path 脱离 guard 后明显改善，才可以认定真实几何改善。
