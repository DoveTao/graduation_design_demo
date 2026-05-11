# S5E10 order-aware signed-direction scale recalibration report

## 最终分类
S5E10_INPUT_GEOMETRY_INSUFFICIENT

## 训练状态
{'attempted': True, 'classification': 'S5E10_SCALE_RECALIBRATION_SMOKE / S5E10_ORDER_SIGNED_DIRECTION_TRAINING_SMOKE', 'checkpoint_dir': 'checkpoints/S5E10_order_aware_signed_direction_scale_recalibration_candidate', 'training_status': {'scale_recalibrated_candidate': 'checkpoints/S5E10_order_aware_signed_direction_scale_recalibration_candidate/scale_recalibration_training_status.json', 'order_aware_signed_direction_candidate': 'checkpoints/S5E10_order_aware_signed_direction_scale_recalibration_candidate/order_signed_direction_training_status.json'}}

## 与 S5E9 的关系
S5E10 在 S5E9 已修复 raw scale explosion 的基础上，继续检查 signed direction 的 pair-order 可观测性，并尝试恢复过度保守的 scale/path。

## pair-order architecture audit
{'experiment': 'S5E10_order_aware_signed_direction_scale_recalibration', 'uses_ordered_concat': True, 'uses_feature_difference': True, 'uses_absolute_difference': False, 'uses_symmetric_pooling': False, 'pair_order_label_flip_consistent': True, 'reversed_pair_export_consistent': True, 'order_invariant_path_suspected': False, 'signed_direction_label_convention_valid': True, 'summary': 'S5E10 显式使用 ordered concat 与差分特征，没有只依赖对称聚合。'}

## 是否发现 order-invariant / label flip / augmentation / export bug
{'order_invariant_path_suspected': False, 'pair_order_label_flip_consistent': True, 'reversed_pair_export_consistent': True}

## order-aware signed direction candidate
{'attempted': True, 'classification': 'S5E10_ORDER_SIGNED_DIRECTION_TRAINING_SMOKE', 'checkpoint': 'checkpoints/S5E10_order_aware_signed_direction_scale_recalibration_candidate/s5e10_order_signed_best.pt', 'num_train_pairs': 707, 'num_val_pairs': 79, 'uses_scene01_seq03_for_training': False, 'reversed_pair_count': 707, 'reversed_pair_fraction': 1.0, 'history': [{'epoch': 0, 'val_metrics': {'signed_tdir_mean': 94.63567130173308, 'signed_tdir_median': 92.4681396484375, 'signed_tdir_p90': 105.48228302001954, 'tdir_abs_mean': 83.52155680596074, 'tdir_abs_median': 86.34136199951172, 'tdir_abs_p90': 89.3910629272461, 'anti_parallel_rate': 0.7215189873417721, 'severe_wrong_sign_rate': 0.012658227848101266, 'pair_order_dir_flip_cosine_mean': -0.9999991979780076, 'pair_order_dir_flip_success_rate': 0.0, 'sign_accuracy': 0.4430379746835443, 'axis_tdir_abs_mean': 83.52155680596074}}, {'epoch': 1, 'val_metrics': {'signed_tdir_mean': 96.68148871313167, 'signed_tdir_median': 92.71532440185547, 'signed_tdir_p90': 111.74635467529298, 'tdir_abs_mean': 81.3239017438285, 'tdir_abs_median': 85.82437896728516, 'tdir_abs_p90': 88.87986450195312, 'anti_parallel_rate': 0.6582278481012658, 'severe_wrong_sign_rate': 0.0759493670886076, 'pair_order_dir_flip_cosine_mean': -0.9999999592575846, 'pair_order_dir_flip_success_rate': 0.0, 'sign_accuracy': 0.4430379746835443, 'axis_tdir_abs_mean': 81.3239017438285}}, {'epoch': 2, 'val_metrics': {'signed_tdir_mean': 98.34583881233311, 'signed_tdir_median': 94.89556884765625, 'signed_tdir_p90': 117.80472564697267, 'tdir_abs_mean': 79.63329764257503, 'tdir_abs_median': 82.77220916748047, 'tdir_abs_p90': 88.6317123413086, 'anti_parallel_rate': 0.759493670886076, 'severe_wrong_sign_rate': 0.10126582278481013, 'pair_order_dir_flip_cosine_mean': -0.9999998377848275, 'pair_order_dir_flip_success_rate': 0.0, 'sign_accuracy': 0.4430379746835443, 'axis_tdir_abs_mean': 79.63329764257503}}], 'architecture_audit': {'uses_ordered_concat': True, 'uses_feature_difference': True, 'uses_absolute_difference': False, 'uses_symmetric_pooling': False, 'pair_order_label_flip_consistent': True, 'reversed_pair_export_consistent': True, 'order_invariant_path_suspected': False, 'signed_direction_label_convention_valid': True, 'summary': 'S5E10 显式使用 ordered concat 与差分特征，没有只依赖对称聚合。'}, 'notes': ['S5E10 显式使用 ordered concat、差分和乘积特征。', '训练时引入 reversed pair anti-symmetry 约束。'], 'validation_snapshot': {'verify_final_candidate': 'PASS', 'project_health_check': 'PASS', 's6_eval_only': 'PASS', 'unittest': 'PASS'}}

## reversed-pair diagnostic
{'pair_order_dir_flip_cosine_mean': -0.9999999907895692, 'pair_order_dir_flip_success_rate': 0.0, 'pair_order_scale_symmetry_error': 0.0, 'pair_order_failure_rate': 1.0, 'signed_direction_order_observable': False}

## signed direction reliability mask
S5E10 对 near_static 和 small_motion 降低 signed direction loss 权重，对 normal_motion / large_motion 保留完整 signed loss。

## axis + sign decomposition 结果
{'signed_tdir_mean': 98.34583881233311, 'signed_tdir_median': 94.89556884765625, 'signed_tdir_p90': 117.80472564697267, 'tdir_abs_mean': 79.63329764257503, 'tdir_abs_median': 82.77220916748047, 'tdir_abs_p90': 88.6317123413086, 'anti_parallel_rate': 0.759493670886076, 'severe_wrong_sign_rate': 0.10126582278481013, 'pair_order_dir_flip_cosine_mean': -0.9999998377848275, 'pair_order_dir_flip_success_rate': 0.0, 'sign_accuracy': 0.4430379746835443, 'axis_tdir_abs_mean': 79.63329764257503}

## scale recalibration 结果
{'attempted': True, 'classification': 'S5E10_SCALE_RECALIBRATION_SMOKE', 'num_train_pairs': 707, 'num_val_pairs': 79, 'uses_scene01_seq03_for_training': False, 'bucket_prior_choice': 'p95', 'bucket_priors': {'global': {'median': 0.011908447102386463, 'p90': 0.2523947547524407, 'p95': 0.3691697395709953, 'p99': 0.526654362281555, 'max': 0.7818686910217085, 'clip_lo': 0.0020031222709819612, 'clip_hi': 0.3691697395709953}, 'near_static': {'median': 0.0020024984394507806, 'p90': 0.0025388581303073536, 'p95': 0.0026094009608409363, 'p99': 0.0026432166918995636, 'max': 0.002647640458973654, 'clip_lo': 0.0011955469109660413, 'clip_hi': 0.0026094009608409363}, 'small_motion': {'median': 0.0035446469453934365, 'p90': 0.004364841540517953, 'p95': 0.0045834041100720375, 'p99': 0.004763641215385276, 'max': 0.004801041553661831, 'clip_lo': 0.002716339369040706, 'clip_hi': 0.0045834041100720375}, 'normal_motion': {'median': 0.011908447102386463, 'p90': 0.02716551357981745, 'p95': 0.03155805742315243, 'p99': 0.03644346638655226, 'max': 0.03816320217172562, 'clip_lo': 0.005123810287804876, 'clip_hi': 0.03155805742315243}, 'large_motion': {'median': 0.2105054393596516, 'p90': 0.4370886400382462, 'p95': 0.5229383635602999, 'p99': 0.5870292489538451, 'max': 0.7818686910217085, 'clip_lo': 0.047578320811733917, 'clip_hi': 0.5229383635602999}}, 'notes': ['S5E10 scale recalibration 改用更宽但仍受控的 bucket p95 prior。', '目标是在不重新引入 raw scale explosion 的前提下，把 S5E9 过度保守的 raw path_ratio 拉回一些。'], 'split_summary': {'split_by': 'scene_seq', 'train_ratio': 0.8, 'split_seed': 3407, 'train_sequence_keys': [['scene01', 'seq01'], ['scene01', 'seq02']], 'test_sequence_keys': [['scene01', 'seq03']], 'all_sequence_keys': [['scene01', 'seq01'], ['scene01', 'seq02'], ['scene01', 'seq03']]}, 'validation_snapshot': {'verify_final_candidate': 'PASS', 'project_health_check': 'PASS', 's6_eval_only': 'PASS', 'unittest': 'PASS'}}

## small-motion / near-static bucket audit
{'small_motion_count': 114, 'small_motion_fraction': 0.25165562913907286, 'small_motion_gt_tmag_median': 0.002354864324871946, 'small_motion_gt_tmag_p95': 0.00338865012809901, 'small_motion_pred_tmag_median': 174.0041547999316, 'small_motion_pred_tmag_p95': 323.1039380153365, 'small_motion_tdir_mean': 140.46317733045174, 'small_motion_tdir_p90': 161.5683127989872, 'small_motion_signed_tdir_mean': 140.46317733045174, 'small_motion_signed_tdir_p90': 161.5683127989872, 'small_motion_anti_parallel_rate': 1.0, 'small_motion_path_fraction': 0.23903692847715013, 'small_motion_error_contribution': 1.176340464262842, 'small_motion_amplification': True}

## raw prediction metrics
{'rot_mean_deg': 0.9134395040767528, 'rot_median_deg': 0.797895569319923, 'rot_p90_deg': 1.3218302317546538, 'tdir_mean_deg': 119.40690777688542, 'tdir_median_deg': 129.19577455736976, 'tdir_p90_deg': 153.27137500763004, 'tdir_abs_mean_deg': 44.462857545185905, 'tdir_abs_median_deg': 44.7427534060093, 'tdir_abs_p90_deg': 59.615143941086686, 'tdir_mean_cosine': -0.4481447417993258, 'anti_parallel_rate': 0.8344370860927153, 'severe_wrong_sign_rate': 0.7461368653421634, 'direction_abs_good_but_signed_bad_rate': 0.3598233995584989, 'tmag_median_ratio': 36.23954472208068, 'tmag_mean_ratio': 73.16532017818942, 'tmag_p90_ratio': 193.86165342434504, 'tmag_p95_ratio': 241.83284101236717, 'tmag_p99_ratio': 348.7862847588531, 'tmag_max_ratio': 1138.9512736753954, 'path_ratio': 6.56978141883022, 'top20_tmag_path_fraction': 0.056935826064121085, 'long_run_path_fraction': 0.008553584123097799}

## guarded prediction metrics
{'rot_mean_deg': 0.9134395040767528, 'rot_median_deg': 0.797895569319923, 'rot_p90_deg': 1.3218302317546538, 'tdir_mean_deg': 119.40690777688542, 'tdir_median_deg': 129.19577455736976, 'tdir_p90_deg': 153.27137500763004, 'tdir_abs_mean_deg': 44.462857545185905, 'tdir_abs_median_deg': 44.7427534060093, 'tdir_abs_p90_deg': 59.615143941086686, 'tdir_mean_cosine': -0.4481447417993258, 'anti_parallel_rate': 0.8344370860927153, 'severe_wrong_sign_rate': 0.7461368653421634, 'direction_abs_good_but_signed_bad_rate': 0.3598233995584989, 'tmag_median_ratio': 36.23954472208068, 'tmag_mean_ratio': 73.16532017818942, 'tmag_p90_ratio': 193.86165342434504, 'tmag_p95_ratio': 241.83284101236717, 'tmag_p99_ratio': 348.7862847588531, 'tmag_max_ratio': 1138.9512736753954, 'path_ratio': 6.56978141883022, 'top20_tmag_path_fraction': 0.056935826064121085, 'long_run_path_fraction': 0.008553584123097799}

## raw vs guarded gap
{'tdir_gap': 0.0, 'tdir_abs_gap': 0.0, 'tmag_p95_gap': 0.0, 'tmag_max_gap': 0.0, 'path_ratio_gap': 0.0, 'sim3_ATE_gap': 0.0}

## external evaluator results
{'guarded': {'none': {'ate': 78.18140398562333, 'drift': 0.47477383857485184, 'path_ratio': 6.569781406717808}, 'se3': {'ate': 43.12801354318913, 'drift': 0.4083338448511545, 'path_ratio': 6.569781406717808}, 'sim3': {'ate': 3.7056513642993028, 'drift': 0.12600668411379823, 'path_ratio': 6.569781406717808}}, 'raw': {'none': {'ate': 78.18140398562333, 'drift': 0.47477383857485184, 'path_ratio': 6.569781406717808}, 'se3': {'ate': 43.12801354318913, 'drift': 0.4083338448511545, 'path_ratio': 6.569781406717808}, 'sim3': {'ate': 3.7056513642993028, 'drift': 0.12600668411379823, 'path_ratio': 6.569781406717808}}}

## comparison vs S5E9/S5E8/S5E7/S5E6/S5E5/S5E4/S5E3/S5E2
{'pair_order_dir_flip_success_rate_improved': False, 'signed_tdir_mean_improved_vs_s5e9': False, 'anti_parallel_rate_improved_vs_s5e9': False, 'scale_recalibrated_vs_s5e9': False, 'raw_vs_guarded_gap_small': True}

## comparison vs ORB-SLAM3
{'coverage': 'S5E10 454/454 vs ORB-SLAM3 273/454', 'edge_level_gap': 'unavailable', 'se3_ate_gap': 42.81946913249664, 'sim3_ate_gap': 3.4813591983126786, 'path_ratio': 'S5E10=6.56978141883022; ORB-SLAM3=0.2998258665660257'}

## blockers
['pair-order 可观测性仍然不足，dir flip success rate 没有明显起来。', 'raw signed direction 没有明显优于 S5E9。', 'raw path_ratio 虽有变化，但没有形成理想的 scale recalibration。']

## validation results
{'verify_final_candidate': 'PASS', 'project_health_check': 'PASS', 's6_eval_only': 'PASS', 'unittest': 'PASS'}

## caveats
- S5E10 是实验性结果，不替换 S5 locked。
- 只有 raw signed direction 和 raw scale/path 同时改善，且 raw vs guarded gap 保持很小，才可称为真实几何改善。
