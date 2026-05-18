# ODOM360B vs TRAIN360E SEQ360B ODOM360A

- selected method: `pg_k3`
- selected params: `{'method_name': 'pg_k3', 'init_method': 'odom360a_selected', 'k_max': 3, 'k_decay': 0.5, 'w_rot': 1.0, 'w_tdir': 1.0, 'w_tmag': 0.5, 'w_smooth_rot': 0.1, 'w_smooth_t': 0.05, 'w_path': 0.05, 'w_rot_prior': 0.2, 'w_pos_prior': 0.1, 'robust_delta': 0.5, 'use_robust': True, 'use_outlier_downweight': True, 'outlier_angle_deg': 120.0, 'outlier_weight': 0.35, 'use_scale_pre_smoothing': 'ema', 'scale_ema_alpha': 0.4, 'use_rotation_pre_smoothing': 'ema', 'rotation_strength': 0.2, 'use_tdir_pre_suppression': True, 'tdir_window': 3, 'tdir_threshold_deg': 120.0, 'tdir_replacement_strength': 0.25, 'iterations': 40, 'lr': 0.005, 'optimizer': 'adam'}`
- compared to TRAIN360E: `better`
- compared to SEQ360B: `better`
- compared to ODOM360A: `worse`
- classification: `regression`
- next recommendation: `keep_ODOM360A_as_best_sequence_backend`
