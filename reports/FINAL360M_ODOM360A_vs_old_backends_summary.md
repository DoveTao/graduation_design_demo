# ODOM360A vs TRAIN360E and SEQ360B

- selected fusion method: `local_window_pose_fusion`
- selected params: `{'use_scale_pre_smoothing': 'ema', 'scale_ema_alpha': 0.4, 'use_rotation_pre_smoothing': 'ema', 'rotation_strength': 0.2, 'use_tdir_pre_suppression': True, 'tdir_window': 3, 'tdir_threshold_deg': 120.0, 'tdir_replacement_strength': 0.25, 'k_max': 2, 'k_decay': 0.7}`
- compared to TRAIN360E: `better`
- compared to SEQ360B: `better`
- classification: `partial`
- next recommendation: `proceed_to_ODOM360B_local_pose_graph_with_kstep_constraints`
