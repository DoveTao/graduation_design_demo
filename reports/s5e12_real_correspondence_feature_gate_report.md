# S5E12 real correspondence feature gate report

## 最终分类
S5E12_FEATURE_GATE_PASSED

## dependency check 结果
{'experiment': 'S5E12_real_correspondence_feature_gate', 'config': 'configs/s5e12_real_correspondence_feature_gate.yaml', 'cv2_available': True, 'cv2_version': '4.13.0', 'cv2_cuda_devices': 0, 'cv2_error': None, 'orb_available': True, 'sift_available': True, 'matcher_available': True, 'essential_matrix_available': True, 'recover_pose_available': True, 'kornia_available': False, 'kornia_error': "No module named 'kornia'", 'optical_flow_available': True, 'camera_intrinsics_available': False, 'camera_intrinsics_path': 'external_baselines/dataset/scene01_seq03/camera.yaml', 'camera_intrinsics_notes': ['camera.yaml 存在，但当前数据是 panorama / non-pinhole 描述，不能当作可直接用于 essential geometry 的标准 intrinsics。'], 'can_extract_real_correspondence_features': True, 'dependency_classification': 'S5E12_DEPENDENCY_READY', 'notes': ['PyTorch 继续负责 CUDA 训练，OpenCV 当前只要求支持 CPU 侧 keypoint / match / essential geometry 特征提取。', '即使 cv2 cuda devices 为 0，也不影响 S5E12 real correspondence feature gate。']}

## 与 S5E11 的关系
S5E12 不训练新模型，而是验证真实 correspondence / optical flow / parallax features 是否能把 signed direction supervision reliable fraction 从 S5E11 的极低水平拉高。

## real correspondence feature availability
{'real_correspondence_features_available': True, 'feature_extraction_coverage': 1.0, 'proxy_fallback_explicitly_marked': True}

## optical flow availability
{'optical_flow_available': True, 'flow_summary_present': 0.44193559995616183}

## strict essential geometry availability
{'experiment': 'S5E12_real_correspondence_feature_gate', 'camera_intrinsics_available': False, 'camera_model': 'equirectangular_panorama', 'strict_essential_geometry_available': False, 'essential_geometry_blocker': 'pinhole_intrinsics_missing_or_non_pinhole_camera_model', 'findEssentialMat_api_available': True, 'recoverPose_api_available': True, 'essential_axis_gt_dir_mean_angle': 'unavailable', 'essential_axis_gt_dir_p90_angle': 'unavailable', 'essential_axis_sign_ambiguity_rate': 'unavailable', 'reason': 'pinhole_intrinsics_missing_or_non_pinhole_camera_model'}

## intrinsics blocker
pinhole_intrinsics_missing_or_non_pinhole_camera_model

## feature extraction coverage
{'experiment': 'S5E12_real_correspondence_feature_gate', 'edge_count': 453, 'correspondence_success_count': 453, 'correspondence_success_fraction': 1.0, 'filtered_match_count_median': 950.0, 'inlier_match_count_median': 936.0, 'inlier_ratio_median': 1.0, 'parallax_proxy_median': 0.2406748538215955, 'low_parallax_fraction': 0.37527593818984545, 'near_static_fraction': 0.7019867549668874, 'parallax_gt_tmag_correlation': 0.4137399575713244, 'inlier_ratio_observability_correlation': -0.38221286193326093, 'matched_keypoint_count_observability_correlation': -0.27259358677478623, 'low_parallax_flag_vs_small_motion': 0.09492273730684327, 'near_static_flag_vs_small_motion': 0.24061810154525387, 'flow_magnitude_vs_gt_tmag_correlation': 0.44193559995616183, 'correspondence_too_sparse': False}

## observability gate
{'observable_edge_count': 135, 'observable_edge_fraction': 0.2980132450331126, 'unobservable_edge_count': 318, 'unobservable_edge_fraction': 0.7019867549668874, 'small_motion_observable_count': 5, 'small_motion_unobservable_count': 109, 'small_motion_observable_fraction': 0.043859649122807015, 'signed_direction_supervision_reliable_fraction': 0.2980132450331126, 'observable_edge_fraction_delta_vs_s5e11': 0.271523178807947, 'reliable_supervision_delta_vs_s5e11': 0.271523178807947}

## small-motion observability comparison vs S5E11
{'s5e11_small_motion_observable_fraction': 0.0, 's5e12_small_motion_observable_fraction': 0.043859649122807015}

## reliable signed direction supervision fraction
{'s5e11': 0.026490066225165563, 's5e12': 0.2980132450331126}

## correlation audit
{'experiment': 'S5E12_real_correspondence_feature_gate', 'edge_count': 453, 'correspondence_success_count': 453, 'correspondence_success_fraction': 1.0, 'filtered_match_count_median': 950.0, 'inlier_match_count_median': 936.0, 'inlier_ratio_median': 1.0, 'parallax_proxy_median': 0.2406748538215955, 'low_parallax_fraction': 0.37527593818984545, 'near_static_fraction': 0.7019867549668874, 'parallax_gt_tmag_correlation': 0.4137399575713244, 'inlier_ratio_observability_correlation': -0.38221286193326093, 'matched_keypoint_count_observability_correlation': -0.27259358677478623, 'low_parallax_flag_vs_small_motion': 0.09492273730684327, 'near_static_flag_vs_small_motion': 0.24061810154525387, 'flow_magnitude_vs_gt_tmag_correlation': 0.44193559995616183, 'correspondence_too_sparse': False}

## exported artifacts
{'dependency_check_json': 'external_baselines/results/s5e12_feature_gate/dependency_check.json', 'correspondence_features_json': 'external_baselines/results/s5e12_feature_gate/correspondence_features.json', 'correspondence_quality_json': 'external_baselines/results/s5e12_feature_gate/correspondence_quality_audit.json', 'essential_geometry_json': 'external_baselines/results/s5e12_feature_gate/essential_geometry_audit.json', 'observability_gate_json': 'external_baselines/results/s5e12_feature_gate/observability_gate.json', 'correspondence_features_csv': 'external_baselines/results/s5e12_feature_gate/correspondence_features.csv'}

## blockers
['strict essential geometry 仍受 pinhole intrinsics 缺失限制。']

## validation results
{'verify_final_candidate': 'PASS', 'project_health_check': 'PASS', 's6_eval_only': 'PASS', 'unittest': 'PASS'}

## caveats
- S5E12 是 feature gate，不替换 S5 locked。
- camera intrinsics 缺失时，strict essential geometry 必须诚实标记 unavailable。
