# S5E11 geometric observability audit

## 几何特征可用性
{'correspondence_features_available': False, 'optical_flow_features_available': False, 'essential_matrix_features_available': False, 'fallback_proxy_features_used': True}

## observability audit
{'observable_edge_count': 18, 'observable_edge_fraction': 0.022900763358778626, 'unobservable_edge_count': 768, 'unobservable_edge_fraction': 0.9770992366412213, 'small_motion_observable_fraction': 0.0, 'small_motion_unobservable_fraction': 1.0, 'signed_direction_supervision_reliable_fraction': 0.022900763358778626, 'signed_direction_loss_weight_mean': 0.04980916030534352, 'excluded_direction_edge_count': 681, 'excluded_direction_edge_fraction': 0.8664122137404581}

## bucket summary
{'near_static_unobservable': {'count': 681, 'fraction': 0.8664122137404581, 'gt_tmag_median': 0.009519453765840966, 'gt_tmag_p95': 0.3197845212013866, 'matched_keypoint_count_median': 5107.0, 'inlier_ratio_median': 1.0, 'parallax_proxy_median': 0.0}, 'low_parallax_unreliable': {'count': 70, 'fraction': 0.089058524173028, 'gt_tmag_median': 0.045827310747346195, 'gt_tmag_p95': 0.5398429996137776, 'matched_keypoint_count_median': 5114.0, 'inlier_ratio_median': 1.0, 'parallax_proxy_median': 1.0}, 'normal_observable': {'count': 5, 'fraction': 0.006361323155216285, 'gt_tmag_median': 0.07662023231497106, 'gt_tmag_p95': 0.12139272069153578, 'matched_keypoint_count_median': 5119.0, 'inlier_ratio_median': 1.0, 'parallax_proxy_median': 3.0}, 'large_motion_observable': {'count': 13, 'fraction': 0.01653944020356234, 'gt_tmag_median': 0.22396198338110934, 'gt_tmag_p95': 0.5137360026277773, 'matched_keypoint_count_median': 5109.0, 'inlier_ratio_median': 1.0, 'parallax_proxy_median': 3.0}, 'outlier_correspondence': {'count': 17, 'fraction': 0.021628498727735368, 'gt_tmag_median': 0.1466220310867368, 'gt_tmag_p95': 0.41409347912549554, 'matched_keypoint_count_median': 5071.0, 'inlier_ratio_median': 1.0, 'parallax_proxy_median': 1.0}}

## 说明
S5E11 先检查显式 correspondence / parallax / essential proxy 特征是否足以支撑 signed direction 监督。
