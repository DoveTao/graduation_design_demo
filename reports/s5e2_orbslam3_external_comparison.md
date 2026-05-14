# S5E2 与 ORB-SLAM3 external comparison

## 执行摘要

S5E2 是 experimental adjacent_dense candidate，目标是解决 S5E1 的 selected-only blocker。它与 ORB-SLAM3 的比较只能作为 diagnostic comparison，不替代 official S5 locked result。

## S5E2

- coverage = 1.0
- none = {'ate': 51.706643679614736, 'drift': 0.25372271542314084, 'path_ratio': 3.555778481577791, 'status': 'ok', 'num_matched_poses': 454, 'tracking_success_rate': 1.0}
- se3 = {'ate': 23.713532954995145, 'drift': 0.2875116253385571, 'path_ratio': 3.555778481577791, 'status': 'ok', 'num_matched_poses': 454, 'tracking_success_rate': 1.0}
- sim3 = {'ate': 4.097680633241629, 'drift': 0.1305592533014556, 'path_ratio': 3.555778481577791, 'status': 'ok', 'num_matched_poses': 454, 'tracking_success_rate': 1.0}
- component_metrics = {'rot_mean_deg': 0.9134395040767528, 'tdir_mean_deg': 51.47429479273258, 'tdir_abs_mean_deg': 46.07968707914153, 'tmag_median_ratio': 32.77577273937451, 'tmag_p90_ratio': 70.60597396909188, 'path_ratio': 3.5557784814144453, 'long_run_overscaling_like_s5d11': True}

## ORB-SLAM3

- coverage = 273/454
- none = {'ate': 11.46685113827757, 'drift': 0.03737053832593786, 'path_ratio': 0.2998258665660257}
- se3 = {'ate': 0.30854441069248173, 'drift': 0.0374760642069601, 'path_ratio': 0.2998258665660257}
- sim3 = {'ate': 0.224292165986624, 'drift': 0.06452708470276013, 'path_ratio': 0.2998258665660257}

## interpretation

S5E2 已建立 traceable dense pipeline，但 aligned ATE 尚未优于 restored S5 dense diagnostic。 与 ORB-SLAM3 se3 ATE 的差距为 23.404988544302665。 S5E2 path_ratio 为 3.5557784814144453，ORB-SLAM3 path_ratio 为 0.2998258665660257。

## recommendations

S5E2 若要接近 ORB-SLAM3，需要把 minimal baseline 升级为真正视觉 backbone，并继续保持 full edge provenance、no GT leakage 和 held-out evaluation。

## caveats

- S5E2 是 experimental candidate。
- 不替代 official S5 locked result。
- S5 locked metrics/policy unchanged。
- ORB-SLAM3 是 external strong baseline。
