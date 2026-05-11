# S5E1 与 ORB-SLAM3 external comparison

## 执行摘要

S5E1 本轮没有生成合法完整 traceable dense trajectory，因此不能进行有效的 none / se3 / sim3 ATE 对比。ORB-SLAM3 仍作为 verified external strong baseline；S5E1 当前结果是 adjacent_dense export blocked。

## 对比输入

- S5E1 checkpoint = `checkpoints/S5E1_traceable_adjacent_dense_candidate.json`
- ORB-SLAM3 trajectory = `external_baselines/results/orbslam3_fisheye_cam0/scene01_seq03_est_tum.txt`

## ORB-SLAM3 reference

- coverage = 273/454
- none = {'ate': 11.46685113827757, 'drift': 0.03737053832593786, 'path_ratio': 0.2998258665660257}
- se3 = {'ate': 0.30854441069248173, 'drift': 0.0374760642069601, 'path_ratio': 0.2998258665660257}
- sim3 = {'ate': 0.224292165986624, 'drift': 0.06452708470276013, 'path_ratio': 0.2998258665660257}

## S5E1 reference

- adjacent_dense_available = False
- coverage = 0.0
- external_eval = {'none': {'ate': None, 'drift': None, 'path_ratio': None}, 'se3': {'ate': None, 'drift': None, 'path_ratio': None}, 'sim3': {'ate': None, 'drift': None, 'path_ratio': None}, 'attempted': False, 'blocked_reason': 'No complete legal traceable dense TUM was exported.'}

## interpretation

ORB-SLAM3 在 tracked subset 上 aligned ATE 很低，但 coverage 只有 273/454。S5E1 的目标是未来实现 full coverage traceable dense prediction；当前由于没有合法 453-edge direct adjacent source，不能声称比 restored S5 dense artifact 更接近 ORB-SLAM3。

## recommendations

下一步应优先实现 experimental direct adjacent prediction head 和 dataloader，然后再运行 same external evaluator。只有当 S5E1 输出完整、可追溯、不使用 GT 生成 prediction 的 TUM 后，才适合计算 ATE gap 与 path_ratio 对比。

## Caveats

- S5E1 是 experimental candidate。
- S5E1 不替代 official S5 locked result。
- S5 locked metrics/policy unchanged。
- ORB-SLAM3 仍是 external strong baseline。
