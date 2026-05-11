# S5E2 adjacent dense candidate report

## 执行摘要

S5E2 新增了 experimental adjacent_dense 训练与推理路径：使用 train split 的 `scene01/seq01` 和 `scene01/seq02` 构建 adjacent image-pair dataset，训练一个 minimal image-statistics ridge pose regressor，并在 held-out `scene01/seq03` 上导出 453 条 direct adjacent predictions。该结果是 experimental candidate，不替代 official S5 locked result。

`final_classification = S5E2_TRACEABLE_DENSE_EXPORTED_FAILED_METRICS`。

## S5E1 blocker 回顾

S5E1 was blocked by missing engineering interfaces: no adjacent-dense dataloader/head/export path existed. The final S5 traceable artifact was selected_k1 only, so 321 adjacent edges remained unavailable.

## adjacent_dense dataset 构建

- adjacent_dataset_ready = True
- split_respected = True
- test_gt_used_for_training = False
- num_train_pairs = 707
- num_val_pairs = 79

## training/fine-tune 情况

- attempted = True
- classification = S5E2_TRAINING_SMOKE_ONLY
- checkpoint_dir = `checkpoints/S5E2_adjacent_dense_candidate`

## 新增模型/head/loss 说明

S5E2 使用 `minimal_image_statistics` 作为 experimental backbone，`adjacent_dense_pose_head` 由 closed-form ridge regression 实现。loss/diagnostic proxy 包括 `SO(3) geodesic`、`tdir`、`tmag log`、`path length consistency`、`short-window consistency`，并保留 small translation mask caveat。

## traceable adjacent dense export coverage

- trajectory_path = `external_baselines/results/s5e2_traceable_dense/scene01_seq03_s5e2_traceable_dense_tum.txt`
- edge_provenance = `external_baselines/results/s5e2_traceable_dense/edge_provenance.jsonl`
- num_poses = 454
- num_edges = 453
- direct_adjacent_prediction_edges = 453
- coverage = 1.0

## component metrics

- rot_mean_deg = 0.9134395040767528
- tdir_mean_deg = 51.47429479273258
- tdir_abs_mean_deg = 46.07968707914153
- tmag_median_ratio = 32.77577273937451
- tmag_p90_ratio = 70.60597396909188
- path_ratio = 3.5557784814144453

## external evaluator none/se3/sim3

- none = {'ate': 51.706643679614736, 'drift': 0.25372271542314084, 'path_ratio': 3.555778481577791, 'status': 'ok', 'num_matched_poses': 454, 'tracking_success_rate': 1.0}
- se3 = {'ate': 23.713532954995145, 'drift': 0.2875116253385571, 'path_ratio': 3.555778481577791, 'status': 'ok', 'num_matched_poses': 454, 'tracking_success_rate': 1.0}
- sim3 = {'ate': 4.097680633241629, 'drift': 0.1305592533014556, 'path_ratio': 3.555778481577791, 'status': 'ok', 'num_matched_poses': 454, 'tracking_success_rate': 1.0}

## 与 ORB-SLAM3 对比

ORB-SLAM3 仍是 external strong baseline。S5E2 的优势是 454/454 full coverage 与完整 edge provenance；主要差距是 aligned ATE 和局部方向/旋转误差仍由极简模型限制。

## 是否接近 ORB-SLAM3

如果 `se3` / `sim3` ATE 仍明显高于 ORB-SLAM3，则只能说明 S5E2 解决了 traceability/export blocker，还没有解决高精度几何估计问题。

## 失败原因或不足

S5E2 当前是 minimal baseline，不复用完整 S5 visual backbone，也没有进行长时间训练；它用于证明合法 adjacent_dense 训练/推理链路可行。

## 下一步建议

1. 将 minimal image-statistics backbone 替换为 existing S5 backbone 或更强视觉编码器。
2. 使用真实 mini-batch training 启用 `SO(3) geodesic`、`tdir`、`tmag log`、`path length consistency` 和 `short-window consistency`。
3. 保持 `scene01/seq03` 作为 held-out evaluation，不将其 GT 用于训练。

## caveats

- S5E2 是 experimental candidate。
- 不替代 official S5 locked result。
- S5 locked metrics/policy unchanged。
- ORB-SLAM3 是 external strong baseline。
- no GT leakage；`scene01/seq03` GT 只用于 evaluation / diagnostics。
