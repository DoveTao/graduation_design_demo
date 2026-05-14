# S5E1 traceable adjacent dense candidate 报告

## 执行摘要

S5E1 已建立 experimental adjacent-dense 诊断链路，但当前 final S5 仍无法合法导出 453 条 direct adjacent predictions。导出结果仅包含 132 条 `selected_prediction` 和 321 条 `unavailable` provenance，因此本轮不生成可评估的完整 dense TUM。

`final_classification = S5E1_ADJACENT_DENSE_EXPORT_BLOCKED`。

## 为什么需要 S5E1

S5D 系列显示，official S5 locked result 与 restored dense artifact 必须分层理解：前者是正式结果，后者只是 diagnostic artifact。S5E1 的目标是为未来实验候选提供每条 adjacent edge 可追溯的模型预测路径，并用 geometry losses 抑制 restored dense artifact 中出现的 long-run `tmag` over-scaling。

## S5D13 selected-only blocker 回顾

S5D13 的结论是 `selected_only`：132 条 selected_k1 预测可追溯，321 条 dense adjacent edge provenance unavailable，`can_generate_453_adjacent_edges = false`。

## 新增 adjacent-dense prediction path / training path

`tools/export_s5e1_adjacent_dense_predictions.py` 生成完整 edge provenance map；`tools/train_s5e1_geometry_candidate.py` 记录 experimental training design 和 blocked 原因；`tools/evaluate_s5e1_traceable_dense.py` 汇总 coverage、component metrics 和 external evaluator 可用性。

## 使用的 loss 和 config

- config = `configs/s5e1_geometry_candidate.yaml`
- `SO(3) geodesic`: true
- `tdir`: true
- `tmag log`: true
- `path length consistency`: true
- `short-window consistency`: true

## traceable dense export coverage

- trajectory_path = `external_baselines/results/s5e1_traceable_dense/scene01_seq03_s5e1_traceable_dense_tum.txt`
- edge_provenance = `external_baselines/results/s5e1_traceable_dense/edge_provenance.jsonl`
- num_poses = 0
- num_edges = 453
- coverage = 0.0
- all_edges_traceable = False

## component metrics

- rot_mean_deg = 20.835182098696297
- tdir_mean_deg = 99.54956140857581
- tdir_abs_mean_deg = 41.117886773689875
- tmag_median_ratio = 0.9819414718338607
- tmag_p90_ratio = 5.680931529985728
- path_ratio = None

## external evaluator none/se3/sim3 结果

- none = {'ate': None, 'drift': None, 'path_ratio': None}
- se3 = {'ate': None, 'drift': None, 'path_ratio': None}
- sim3 = {'ate': None, 'drift': None, 'path_ratio': None}

## 与 ORB-SLAM3 对比

S5E1 没有完整 traceable dense trajectory，因此不能与 ORB-SLAM3 做有效 ATE gap 对比。ORB-SLAM3 的 `se3` ATE 0.30854441069248173 和 `sim3` ATE 0.224292165986624 仍作为 external strong baseline 参考。

## 是否接近 ORB-SLAM3

不能判断；本轮的主要结果是发现 adjacent_dense export 仍被 selected-only provenance 阻塞。

## 失败或不足原因

当前仓库可合法追溯的 final S5 输出仍是 selected_k1 pairwise artifact。没有 direct adjacent-dense model output，也没有可审计的 documented fill logic，因此不能生成 454-pose / 453-edge traceable dense candidate。

## 下一步建议

1. 在 experimental config 下新增 direct adjacent pair dataloader 和 inference hook。
2. 增加 `adjacent_dense_pose_head` 或等价输出接口，确保每个 adjacent edge 都有模型预测 provenance。
3. 训练时启用 `SO(3) geodesic`、`tdir`、`tmag log`、`path length consistency`、`short-window consistency`，并监控 long-run over-scaling。
4. 生成完整 TUM 后再运行 none / se3 / sim3 external evaluator。

## Caveats

- S5E1 是 experimental candidate。
- S5E1 不替代 official S5 locked result。
- S5 locked metrics/policy unchanged。
- ORB-SLAM3 仍是 external strong baseline。
- no GT used for prediction；GT 仅用于 diagnostics/evaluation。
