# S5E3 scale calibrated adjacent dense report

## 执行摘要

S5E3 基于 S5E2 的 full traceable adjacent-dense pipeline，新增 direction head 和 log-magnitude calibration head，并继承 S5E2 的低 rot 输出。本轮 `final_classification = S5E3_TMAG_IMPROVED_TDIR_STILL_BAD`。

## S5E2 failure recap

S5E2 已有 453 条 direct adjacent predictions，但 `tmag_median_ratio=32.77577273937451`、`path_ratio=3.5557784814144453`。worst 20 edges 不单独主导全部路径，tmag over-scaling 更像是系统性小位移放大问题。

## S5E3 方法设计

S5E3 采用 multi-head adjacent regressor：rotation head 继承 S5E2，translation direction head 单独回归单位方向，translation magnitude head 回归 `log(||t||)`，并用 train split 的 magnitude quantile 做保守 clipping。

## scale calibration / tmag loss 说明

`tmag_log` 使用 train split 的 adjacent labels 训练；`scene01/seq03` GT 只在本报告的 diagnostics/evaluation 阶段使用。

## rot / tdir / tmag gating 说明

S5E3 不以 ATE 单独判定成功；gate 同时检查 rot、tdir、tdir_abs、tmag、path_ratio 和 sim3 ATE。

## training 结果

- classification = S5E3_TRAINING_SMOKE_ONLY
- num_train_pairs = 707
- num_val_pairs = 79

## traceable dense export coverage

- coverage = 1.0
- all_edges_traceable = True
- direct_adjacent_prediction_edges = 453

## component metrics

- rot_mean/median/p90 = 0.9134395040767528 / 0.797895569319923 / 1.3218302317546538
- tdir_mean/median/p90 = 135.28985476811536 / 147.8383695326091 / 168.1261181695304
- tdir_abs_mean/median/p90 = 37.14982041104558 / 32.16163046739092 / 70.84358920466546
- tdir_mean_cosine = -0.6180581770365583
- tmag_median/mean/p90/p95 = 12.109093390318419 / 19.468279568761744 / 48.301946228184384 / 73.00488324816631
- path_ratio = 2.1345479454214416

## external evaluator none/se3/sim3

- none = {'ate': 30.682466118163305, 'drift': 0.24466560946041588, 'path_ratio': 2.134547945404069, 'status': 'ok', 'num_matched_poses': 454, 'tracking_success_rate': 1.0}
- se3 = {'ate': 10.774498124607831, 'drift': 0.1896426043517595, 'path_ratio': 2.134547945404069, 'status': 'ok', 'num_matched_poses': 454, 'tracking_success_rate': 1.0}
- sim3 = {'ate': 3.9115097570948705, 'drift': 0.12574760443097996, 'path_ratio': 2.134547945404069, 'status': 'ok', 'num_matched_poses': 454, 'tracking_success_rate': 1.0}

## 与 S5E2 比较

- improvement_vs_s5e2 = {'rot_improved_or_preserved': True, 'tdir_improved': False, 'tdir_abs_improved': True, 'tmag_improved': True, 'path_ratio_improved': True, 'sim3_ate_improved': True, 'overall_geometry_improved': True}

## 与 ORB-SLAM3 比较

ORB-SLAM3 仍是 external strong baseline。S5E3 的优势是 full coverage；如果 aligned ATE 仍远高于 ORB-SLAM3，则不能声称接近 ORB-SLAM3。

## 是否更接近 ORB-SLAM3

S5E3 full coverage 保留，但 aligned ATE/tdir 与 ORB-SLAM3 仍有差距。

## rot / tdir 是否仍是主要差距

rot 由 S5E2 继承，通常保持较好；tdir 若仍高于目标阈值，则仍是主要几何差距。

## 下一步建议

1. 用更强视觉 backbone 替换 image-statistics features。
2. 引入 sequence-level path loss，但继续禁止 `scene01/seq03` GT 参与训练。
3. 若要借助 ORB-SLAM3 轨迹做蒸馏，应另开 S5E4 并明确 distillation protocol。

## caveats

- S5E3 是 experimental candidate。
- 不替代 official S5 locked result。
- S5 locked metrics/policy unchanged。
- ORB-SLAM3 是 external strong baseline。
