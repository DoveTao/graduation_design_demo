# T51 实验最终选择总结（不含重新训练）

## 1) 实验演进表

| 实验 | 关键说明 | drift | ATE | tmag_rel_err | path_ratio | tmag_ratio 统计(中位数 / <0.5) |
|---|---|---:|---:|---:|---:|---|
| T51a2 | O49m 对齐基线；`best_odom_drift=inf / best_odom_upd=-1`，里程计选模未稳定 | inf | N/A | N/A | N/A | 中位数≈0.6305 / 40.0% |
| T51a3 | 从 T51a2 继续，但配置未复用主线关键项，checkpoint 加载出现 `unexpected=['log_tmag_bias']` | inf（或 ~7.x） | N/A | N/A | N/A | N/A |
| T51a3b | 从 T51a3 继承并恢复主线配置后回到可用区间；shape-aware 主线候选 | 1.350982 @upd400；1.354279 @upd500 | 8.431564 @upd400；8.397043 @upd500 | 0.910864 @upd400；0.876205 @upd500 | 0.386013 @upd400；0.362376 @upd500 | 中位数≈0.380（T51a3b 最终）；<0.5=55% |
| T51a4 | `w_tmag_scale=0.01`（消融） | 1.233451 @upd100；1.307554 @upd600 | 7.803045 @upd100；8.238738 @upd600 | 0.781310 @upd100；0.726636 @upd600 | 0.289101 @upd100；0.264809 @upd600 | 中位数≈0.2982；<0.5=65% |
| T51a5 | `use_tmag_under_reg=True,w=0.02,target=0.35` | 1.274320 @upd100(best)；1.326563 @upd400(final) | 8.036951 @best_odom；8.360192 @final | 0.815884 @upd100(best)；0.768046 @final | 0.322886 @best_odom；0.311953 @final | 中位数≈0.3507；<0.5=60% |

## 2) 最终候选 checkpoint 表

| 候选 | checkpoint | 备注 | drift | ATE | tmag_rel_err | path_ratio | tmag_ratio（median / <0.5） |
|---|---|---|---:|---:|---:|---:|---|
| T51a3b upd400（shape-aware 主线） | `checkpoints/T51a3b_from_t51a2_upd200_restore_t51a2cfg_1200/eval_upd_0400.pt` | shape-aware 主线主候选 | 1.350982 | 8.431564 | 0.910864 | 0.386013 | ~0.380 / 55% |
| T51a3b upd500（best-drift 对照） | `checkpoints/T51a3b_from_t51a2_upd200_restore_t51a2cfg_1200/eval_upd_0500.pt` | 同目录 best_odom_drift 对齐点（`best_odom_drift.pt`） | 1.354279 | 8.397043 | 0.876205 | 0.362376 | ~0.3797 / 55% |
| T51a3b best-drift 对照 ckpt | `checkpoints/T51a3b_from_t51a2_upd200_restore_t51a2cfg_1200/best_odom_drift.pt` | 对应 `upd500` 里程计最优 | 1.354279 | 8.397043 | 0.876205 | 0.362376 | ~0.3797 / 55% |
| T51a3b upd500（额外对照） | `checkpoints/T51a3b_from_t51a2_upd200_restore_t51a2cfg_1200/eval_upd_0500.pt` | 与上对齐 | 1.354279 | 8.397043 | 0.876205 | 0.362376 | ~0.3797 / 55% |
| T51a4 upd100（负结果） | `checkpoints/T51a4_from_t51a3b_upd500_scale_w001_1200/best_odom_drift.pt`（`upd100`） | scale-reg 负结果 | 1.233451 | 7.803045 | 0.781310 | 0.289101 | ~0.2982 / 65% |
| T51a4 final（负结果） | `checkpoints/T51a4_from_t51a3b_upd500_scale_w001_1200/final.pt` | 最终快照 | 1.307554 | 8.238738 | 0.726636 | 0.264809 | ~0.2982 / 65% |
| T51a5 upd100（负结果） | `checkpoints/T51a5_from_t51a3b_upd400_under_w002_target035_800/best_odom_drift.pt` | under-reg 负结果 | 1.274321 | 8.036951 | 0.815884 | 0.322886 | ~0.3507 / 60% |
| T51a5 final（负结果） | `checkpoints/T51a5_from_t51a3b_upd400_under_w002_target035_800/final.pt` | 最终快照 | 1.326563 | 8.360192 | 0.768046 | 0.311953 | ~0.3507 / 60% |

## 3) 最终选择

- **最终主线 checkpoint**：`checkpoints/T51a3b_from_t51a2_upd200_restore_t51a2cfg_1200/eval_upd_0400.pt`（可配套 `checkpoints/T51a3b_from_t51a2_upd200_restore_t51a2cfg_1200/best_eval_joint.pt`）。
- **best-drift 对照**：`checkpoints/T51a3b_from_t51a2_upd200_restore_t51a2cfg_1200/best_odom_drift.pt`（数值与 `eval_upd_0500.pt` 对齐）。
- **不采用 T51a4/T51a5 的原因**：
  - T51a4：虽然 drift/ATE 在某些点改善，但 `path_ratio` 明显下滑（0.386→0.289 / 0.265），形状保持更差；说明 scale-reg 方向收益不足，且在路径形状约束上回撤。
  - T51a5：under-reg 试验降低了一些漂移和 tmag 相对误差，但仍未提升 shape-aware 指标；`path_ratio` 反降（≈0.312，低于 T51a3b upd400 的 0.386），且 `<0.5` 比例不低于 60%，未显著抑制欠估计。

## 4) 可视化

- `reports/fig_repro_shape_compare_clean.png`

## 5) 下一步建议

- **短期建议（冻结为当前结论）**：停止继续调 batch-level 的 tmag mean/under/scale 类正则；优先固化 shape-aware 选模逻辑与 T51a3b 主线的稳定训练范式。
- **长期建议**：若继续研究，优先引入 trajectory-level 或 odom-chain-level 的 path/shape 约束，而不是单步 tmag 损失（无论是均值型还是下分位约束型）。
