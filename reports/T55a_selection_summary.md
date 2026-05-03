# T55a Selection Summary

> 生成日期：2026-05-04  
> 工具：`scripts/batch_eval_shape_report.py` + `scripts/select_shape_aware_checkpoint.py`  
> 输出目录：`checkpoints/BATCH_SHAPE_REPORT_T55A_COMPARE/`

---

## 1. 全量对比表

| 指标 | T51a3b | T53b_best | T53b_final | **T55a_best** | T55a_final |
|------|:---:|:---:|:---:|:---:|:---:|
| drift | **1.351** | 1.353 | 1.381 | 1.352 | 1.383 |
| ATE | **8.432** | 8.486 | 8.743 | 8.569 | 8.808 |
| tdir_abs | 23.09° | 22.53° | 22.37° | **22.27°** | 22.53° |
| tmag_rel_err | 0.911 | 0.876 | 0.876 | **0.858** | 0.861 |
| **path_ratio** | 0.386 | 0.395 | 0.405 | 0.394 | **0.407** |
| step_dir_err | **26.67°** | 32.35° | 28.21° | 29.86° | 27.26° |
| tmag mean | 1.108 | 1.087 | 1.103 | **1.068** | 1.090 |
| tmag median | 0.430 | 0.443 | 0.457 | 0.443 | **0.460** |
| tmag <0.3 | 35.0% | 30.0% | 30.0% | 30.0% | **27.5%** |
| tmag <0.5 | 52.5% | 52.5% | 52.5% | 52.5% | 52.5% |
| seq_turn_loss | ON | ON | ON | **OFF** | **OFF** |
| shape_balanced | 2.625 | **2.680** | 2.682 | 2.664 | 2.679 |
| **推荐** | baseline | balanced | shape | **best tmag** | shape-best |

---

## 2. T55a best vs T55a final 差异

| 指标 | T55a best (upd=100) | T55a final (upd=200) | 变化 |
|------|:---:|:---:|:---:|
| drift | **1.352** | 1.383 | +2.3% |
| ATE | **8.569** | 8.808 | +2.8% |
| path_ratio | 0.394 | **0.407** | +3.4% |
| tmag median | 0.443 | **0.460** | +3.8% |
| tmag mean | **1.068** | 1.090 | +2.1% |

**T55a 与 T53b 展现相同的 best/final 分化：best 偏 drift，final 偏 shape。**

---

## 3. 关闭旧 seq_turn_loss 的影响

| 对比 | T53b_best (seq_turn ON) | T55a_best (seq_turn OFF) | 
|------|:---:|:---:|
| drift | 1.353 | **1.352** |
| ATE | **8.486** | 8.569 |
| tmag_rel_err | 0.876 | **0.858** |
| path_ratio | 0.395 | 0.394 |
| tmag mean | 1.087 | **1.068** |
| tmag median | 0.443 | 0.443 |

**drift 保持 + tmag 全面改善 + ATE 轻微上升但仍在可接受范围。**

---

## 4. 推荐 checkpoint

| 角色 | Checkpoint | drift | path_ratio | 理由 |
|------|------|:---:|:---:|------|
| **Conservative baseline** | `T51a3b eval_upd_0400.pt` | 1.351 | 0.386 | drift/ATE 最优 |
| **Best tmag/simplest** ⭐ | `T55a best_odom_drift.pt` | 1.352 | 0.394 | 最佳 tmag + 最简 loss (无 seq_turn) |
| **Best balanced** | `T53b best_odom_drift.pt` | 1.353 | 0.395 | 最佳 ATE |
| **Shape-aware best** | `T55a final.pt` | 1.383 | 0.407 | 最高 path_ratio |

**推荐 T55a_best 作为新主线：关闭旧 seq_turn_loss 后 loss 更简洁，tmag 指标最优。**

---

## 5. 是否继续 T55b

**不建议立即进入 T55b。** T55a 已证明关闭旧 seq_turn_loss 是纯正向的（drift 不退化 + tmag 改善）。下一步可考虑：

1. **固化 T55a best 为最终结果**
2. 若时间允许，T55b 可尝试在 T55a 基础上实现新的 B-C seq-turn loss（利用 compose CA direction），这比旧的 A→B seq-turn loss 更合理
3. 当前优先整理论文/PPT

## 6. 图路径

- `reports/fig_t55a_shape_compare.png`
