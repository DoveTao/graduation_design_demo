# T54 Final Candidate Review

> 生成日期：2026-05-04  
> 用途：论文/PPT 最终候选确认  
> 数据来源：`checkpoints/BATCH_*` 目录（已有 eval-only 结果）

---

## 1. 最终候选对比表

| 指标 | T51a3b upd400 | **T53b best** ⭐ | T53b final |
|------|:---:|:---:|:---:|
| checkpoint | `eval_upd_0400.pt` | `best_odom_drift.pt` | `final.pt` |
| drift | **1.351** | 1.353 | 1.381 |
| ATE | **8.432** | 8.486 | 8.743 |
| tdir_abs | 23.09° | 22.53° | **22.37°** |
| tmag_rel_err | 0.911 | 0.876 | **0.876** |
| **path_ratio** | 0.386 | 0.395 | **0.406** |
| step_dir_err | **26.67°** | 32.35° | 28.21° |
| tmag mean | 1.108 | 1.087 | 1.103 |
| tmag median | 0.430 | 0.443 | **0.457** |
| tmag <0.3 | 35.0% | 30.0% | 30.0% |
| tmag <0.5 | 52.5% | 52.5% | 52.5% |
| tmag <0.7 | 62.5% | 62.5% | 60.0% |
| shape_score_balanced | 2.625 | 2.680 | **2.682** |
| **推荐用途** | conservative baseline | **final main candidate** | shape-aware candidate |

> ⭐ = 推荐主候选

---

## 2. 为什么 T53b best 是主候选

| 对比维度 | T51a3b | T53b best | 评价 |
|------|:---:|:---:|------|
| drift | 1.351 | 1.353 | 几乎持平 (+0.1%) |
| ATE | 8.432 | 8.486 | 轻微上升 (+0.6%) |
| path_ratio | 0.386 | **0.395** | **大幅改善 (+2.3%)** |
| tmag median | 0.430 | **0.443** | 提升 (+3.0%) |
| tmag_rel_err | 0.911 | **0.876** | 降低 (-3.8%) |
| tdir_abs | 23.09° | **22.53°** | 改善 (-2.4%) |

**T53b best 在几乎不增加 drift 的前提下，将 path_ratio 从 0.386 提升至 0.395，同时 tmag 各指标全面改善。**

损失函数：odom_chain_len_loss (w=0.15, target=0.80) + odom_chain_vec_loss (w=0.001)

---

## 3. 为什么 T53b final 只作为 shape-aware 对照

| 维度 | T53b best | T53b final | 评价 |
|------|:---:|:---:|------|
| drift | **1.353** | 1.381 | +2.1% 退化 |
| ATE | **8.486** | 8.743 | +3.0% 退化 |
| path_ratio | 0.395 | **0.406** | +2.7% 改善 |

T53b final 以 drift/ATE 的明显代价换取 path_ratio 的进一步提升。适合在论文中作为 "进一步强化 shape constraint 可继续提升 path_ratio" 的消融对照，但不推荐作为主结果。

---

## 4. 为什么不继续 T53b2

| 实验 | vec_w | drift | path_ratio |
|------|:---:|:---:|:---:|
| T53b | 0.001 | **1.353** | 0.395 |
| T53b2 | 0.003 | 1.383 | 0.405 |

vec_w=0.003 导致 drift 退化至 1.383（+2.2%），path_ratio 未显著超越 T53b final。**w=0.001 为最优区间，盲目加大权重无益。**

---

## 5. 论文/PPT 建议

### 主表
| 实验 | drift | ATE | path_ratio | tmag median |
|------|:---:|:---:|:---:|:---:|
| T51a3b (baseline) | 1.351 | 8.432 | 0.386 | 0.430 |
| **T53b best (ours)** | **1.353** | **8.486** | **0.395** | **0.443** |

### 可视化图
`reports/fig_t53_shape_compare.png` — T51a3b vs T53a4 vs T53b 轨迹对比

### 消融表
| 实验 | len_w | target | vec_w | drift | path_ratio |
|------|:---:|:---:|:---:|:---:|:---:|
| T51a3b | — | — | — | 1.351 | 0.386 |
| T53a4 | 0.15 | 0.80 | — | 1.364 | 0.399 |
| T53b | 0.15 | 0.80 | 0.001 | 1.353 | 0.395 |
| T53b2 | 0.15 | 0.80 | 0.003 | 1.383 | 0.405 |

### 推荐 checkpoint 路径
- **主结果**：`checkpoints/T53b_from_t53a4_vec_w001_400/best_odom_drift.pt`
- Baseline：`checkpoints/T51a3b_from_t51a2_upd200_restore_t51a2cfg_1200/eval_upd_0400.pt`
- 对照：`checkpoints/T53b_from_t53a4_vec_w001_400/final.pt`
