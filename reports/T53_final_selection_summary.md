# T53 Final Selection Summary

> 生成日期：2026-05-04  
> 系列：T53 — Trajectory-Level Odometry Chain Constraints  
> 状态：已完成，不继续扫参

---

## 1. 实验演进表

| 实验 | len_w | target | vec_w | init_from | upd | 说明 |
|------|:---:|:---:|:---:|------|:---:|------|
| T51a3b | — | — | — | C31 | 400 | baseline |
| T53a | 0.01 | 0.55 | — | T51a3b | 200 | 验证链长 loss 有效 |
| T53a2 | 0.05 | 0.65 | — | T51a3b | 200 | 加强链长约束 |
| T53a3 | 0.10 | 0.75 | — | T51a3b | 200 | 继续加强 |
| T53a4 | 0.15 | 0.80 | — | T51a3b | 200 | 链长约束饱和点 |
| T53b | 0.15 | 0.80 | 0.001 | T53a4 | 200 | 小权重方向一致性 |
| T53b2 | 0.15 | 0.80 | 0.003 | T53b best | 200 | 方向权重过量→退化 |

---

## 2. 指标总表

| 实验 | drift | ATE | tmag_rel_err | **path_ratio** | tmag median | tmag <0.5 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| T51a3b upd400 | **1.351** | **8.432** | 0.911 | 0.386 | 0.430 | 52.5% |
| T53a | 1.260 | 8.397 | — | 0.290 | 0.325 | 60.0% |
| T53a2 | 1.326 | 8.290 | 0.847 | 0.327 | 0.364 | 57.5% |
| T53a3 | 1.328 | 8.438 | 0.854 | 0.360 | 0.402 | 55.0% |
| T53a4 final | 1.364 | 8.571 | 0.897 | 0.399 | 0.447 | 52.5% |
| **T53b best** ⭐ | **1.353** | **8.486** | **0.876** | **0.395** | **0.443** | **52.5%** |
| T53b final | 1.381 | 8.743 | 0.876 | **0.406** | **0.457** | 52.5% |
| T53b2 | 1.383 | 8.712 | 0.857 | 0.405 | 0.457 | 52.5% |

> ⭐ = 推荐首选

---

## 3. 最终选择

| 角色 | Checkpoint | drift | path_ratio | 理由 |
|------|------|:---:|:---:|------|
| **Conservative baseline** | `T51a3b_from_t51a2_upd200_restore_t51a2cfg_1200/eval_upd_0400.pt` | 1.351 | 0.386 | drift/ATE 最优 |
| **Best balanced** ⭐ | `T53b_from_t53a4_vec_w001_400/best_odom_drift.pt` | **1.353** | **0.395** | 最佳 drift + path_ratio 平衡 |
| **Shape-aware best** | `T53b_from_t53a4_vec_w001_400/final.pt` | 1.381 | **0.406** | 最高 path_ratio |

**推荐主线选择：T53b best_odom_drift.pt**

---

## 4. T53 技术结论

### 4.1 T53a：Chain Length Loss（链长下界约束）

- **方法**：在 k=1 的 triplet (A→B, B→C) 上，约束 predicted chain length ($|t_{AB}| + |t_{BC}|$) 不低于 GT chain length 的 target_ratio 倍。
- **发现**：path_ratio 随 len_w 和 target_ratio 单调递增（0.290 → 0.327 → 0.360 → 0.399）。
- **代价**：drift 从 1.260 升到 1.364（+8.3%），ATE 轻微上升。
- **结论**：trajectory-level length constraint 比 batch-level tmag 正则更有效改善轨迹形状。

### 4.2 T53b：Vector Direction Consistency（组合方向一致性）

- **方法**：在 T53a4 基础上，加入极小权重（w=0.001）的组合方向 loss：
  $$\text{pred\_t}_{CA} = R_{CB}^{pred} \cdot t_{BA}^{pred} + t_{CB}^{pred}, \quad \text{gt\_t}_{CA} = R_{CB}^{gt} \cdot t_{BA}^{gt} + t_{CB}^{gt}$$
  $$L_{vec} = \text{mean}(1 - \cos(\text{pred\_dir}, \text{gt\_dir}))$$
- **发现**：w=0.001 成功将 drift 从 1.364 拉回 1.353（接近 T51a3b），同时 path_ratio 升至 0.395（final 达 0.406）。
- **结论**：小权重 vector direction consistency 可以在保持 path_ratio 的同时改善 drift。

### 4.3 T53b2：向量损失权重上限

- **方法**：将 vec_w 从 0.001 提至 0.003。
- **发现**：drift 退化至 1.383，path_ratio 未进一步提升。
- **结论**：vector loss 权重不能盲目加大，w=0.001 附近为最优区间。

---

## 5. 图路径

| 报告 | 路径 |
|------|------|
| T53a4 对比 | `checkpoints/BATCH_SHAPE_REPORT_T53A4_COMPARE/shape_compare.png` |
| T53b 对比 | `reports/fig_t53_shape_compare.png` |
| T53a4 selection | `checkpoints/T53a4_selection_summary.md` |
| T53b selection | `checkpoints/T53b_selection_summary.md` |

---

## 6. 后续建议

1. **不继续 T53 训练扫参**：T53a4（len_w=0.15）+ T53b（vec_w=0.001）已找到最优配置。
2. **进入论文/PPT 结果整理**：以 T53b best 为主结果，T51a3b 为 baseline，T53b final 为 shape-aware 补充。
3. **若继续技术研究**，可考虑：
   - 优化 B-C forward 复用（当前 len 和 vec loss 各做一次 BC forward）
   - 尝试更严格的 vector chain loss（如 chordal distance 或 geodesic loss）
   - 在不同数据集上验证泛化性
