# O49 Translation Magnitude & Trajectory Shape Diagnosis

Date: 2026-05-02  
Commits: `801cd06` (T50 probe runner), `c2e6ca1` (tmag bucket inspector), `e3448b4` (shape debug inspector)  
Status: Diagnostic phase complete. No further training recommended until magnitude head is redesigned.

---

## 1. Background

O49m is the current coarse-only seq-turn pair-loss mainline.  
O39 is the pre-seq-turn baseline (no `use_seq_turn_loss`).  
Both were trained with identical core architecture (`D=256, Nc=192, coarse_interaction=True, fine_stage=False`).

Goal: understand why O49m achieves low odometry drift (~1.22) but exhibits severe trajectory compression (predicted path ~2 m vs. ground truth ~10 m).

---

## 2. O39 vs O49m3: Shape Comparison

Eval-only trajectory debug on `best_odom_drift.pt` / `best_smallk_odom.pt`, k=1 odometry chain of 40 steps.

| Metric | O39 (no seq-turn) | O49m3 (seq-turn) | Δ |
|---|---:|---:|---|
| **drift** | 1.3164 | **1.2234** | −0.09 ✅ |
| **ATE** | 8.33 | **7.75** | −0.58 ✅ |
| **path_length_ratio** | **0.249** | **0.203** | −0.046 |
| pred_path_length | 2.54 m | 2.07 m | — |
| gt_path_length | 10.20 m | 10.20 m | same |
| pred_turn_sum | 55.8° | 69.1° | +13.3° |
| gt_turn_sum | 322.7° | 322.7° | same |
| turn_sum_abs_err | 266.9° | **253.5°** | −13.4° ✅ |
| mean_step_dir_err | 19.8° | **18.3°** | −1.5° ✅ |
| mean_turn_abs_err | 7.3° | **7.0°** | −0.3° ✅ |
| straightness_err | 0.035 | **0.019** | −0.016 ✅ |
| tdir_abs | 22.9° | 23.1° | +0.2° |

**Finding:** seq-turn pair loss improves direction, step-level error, and straightness.  
**But:** path length compression is **systemic**, not introduced by seq-turn — O39 already has `path_ratio=0.25`.

---

## 3. Translation Magnitude: Almost Constant

Per-step CSV from `odom_trajectory_steps_latest.csv` shows the model outputs nearly the same `tmag_pred` regardless of ground-truth baseline distance.

| step | gt_tmag (m) | O39 pred_tmag | O49m3 pred_tmag |
|---|---:|---:|---:|
| 0 | 0.026 | 0.066 | 0.054 |
| 1 | 0.127 | 0.066 | 0.054 |
| 2 | 0.219 | 0.066 | 0.054 |
| 3 | 0.658 | 0.066 | 0.054 |
| 4 | 0.454 | 0.066 | 0.054 |
| 5 | 0.283 | 0.066 | 0.054 |
| 6 | 0.335 | 0.066 | 0.054 |
| 7 | 0.439 | 0.066 | 0.054 |

**Finding:** `pred_tmag ≈ 0.05–0.07` for both models, completely ignoring `gt_tmag` variation (0.03 → 0.66).  
The magnitude head is essentially non-functional — it learned to output a constant.

### Per-k tmag_rel_err

| k | O39 | O49m3 |
|---|---:|---:|
| 1 | 0.762 | 0.695 |
| 2 | 0.673 | 0.705 |
| 5 | 0.841 | 0.870 |
| 10 | 0.905 | 0.923 |
| 20 | 0.894 | 0.914 |

`tmag_rel_err` worsens with k/dt, but stays in the 0.7–0.9 range even for k=1 — the constant output means error is high at all baselines.

---

## 4. O49m3 Bias Sweep: Post-hoc Affine Calibration

Eval-only on O49m3 `best_odom_drift.pt` with `use_tmag_affine_calib=True`, varying `tmag_affine_init_bias` (scale=1.0 fixed).

| bias | scale = exp(bias) | drift | ATE | path_ratio | pred_path |
|---|---:|---:|---:|---:|---:|
| 0.0 (identity) | 1.00 | **1.22** | **7.75** | 0.203 | 2.1 m |
| 0.7 | 2.01 | 1.34 | 8.63 | 0.409 | 4.2 m |
| 1.1 | 3.00 | 1.49 | 9.83 | 0.610 | 6.2 m |
| 1.4 | 4.06 | 1.69 | 11.36 | 0.823 | 8.4 m |
| **1.6** | **4.95** | 1.93 | 12.81 | **1.006** | 10.3 m |
| 1.8 | 6.05 | 2.32 | 14.69 | 1.228 | 12.5 m |

**Finding:** bias=1.6 (scale≈5×) perfectly matches `path_ratio≈1.0`.  
**But:** drift degrades from 1.22 → 1.93 (+58%), ATE from 7.75 → 12.81 (+65%).  
The "good" drift of O49m3 is partially an artifact of path compression — scaling up the magnitude exposes the true odometry error.

**Note:** `pred_turn_sum`, `mean_step_dir_err`, `mean_turn_abs_err` are **unchanged** across all biases — affine only scales magnitude, not direction.

---

## 5. T50a: Trainable Affine Calibration (FAILED)

Training from O49m3 checkpoint with `use_tmag_affine_calib=True`, `tmag_affine_init_bias=1.6`, 400 steps.

| Metric | Init | After 400 steps | Δ |
|---|---|---|---|
| `tmag_affine_scale` | 1.0000 | **1.0004** | +0.0004 |
| `tmag_affine_bias` | 1.6000 | **1.5996** | −0.0004 |
| `path_length_ratio` | — | 0.243 | — |
| `pred_tmag` | — | ~0.064 | — |
| `drift` | — | 1.253 | — |

**Finding:** affine parameters learned **almost nothing** in 400 steps.  
The model compensated by reducing raw `tc_mag` output through the coarse interaction weights, effectively neutralizing the affine amplification.

### Root Cause

```
Init:
  raw_tc_mag ≈ 0.05  →  affine(×5)  →  t_mag ≈ 0.25  →  tmag_loss HUGE
Training (gradient backprop):
  Path A: adjust affine scale/bias (2 params)          ← tiny gradient signal
  Path B: adjust coarse interaction weights (100k+ params) ← dominant gradient
  Model chooses Path B → raw_tc_mag drops to ~0.01
After training:
  raw_tc_mag ≈ 0.01  →  affine(×5)  →  t_mag ≈ 0.05  ← back to square one
```

This is a classic end-to-end training trap: 2 scalar parameters cannot compete with the full network's representational capacity for the same loss signal.

---

## 6. Key Conclusions

1. **Path compression is systemic**, not caused by seq-turn loss. O39 (no seq-turn) already has `path_ratio=0.25`.

2. **Magnitude head outputs near-constant values** (~0.05–0.07) regardless of baseline distance. The head has essentially not learned to use token features for distance regression.

3. **Post-hoc affine calibration works** — bias=1.6 (scale≈5×) can restore `path_ratio≈1.0`. But it exposes degraded drift/ATE, confirming the "good" low drift is partially a compression artifact.

4. **Trainable affine fails** — the network learns to counteract the affine through its main weights rather than adjusting the affine parameters.

5. **Seq-turn pair loss improves direction** — O49m3 beats O39 on step_dir_err, straightness, turn_abs_err, but magnitude problems persist.

6. **O39→O49m3 improvement is real but incomplete** — direction/rotation get better, magnitude gets marginally worse (`path_ratio` 0.25→0.20).

---

## 7. Next Steps (Recommended)

| Priority | Action | Rationale |
|---|---|---|
| ⭐⭐⭐ | Redesign magnitude head to be dt/k-conditioned | Current head ignores baseline distance entirely |
| ⭐⭐ | Add stronger magnitude supervision (e.g., scale-invariant log-loss, dt-bucketed weights) | Current `w_tmag=0.1` may be insufficient |
| ⭐ | Post-hoc affine as diagnostic tool for future experiments | bias=1.6 reliably reveals "true" drift |

---

## 8. What NOT to Do

- ❌ **O50a chain loss** — would add constraints on top of broken magnitude, likely making things worse
- ❌ **Fine stage** — fine tokens won't fix if coarse tokens already fail at magnitude regression
- ❌ **More T50 affine training** — proven ineffective in end-to-end setting
- ❌ **Further seq-turn weight sweeps** — direction is not the bottleneck; magnitude is

---

## 9. Diagnostic Tool Inventory

| Script | Commit | Purpose |
|---|---|---|
| `scripts/inspect_o49_shape_debug.py` | `e3448b4` | Per-experiment trajectory shape metrics |
| `scripts/inspect_tmag_buckets.py` | `c2e6ca1` | Per-k, per-dt, per-step tmag diagnostics |
| `scripts/run_t50_tmag_affine_probe.sh` | `801cd06` | T50 affine calibration launcher |
| `scripts/summarize_odom_cycle.py` | `8331469` | Multi-experiment ranking + selection |

---

*Report auto-generated from checkpoint artifacts. No training was run for this report.*
