# T51: dt/k-Conditioned Translation Magnitude Head — Design Plan

Date: 2026-05-02  
Author: GitHub Copilot + User  
Status: Design only. No implementation, no training.

---

## 1. Background

Coarse-only O49m3 achieves drift=1.22 but `path_length_ratio=0.203` — predicted
path is only 20% of ground truth. The root cause is a near-constant magnitude
output (`pred_tmag ≈ 0.05–0.07`) that ignores the actual baseline distance
(`gt_tmag` varies 0.03→0.66 m across steps in a single odometry chain).

Diagnosis across O39, O49, T50, F0 has established:

- The problem is systemic (O39 has it too), not caused by seq-turn loss.
- Post-hoc affine calibration can fix path_ratio (bias=1.6 → path_ratio=1.0),
  but drift/ATE regress because the "good" drift was partially a compression artifact.
- Trainable affine calibration fails end-to-end: the network counteracts the
  affine by reducing its raw magnitude output through coarse interaction weights.
- Fine matching (F0 topk64) improves correspondence quality (epi_mass +41%) but
  does not fix magnitude.

This document proposes T51: a lightweight architectural change to make the
magnitude head **condition on baseline distance (dt_world)** and optionally
**frame interval (k)**, so it can learn to output scale-appropriate magnitudes.

---

## 2. Evidence Summary

| Experiment | pred_tmag | path_ratio | Key Finding |
|---|---|---|---|
| O39 eval-only shape | ~0.066 (constant) | 0.249 | Pre-seq-turn model already has magnitude problem |
| O49m3 eval-only shape | ~0.054 (constant) | 0.203 | Seq-turn improves direction but not magnitude |
| O49m3 bias=1.6 eval-only | ~0.25 (×5) | **1.006** | Affine can fix path_ratio as post-hoc calibration |
| O49m3 bias=1.6 drift | 1.93 | — | "True" drift without compression is ~1.9 |
| T50a trainable affine | ~0.064 | 0.243 | Network counteracts affine; learned bias≈init |
| F0 topk64 fine matching | ~0.05 | — | Better matching ≠ better magnitude |

### Why is pred_tmag constant?

The `TranslationMagnitudeHead` (interaction.py L188) pools token features via
mean+max+std and feeds them through an MLP to output a single `log_t_mag`:

```
mag_feat = mean(token_feat) || max(token_feat) || std(token_feat)  # [B, 3D]
log_t_mag = MLP(mag_feat)                                           # [B, 1]
```

This pooled representation **contains no explicit distance information**.
The model must infer baseline distance from token matching patterns alone,
but the matching is dominated by appearance similarity, not geometric scale.

**Hypothesis:** The pooling operation discards the scale signal. The MLP finds a
local optimum at a constant output that minimizes loss across the k=1–20 range
by outputting the training-set-mean magnitude.

---

## 3. Candidate Designs

### Design A: log(dt_world) Concatenation (Minimal)

Add `log(dt_world)` as an extra scalar input to the magnitude head MLP.

```
mag_feat = mean(token_feat) || max(token_feat) || std(token_feat)  # [B, 3D]
dt_feat  = log(dt_world.clamp(min=0.01))                            # [B, 1]
mag_feat = concat(mag_feat, dt_feat)                                 # [B, 3D+1]
log_t_mag = MLP(mag_feat)                                            # [B, 1]
```

**Pros:** Minimal change (1 extra input dim). dt_world is already available in
the training loop's meta dict.  
**Cons:** Only 1 scalar — may not capture non-linear dt→scale relationship.
Doesn't account for k separately.

### Design B: k Embedding + log(dt_world)

Add a learned k embedding and concatenate with dt_world.

```
k_emb    = Embedding(k_max=20, dim=8)[k]                             # [B, 8]
dt_feat  = log(dt_world.clamp(min=0.01))                             # [B, 1]
cond     = concat(k_emb, dt_feat)                                     # [B, 9]
cond_proj = Linear(9, D//4)                                           # [B, D//4]
mag_feat = mean(token_feat) || max(token_feat) || std(token_feat)    # [B, 3D]
mag_feat = concat(mag_feat, cond_proj)                                # [B, 3D + D//4]
log_t_mag = MLP(mag_feat)                                             # [B, 1]
```

**Pros:** Both dt and k information. k embedding can learn frame-interval
specific biases (wide-baseline pairs may need different scale than narrow ones).  
**Cons:** More parameters, more invasive change to forward signature (k must be passed).

### Design C: dt/k-Conditioned Residual Calibration

Keep the existing magnitude head unchanged. Add a lightweight residual
calibration branch that takes dt_world/k and outputs a correction to log_t_mag.

```
# Existing: unchanged
log_t_mag_raw = MLP(pool(token_feat))                                # [B, 1]

# New: calibration branch
k_emb    = Embedding(k_max=20, dim=4)[k]                             # [B, 4]
dt_feat  = log(dt_world.clamp(min=0.01))                             # [B, 1]
cond     = concat(k_emb, dt_feat)                                     # [B, 5]
log_calib = MLP_calib(cond)                                           # [B, 1]

log_t_mag = log_t_mag_raw + log_calib                                 # [B, 1]
```

**Pros:** Existing head untouched — minimal regression risk. Calibration branch
is tiny (5→16→1 MLP). Can be initialized to output ~0 (no correction).  
**Cons:** If raw head is fundamentally broken (constant 0), the residual branch
must learn the full mapping, which may be as hard as Design A.

---

## 4. Recommended: T51a — Design A (log(dt) First)

**Rationale:**

1. **Minimal risk.** Only 1 extra input dimension to the magnitude head MLP.
   All other code paths (token features, matching, pose, tmag_bias, affine) unchanged.
2. **Direct signal.** `log(dt_world)` provides a strong inductive bias: the model
   should learn `log_tmag ≈ log(dt_world) + bias` (i.e., magnitude proportional to distance).
3. **Easy to ablate.** A single config flag (`tmag_condition_on_dt: bool = False`)
   can toggle it on/off for clean comparison.
4. **dt_world is already available** in the training meta dict and doesn't need
   new dataset code.

**If T51a shows partial improvement but magnitude still not fully calibrated:**
→ escalate to Design C (residual calibration with k embedding).

---

## 5. Required Code Changes

### 5.1 New Config Fields (config.py)

```python
# T51: dt-conditioned magnitude head
tmag_condition_on_dt: bool = False          # enable log(dt_world) conditioning
tmag_dt_clamp_min: float = 0.01             # min dt for log(dt) stability
tmag_dt_log_scale: float = 1.0              # optional scaling factor on log(dt)
```

### 5.2 Model Changes (interaction.py)

**TranslationMagnitudeHead.__init__:** accept `cond_dim: int = 0`. If >0, add a
small Linear(3*D + cond_dim, 3*D) projection to fuse conditioned features.

**CoarseInteraction.__init__:** new arg `tmag_condition_on_dt: bool = False`,
`tmag_dt_clamp_min: float = 0.01`.

**CoarseInteraction.forward:** new kwarg `dt_world: Optional[Tensor] = None`.
If `tmag_condition_on_dt` and `dt_world is not None`:
```python
log_dt = torch.log(dt_world.float().view(-1, 1).clamp_min(self.tmag_dt_clamp_min))
cond = log_dt  # [B, 1]
```
pass `cond` to `self.mag_head(..., condition=cond)`.

**TranslationMagnitudeHead.forward:** new kwarg `condition: Optional[Tensor] = None`.
If condition is not None, concat to pooled features before MLP.

### 5.3 Model Changes (model.py)

**PanoramaRelPoseModel.forward:** new optional kwarg `dt_world: Optional[Tensor] = None`.
Forward to `self.coarse(..., dt_world=dt_world)`.

**PanoramaRelPoseModel.__init__:** pass `tmag_condition_on_dt` and
`tmag_dt_clamp_min` from cfg to `CoarseInteraction`.

### 5.4 Training Changes (train_mvp.py)

**Step 1 — pass dt_world to model:**
```python
dt_world = _meta_batch_field(meta, "dt_world", bsz, default=None)
dt_tensor = torch.tensor([float(v) if v is not None else 0.01 for v in dt_world],
                          device=device, dtype=torch.float32)
R_pred, t_pred, aux = model(IA, IB, dt_world=dt_tensor)
```

**Step 2 — (optional) dt-bucketed tmag weight:**
If magnitude still degrades on long baselines after T51a, add dt-dependent loss
weights: `w_tmag_dt_scale = 1.0 / dt_world` or similar.

### 5.5 Files Changed

| File | Change | Risk |
|---|---|---|
| `config.py` | 3 new fields | Low |
| `interaction.py` | cond_dim in TranslationMagnitudeHead, dt_world in CoarseInteraction.forward | Medium |
| `model.py` | forward dt_world to coarse | Low |
| `train_mvp.py` | extract dt_world from meta, pass to model | Low |

**Total: ~30 lines of new code.**

---

## 6. Training / Evaluation Protocol

### 6.1 T51a Probe

- Init from O49m3 `best_odom_drift.pt` (strict_load=False for new params).
- `tmag_condition_on_dt=True`, all other settings = O49m (seq-turn, k_choices, etc.).
- `max_steps=400`, `eval_every=50`, `batch_size=4`.
- Compare with O49m3 baseline on:
  - `path_length_ratio` (should rise from 0.20 toward 1.0)
  - `pred_tmag` per-step (should vary with gt_tmag, not constant)
  - `tmag_rel_err` per-k and per-dt
  - `drift`, `ATE` (may initially rise as compression is removed)
  - `tdir_abs`, `rot` (should not degrade)

### 6.2 Success Criteria

| Criterion | Threshold | Priority |
|---|---|---|
| path_length_ratio > 0.5 | Must pass | ⭐⭐⭐ |
| pred_tmag std > 0.02 (previously ~0.001) | Must pass | ⭐⭐⭐ |
| tmag_rel_err < 0.80 at k=1 | Should pass | ⭐⭐ |
| drift < 1.5 (willing to accept rise from 1.22) | Should pass | ⭐⭐ |
| tdir_abs < 25° | Should pass | ⭐⭐ |

### 6.3 Fallback if T51a Fails

If T51a (log(dt) conditioning) does not improve magnitude:
1. Try Design C (residual calibration branch with k embedding).
2. Consider stronger supervision: dt-bucketed tmag weights, or direct
   `|pred_tmag - gt_tmag|/gt_tmag` loss with higher `w_tmag`.
3. If both fail, accept that coarse-only magnitude regression is architecturally
   limited and evaluate post-hoc affine as a pragmatic fix.

---

## 7. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Model ignores dt_world and outputs constant anyway | Medium | Start with very high w_tmag (0.5+) to force gradient through dt |
| Drift rises above 2.0 (unacceptable) | Medium | Accept drift 1.3–1.6 as "real" drift without compression |
| New params cause NaN / training instability | Low | log(dt) clamp_min prevents log(0); init MLP with small weights |
| Strict_load_checkpoint fails due to new params | Low | strict_load=False already handles missing params |
| Batch meta dt_world is None for some pairs | Low | Default to 0.01 and log; O49 dataset has dt_world for all pairs |

---

## 8. Relation to Fine Stage (F0)

T51a is **independent** of fine stage. It modifies only the coarse interaction's
magnitude head. When/if fine_pose_fuse_strength > 0 is enabled later, the
magnitude blending path already uses `aux["tc_mag"]` (biased by T51a conditioning).

The recommended sequence:
```
T51a (dt-conditioned tmag) → evaluate → if magnitude fixed:
  → Re-run F0 topk64 with fine_pose_fuse_strength > 0
  → Evaluate O50a chain loss (only after magnitude is healthy)
```

---

## 9. Why NOT O50a Chain Loss Now

Chain loss applies constraints across consecutive frames in an odometry chain.
If each frame's tmag is systematically wrong (constant ~0.05 vs actual 0.03–0.66),
chain loss will propagate and amplify these errors across the sequence.

Chain loss should be evaluated **after** T51a establishes per-frame magnitude
accuracy. The sequence-level shape diagnostics (path_ratio, turn_sum, straightness)
are the right metrics to judge chain loss — but only with a working magnitude head.

---

## 10. Next Steps

1. **Implement T51a** (Design A): ~30 lines across 4 files.
2. **Run T51a short probe** from O49m3 checkpoint (400 steps).
3. **Evaluate** with `inspect_o49_shape_debug.py` and `inspect_tmag_buckets.py`.
4. **If successful:** escalate to T51b (Design C with k embedding).
5. **After magnitude fixed:** re-evaluate F0 fine_pose_fuse_strength and O50a chain loss.

---

*Plan only. No code was modified for this document.*
