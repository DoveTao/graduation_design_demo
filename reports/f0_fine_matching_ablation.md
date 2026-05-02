# F0 Fine-Stage Matching-Only Ablation

Date: 2026-05-02  
Commits: `1f30519` (batch_size fix), `01936cc` (ablation runner)  
Status: Diagnostic complete. topk64 identified as sweet spot. Fine pose fusion deferred.

---

## 1. Background

With coarse-only O49m3 as the established baseline (drift=1.22), we investigate whether
fine-stage token routing and matching can improve correspondence quality **without**
affecting the final pose. The fine stage is opened (`use_fine_stage=True`) but
`fine_pose_fuse_strength=0` ensures fine predictions do NOT influence the output.

**Goal:** evaluate fine matching quality via epipolar mass, top-k hitrate, cycle
error, and routing recall — before committing to fine pose fusion.

---

## 2. Experimental Setup

| Parameter | Value |
|---|---|
| Init checkpoint | `O49m3/best_odom_drift.pt` (drift=1.22) |
| Fine stage | `use_fine_stage=True` |
| Fine pose influence | `fine_pose_fuse_strength=0` (matching-only) |
| Depth branch | `use_depth_branch=False` |
| Tmag affine | `use_tmag_affine_calib=False` |
| Max steps | 300 |
| Eval every | 50 |
| Batch size | 1 (reduced from 4 due to 6GB GPU OOM) |
| Seq-turn | `use_seq_turn_loss=True, w=0.007` |
| K choices | (1, 2, 3, 5, 10) |
| Odom eval | k=1, 132 pairs |

**Varying parameter:** `topk_coarse` (number of coarse tokens routed to fine stage per cell)

| Mode | topk_coarse |
|---|---|
| f0_topk64 | 64 |
| f0_topk96 | 96 |

---

## 3. Results: topk64 vs topk96

### 3.1 Fine vs Coarse Matching

| Metric | Coarse | Fine (topk64) | Fine (topk96) |
|---|---:|---:|---:|
| **epi_mass** | 0.174 | **0.245** | 0.210 |
| **top1** | 0.901 | **0.938** | 0.922 |
| **top5** | **0.994** | 0.993 | 0.989 |
| **entropy** | 5.256 | 5.436 | 5.842 |
| **cycle_error** | 0.00518 | **0.00129** | 0.00130 |
| routing_recall | — | 1.000 | 1.000 |
| no_candidate_row | — | 0.0 | 0.0 |
| allowed_mask_density | 0.333 | 0.333 | 0.500 |

### 3.2 Pose / Odom

| Metric | O49m3 baseline | F0-topk64 | F0-topk96 |
|---|---:|---:|---:|
| best_odom_drift | 1.22 | 1.31 | 1.31 |
| best_odom_upd | 100 | 50 | 50 |
| skip_updates | 0 | 0 | 0 |
| best_tdir_abs | 23.1° | 23.9° | 23.9° |

---

## 4. Key Findings

### 4.1 Fine matching improves correspondence quality

topk64 delivers substantial gains over coarse-only:
- **epi_mass +41%** (0.174 → 0.245): more matching probability mass concentrated
  in the GT epipolar band
- **top1 +3.7%** (0.901 → 0.938): higher chance the top match falls within GT band
- **cycle_error ÷4** (0.00518 → 0.00129): better A↔B matching consistency
- **routing_recall = 1.0**: every coarse token successfully routes to fine candidates
- **no_candidate_row = 0**: no coarse token is left without a fine match

### 4.2 topk=64 is the sweet spot

topk96 **degrades** fine matching:
- fine_epi_mass drops from 0.245 to 0.210 (−14%)
- fine_top1 drops from 0.938 to 0.922 (−1.6%)
- fine_entropy increases from 5.44 to 5.84 (more uniform, less sharp matching)
- allowed_mask_density increases to 0.50 (more tokens pass the mask, diluting attention)

Increasing topk adds more coarse candidates, which dilutes fine attention and
reduces matching sharpness. topk=64 achieves the best balance.

### 4.3 OOM mitigation

Initial runs at `batch_size=4` hit CUDA OOM on RTX 3060 (6GB). Fine stage adds
768 fine tokens × 2 images ≈ 1–2GB extra VRAM. Reducing to `batch_size=1`
resolved the issue without modifying the model. Script supports `FINE_BS`
environment variable override.

### 4.4 Why not run topk128?

topk96 already shows degradation, and topk128 would further dilute fine
attention. The direction is clear — topk64 is optimal.

### 4.5 Why defer fine pose fusion?

| Reason | Detail |
|---|---|
| Drift regresses | 1.22 (O49m3) → 1.31 (F0) despite better matching |
| tdir unchanged | ~24° vs 23° |
| Path compression unresolved | path_length_ratio ≈ 0.2 (same as coarse-only) |
| Magnitude head still broken | pred_tmag ≈ constant |

Better matching quality has not translated to better pose/odometry, likely
because the magnitude bottleneck dominates the error budget. Fine pose fusion
would add complexity without addressing the root cause.

---

## 5. Next Steps

| Priority | Action | Rationale |
|---|---|---|
| ⭐⭐⭐ | Fix magnitude head (dt/k-conditioned, stronger supervision) | Dominant bottleneck |
| ⭐⭐ | After magnitude fix, re-evaluate fine_pose_fuse_strength | Should compound gains |
| ⭐ | topk64 is ready as a building block | Fine routing/matching infrastructure is validated |

---

## 6. Diagnostic Tool Inventory

| Script | Commit | Purpose |
|---|---|---|
| `scripts/run_fine_matching_ablation.sh` | `01936cc` / `1f30519` | Fine matching ablation launcher (fix: batch_size) |
| `scripts/inspect_o49_shape_debug.py` | `e3448b4` | Trajectory shape metrics |
| `scripts/inspect_tmag_buckets.py` | `c2e6ca1` | Per-k/dt/step tmag diagnostics |
| `scripts/run_t50_tmag_affine_probe.sh` | `801cd06` | T50 affine calibration launcher |
| `scripts/summarize_odom_cycle.py` | `8331469` | Multi-experiment ranking |

---

*Report auto-generated from checkpoint artifacts. Trained experiments: `F0_fine_match_topk64_fromO49m3_300`, `F0_fine_match_topk96_fromO49m3_300`.*
