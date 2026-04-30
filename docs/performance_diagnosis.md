# Stage 9 Performance Diagnosis

Date: 2026-04-30

This report reads existing artifacts only. No training was run and no model or loss code was changed.

## Data Sources

Scanned artifacts under `checkpoints/`:

- `final_summary.json`: 11 experiments
- `eval_buckets_latest.json`: 11 experiments
- `odom_metrics_latest.json`: 1 experiment
- `matching_diag_latest.json`: 1 experiment

Generated temporary analysis tables under `/tmp/gdd_stage9_diag/` using the existing summary tools:

- `summary_all_experiments.csv`
- `bucket_k.csv`
- `bucket_dt.csv`
- `bucket_kdt.csv`

Important caveat: `C11_tbranch_gated_L0` is the only run with the newer t_mag, odometry, and geometry-refinement metrics, but its latest eval appears to be a very short smoke/debug run with only 2 bucket samples. Treat it as pipeline sanity, not as a trained-performance baseline.

## Current Best Experiments

Sorted mainly by pair-level translation direction quality (`tdir_abs`) among the existing trained runs.

| experiment | rot | tdir | tdir_abs | tdir_local_A_abs | tmag_rel_err | trans_vec_l2 | epi_mass | top1 | top5 | entropy | cycle_error | odom_ATE | odom_drift | norm_drift | geom_refine_success |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C19_no_color_aug_800 | 6.927 | 20.141 | 15.671 | 15.451 | - | - | 0.174 | 0.712 | 0.966 | 5.256 | 0.00518 | - | - | - | - |
| C21_fine_stage_fuse0_no_dup_pose_800 | 7.205 | 20.122 | 17.180 | 16.005 | - | - | 0.301 | 0.747 | 0.898 | 5.033 | 0.00129 | - | - | - | - |
| C20_fine_stage_fuse0_800 | 7.178 | 24.040 | 20.904 | 19.829 | - | - | 0.309 | 0.746 | 0.897 | 4.979 | 0.00129 | - | - | - | - |
| C16_coloraug025_stats_800 | 7.380 | 24.420 | 21.118 | 20.777 | - | - | 0.174 | 0.712 | 0.967 | 5.254 | 0.00518 | - | - | - | - |
| C12_bearing_fuse_stats_800 | 7.868 | 26.321 | 23.829 | 20.920 | - | - | 0.176 | 0.620 | 0.905 | 5.225 | 0.00518 | - | - | - | - |

Pipeline sanity row with newer metrics:

| experiment | status | selected_k | num_pairs | tmag_rel_err | odom_RPE_rot | odom_RPE_tdir | odom_RPE_tmag | odom_ATE | odom_drift | norm_drift |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C11_tbranch_gated_L0 | ok, fallback from k=1 to k=5 | 5 | 129 | 0.732 | 161.525 | 94.168 | 0.611 | 7.939 | 7.034 | 0.970 |

## Pair-Level Ranking

Best rotation:

| rank | experiment | rot | tdir_abs |
|---:|---|---:|---:|
| 1 | C19_no_color_aug_800 | 6.927 | 15.671 |
| 2 | C15_cross_context_s010_stats_800 | 7.164 | 41.782 |
| 3 | C20_fine_stage_fuse0_800 | 7.178 | 20.904 |
| 4 | C21_fine_stage_fuse0_no_dup_pose_800 | 7.205 | 17.180 |
| 5 | C14_cross_context_stats_800 | 7.307 | 26.175 |

Best translation direction (`tdir_abs`):

| rank | experiment | rot | tdir_abs | tdir_local_A_abs |
|---:|---|---:|---:|---:|
| 1 | C19_no_color_aug_800 | 6.927 | 15.671 | 15.451 |
| 2 | C21_fine_stage_fuse0_no_dup_pose_800 | 7.205 | 17.180 | 16.005 |
| 3 | C20_fine_stage_fuse0_800 | 7.178 | 20.904 | 19.829 |
| 4 | C16_coloraug025_stats_800 | 7.380 | 21.118 | 20.777 |
| 5 | C12_bearing_fuse_stats_800 | 7.868 | 23.829 | 20.920 |

## Matching-Level Ranking

These values are count-weighted averages from `bucket_k.csv`.

| experiment | epi_mass | top1 | top5 | entropy | cycle_error | rot | tdir_abs |
|---|---:|---:|---:|---:|---:|---:|---:|
| C20_fine_stage_fuse0_800 | 0.309 | 0.746 | 0.897 | 4.979 | 0.00129 | 8.235 | 35.920 |
| C21_fine_stage_fuse0_no_dup_pose_800 | 0.301 | 0.747 | 0.898 | 5.033 | 0.00129 | 8.721 | 35.930 |
| C22_fine_sep_pos_fuse0_800 | 0.269 | 0.735 | 0.905 | 5.152 | 0.00129 | 8.134 | 38.440 |
| C12_bearing_fuse_stats_800 | 0.176 | 0.620 | 0.905 | 5.225 | 0.00518 | 9.103 | 36.470 |
| C19_no_color_aug_800 | 0.174 | 0.712 | 0.966 | 5.256 | 0.00518 | 8.759 | 37.720 |

Fine stage is doing something useful at the matching level: `C20/C21` roughly double `epi_mass` versus `C19`, lower entropy, and reduce cycle error by about 4x. However, that matching improvement does not consistently beat `C19` on final pair-level pose. That points away from a pure matching bottleneck and toward pose usage/fusion or training recipe issues.

## Bucket Observations

Selected `k` buckets:

| experiment | k | rot | tdir_abs | local_A_abs | epi_mass | top1 | top5 | entropy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C19_no_color_aug_800 | 5 | 3.647 | 39.598 | 37.689 | 0.173 | 0.928 | 0.997 | 5.256 |
| C19_no_color_aug_800 | 10 | 4.329 | 36.699 | 34.374 | 0.174 | 0.865 | 0.995 | 5.256 |
| C19_no_color_aug_800 | 20 | 8.504 | 36.579 | 31.930 | 0.176 | 0.830 | 0.981 | 5.256 |
| C19_no_color_aug_800 | 40 | 18.557 | 37.984 | 28.708 | 0.173 | 0.225 | 0.892 | 5.256 |
| C20_fine_stage_fuse0_800 | 5 | 3.590 | 38.717 | 37.058 | 0.317 | 0.962 | 0.997 | 4.999 |
| C20_fine_stage_fuse0_800 | 10 | 3.872 | 35.222 | 33.346 | 0.327 | 0.934 | 0.996 | 4.956 |
| C20_fine_stage_fuse0_800 | 20 | 7.723 | 34.293 | 30.124 | 0.316 | 0.831 | 0.979 | 4.970 |
| C20_fine_stage_fuse0_800 | 40 | 17.754 | 35.457 | 26.667 | 0.278 | 0.258 | 0.614 | 4.990 |
| C21_fine_stage_fuse0_no_dup_pose_800 | 5 | 3.749 | 37.728 | 35.793 | 0.306 | 0.962 | 0.997 | 5.055 |
| C21_fine_stage_fuse0_no_dup_pose_800 | 10 | 4.342 | 34.541 | 32.198 | 0.316 | 0.934 | 0.995 | 5.012 |
| C21_fine_stage_fuse0_no_dup_pose_800 | 20 | 8.406 | 34.878 | 29.845 | 0.308 | 0.841 | 0.981 | 5.024 |
| C21_fine_stage_fuse0_no_dup_pose_800 | 40 | 18.386 | 36.560 | 26.798 | 0.272 | 0.252 | 0.617 | 5.042 |

Main readings:

- Wide-baseline (`k=40`) is still hard: rotation rises to about 18 deg and top1 drops to about 0.23-0.26 even with fine stage.
- Small/medium baselines have good top1 but still poor `tdir_abs` in the bucket table. That is consistent with small-baseline translation ambiguity: rotation can be good while translation direction remains noisy.
- Fine stage helps `epi_mass`, entropy, and cycle consistency, especially for `k=10/20`; for `k=40`, top5 recall falls sharply in fine-stage buckets, so routing/top-k/bias ablations should be prioritized before strong fine pose fusion.

## Bottleneck Classification

### 1. Pose-head / pose-fusion bottleneck

Evidence:

- `C20/C21` improve matching diagnostics a lot over `C19`.
- Despite better matching, the best pair-level model remains `C19_no_color_aug_800`.
- This suggests the network is not fully converting better correspondences into better `R` and `t_dir`, or the fine-stage training/fusion recipe is not yet aligned with the pair-level objective.

### 2. Wide-baseline and fine-routing bottleneck

Evidence:

- At `k=40`, top1 is only about 0.225 for `C19` and about 0.25 for `C20/C21`.
- Fine stage improves `epi_mass`, but `k=40` top5 drops to about 0.61 for fine-stage runs, which suggests the routed candidate set may miss valid matches or become too restrictive.

### Secondary bottleneck: small-baseline translation direction

Evidence:

- `k=5` rotation is already around 3.6 deg, but `tdir_abs` remains around 38 deg in bucket eval.
- This is a known odometry issue: small baselines often have weak translation direction observability. It should be handled with dt-aware loss weighting and odom-small-k curriculum rather than by blindly increasing tdir loss.

### Not yet diagnosable: scale and odometry accumulation

The new t_mag and odometry metrics exist only for the smoke/debug `C11_tbranch_gated_L0` artifact. We need at least one short `odom-small-k` or t_mag-enabled debug run before making a serious scale or drift diagnosis.

## Addendum: C31/O28/O29 Small-k Findings

Later candidate experiments added meaningful t_mag and odometry evidence:

- `C31` remains the wide/stable baseline and teacher.
- `O28` is the conservative small-k finetune candidate: it improves drift while keeping pair-level scale cleaner.
- `O29` validates odometry-aware checkpoint selection: its `best_odom_drift.pt` lowers unified small-k drift to `1.410`, with `tdir_abs=24.865`, but its separate eval-only `tmag_rel=0.918` is slightly above the planned `0.9` gate.
- `O30` extends the same idea with a small-k bucket gate and is now the drift-first small-k candidate: eval-only drift `1.408`, `tdir_abs=24.776`, and slightly better `k=1/2/3` buckets than O29. It still has the same eval-only scale caveat, `tmag_rel=0.918`.
- `O31` tested hard ignoring `dt<0.05` samples for tdir loss. It is not adopted: eval-only drift worsened to `1.470` and `tdir_abs` crossed `25°`.

Updated bottleneck reading:

- The main remaining performance bottleneck is still small-baseline translation direction, especially `k=1/2/3`.
- Checkpoint selection can reduce odometry drift without changing the model, so odom-aware selection should stay enabled.
- Further gains should target small-k `tdir_abs` while preserving O29-level drift and avoiding scale regression.

## Recommended Next 3 Changes

### Change 1: Fine-routing ablation script and diagnostics-first fine evaluation

Files:

- `scripts/run_effect_ablation.sh` or a focused `scripts/run_fine_routing_ablation.sh`

Why:

- Fine stage improves matching quality but still struggles at `k=40`.
- Before changing model internals, test whether `topk_coarse`, `epi_bias_strength`, and `fine_pose_fuse_strength` are the limiting factors.

Expected metrics:

- Improve `routing_recall_in_gt_band`, `top5_in_gt_band`, `epi_mass_in_gt_band`, and `cycle_error`.
- Pair-level improvement should appear first in `k=20/40` buckets.

Verification:

```bash
conda run --no-capture-output -n pytorch python -m py_compile config.py dataset_pano_only.py losses.py model.py train_mvp.py interaction.py transformer_encoder.py pose_head.py geometry_refine.py
bash -n scripts/run_effect_ablation.sh
```

Rollback:

```bash
git revert <commit>
```

### Change 2: Matching schedule for epipolar supervision

Files:

- `config.py`
- `train_mvp.py`

Why:

- Existing matching losses help, but early over-constraint can make matching or pose fusion brittle.
- A staged epipolar angle and `w_epi` ramp can improve matching stability without changing architecture.

Expected metrics:

- Improve `epi_mass_in_gt_band`, `top1/top5`, and entropy stability.
- Should indirectly improve `rot` and `tdir_abs` if pose head can use the cleaner matching.

Verification:

```bash
conda run --no-capture-output -n pytorch python -m py_compile config.py dataset_pano_only.py losses.py model.py train_mvp.py interaction.py transformer_encoder.py pose_head.py geometry_refine.py
conda run --no-capture-output -n pytorch python train_mvp.py --set max_steps=20 --set eval_every=20 --set max_eval_batches=2 --set max_train_eval_batches=2 --set num_workers=0 --set persistent_workers=False --set batch_size=1 --set grad_accum=1 --set epi_angle_schedule_enable=True
```

Rollback:

```bash
git revert <commit>
```

### Change 3: Train a short t_mag / odom-small-k baseline before judging scale

Files:

- No code change required initially; use existing `scripts/run_odom_small_k.sh` or add a small debug variant if needed.

Why:

- Current trained runs do not have meaningful t_mag/odom metrics.
- Without a trained scale head, any drift conclusion would be premature.

Expected metrics:

- Establish baseline for `tmag_rel_err`, `trans_vec_l2`, `odom_metric_ATE`, `odom_metric_drift`, and normalized drift.

Verification:

```bash
bash -n scripts/run_odom_small_k.sh
# Optional short run only when approved:
# conda run --no-capture-output -n pytorch python train_mvp.py --set exp_name=O0_odom_small_k_probe --set max_steps=200 ...
```

Rollback:

- No code rollback if only running an experiment.
- Delete or ignore the generated checkpoint if the run is not useful.

## Temporarily Not Recommended

- Do not add another large model module yet. The current evidence says better matching is not automatically improving pose, so architecture growth may hide the real bottleneck.
- Do not default to geometry refinement as the final pose until it beats network pose on trained checkpoints. Current geometry metrics are not from a trained post-stage-7 model.
- Do not optimize odometry drift from the current `C11` numbers. That artifact is a smoke sanity result, not a reliable trained odometry baseline.
- Do not strengthen fine-stage epipolar bias aggressively. The `k=40` fine-stage top5 drop suggests strong routing/bias may remove valid candidates.

## Next Stage Recommendation

Proceed with Stage 10: generate a compact effect-ablation script, but keep it focused. The first matrix should emphasize:

- fine routing: `topk_coarse`, `epi_bias_strength`, `fine_pose_fuse_strength`
- epipolar schedule: wide-to-narrow band and `w_epi` ramp
- odom-small-k probe: one short t_mag-enabled run to establish scale/drift baselines

## Current Model Candidates

Decision recorded after the C31/O27 small-k anchor experiments:

| role | experiment | checkpoint | rot | tdir_abs | tdir_local_A_abs | best_joint | last_tmag_rel_err | last_odom_drift | note |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| wide/stable baseline | C31_coarse_gtmatch_tmag_detach_workers6_800 | `checkpoints/C31_coarse_gtmatch_tmag_detach_workers6_800/best_joint_local_A_abs.pt` | 7.116 | 16.318 | 16.039 | 30.549 | - | - | Keep as the original wide/stable baseline and teacher/init checkpoint. |
| small-k finetune fallback | O28_c31_smallk_anchor2_all_1200 | `checkpoints/O28_c31_smallk_anchor2_all_1200/best_joint_local_A_abs.pt` | 2.204 | 23.307 | 23.406 | 27.715 | 0.802 | 1.420 | Conservative small-k fallback; good final pair-level guard, but later odom-aware selection found a better drift checkpoint. |
| small-k drift candidate | O30_c31_smallk_anchor2_smallkselect_1200 | `checkpoints/O30_c31_smallk_anchor2_smallkselect_1200/best_smallk_odom.pt` | 2.382 | 24.776 | 24.794 | 29.960 | 0.918 | 1.408 | Current drift-first small-k candidate; selected by odom and k=1/2/3 bucket guards. |

Recommended comparison policy:

- Use C31 as the reference model for wide/stable pair-level quality and as the teacher for anchor-distillation finetunes.
- Use O30 as the current small-k odometry candidate when evaluating `k=1/2/3/5`, drift, and sequence-level behavior.
- Keep O28 as a conservative fallback because its final pair-level `tdir_abs`/`tmag_rel` were cleaner than O30's eval-only selected checkpoint.
- Do not adopt O31 or O32: hard tiny-dt ignore and fixed 0.05 tiny-dt weighting both failed to beat O30 under unified eval.
- Do not replace C31 with O28/O30 globally yet: the small-k candidates were optimized for odometry behavior and should remain separate from the wide/stable baseline.
