# C31 vs O27 Candidate Comparison

Date: 2026-04-30

This report uses `eval_only=True` to evaluate fixed checkpoints under the same protocols. No training updates were performed.

## Summary

- Keep `C31_coarse_gtmatch_tmag_detach_workers6_800` as the wide/stable baseline.
- Promote `O28_c31_smallk_anchor2_all_1200` to the current small-k finetune candidate.
- Keep `O27_c31_smallk_anchor2_all_800` as the small-k fallback: it is close to O28, but O28 is better under the unified eval protocol.
- Do not promote O27/O28 to global default yet: small-k rotation and drift improve, but wide/stable direction and scale are still safer with C31.

## Pair, Matching, and Odometry Metrics

| eval | rot | tdir_abs | local_A_abs | tmag_rel | tvec_l2 | epi_mass | top1 | top5 | entropy | cycle | RPE_rot | ATE | drift | norm_drift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E_C31_smallk_eval | 5.780 | 23.623 | 23.439 | 0.795 | 1.421 | 0.174 | 0.910 | 0.996 | 5.256 | 0.005 | 6.736 | 9.167 | 1.842 | 1.468 |
| E_O27_smallk_eval | 2.423 | 24.651 | 24.874 | 0.881 | 1.556 | 0.174 | 0.899 | 0.993 | 5.256 | 0.005 | 0.960 | 10.286 | 1.610 | 1.283 |
| E_C31_wide_eval | 6.919 | 18.470 | 18.801 | 0.709 | 1.526 | 0.174 | 0.698 | 0.966 | 5.256 | 0.005 | 5.845 | 9.233 | 7.511 | 1.036 |
| E_O27_wide_eval | 7.830 | 19.491 | 19.821 | 0.881 | 1.681 | 0.175 | 0.687 | 0.961 | 5.256 | 0.005 | 1.388 | 8.677 | 7.309 | 1.008 |
| O28 final small-k | 2.204 | 23.307 | 23.406 | 0.802 | 0.919 | 0.174 | 0.919 | 0.995 | 5.256 | 0.005 | 0.672 | 8.949 | 1.420 | 1.131 |

## Small-k Buckets

| eval | k | rot | tdir_abs | tmag_rel |
|---|---:|---:|---:|---:|
| C31 small-k | 1 | 6.91 | 28.73 | 0.857 |
| C31 small-k | 2 | 6.71 | 25.80 | 0.669 |
| C31 small-k | 3 | 6.49 | 24.27 | 0.723 |
| C31 small-k | 5 | 5.99 | 21.27 | 0.845 |
| C31 small-k | 10 | 4.55 | 19.45 | 0.911 |
| C31 small-k | 20 | 4.03 | 22.17 | 0.759 |
| O27 small-k | 1 | 1.06 | 29.59 | 0.779 |
| O27 small-k | 2 | 0.94 | 26.72 | 0.811 |
| O27 small-k | 3 | 0.89 | 25.26 | 0.889 |
| O27 small-k | 5 | 1.02 | 22.41 | 0.938 |
| O27 small-k | 10 | 2.31 | 20.72 | 0.964 |
| O27 small-k | 20 | 8.38 | 23.17 | 0.904 |

## Decision

- C31 remains the safer baseline for direction and scale, especially outside the small-k rotation-focused setting.
- O27 is better for small-k rotation and improves small-k drift from `1.842` to `1.610`, but it trades off `tdir_abs` and `tmag_rel`.
- O28 improves the unified small-k candidate further: final `tdir_abs=23.31`, `tmag_rel=0.802`, and drift `1.420`.
- O28 does not replace C31 globally. It is the preferred small-k odometry candidate, while C31 remains the wide/stable baseline and teacher.

## O28 Gate Result

Adoption rules and outcome:

- `tdir_abs <= 25.0`: pass, final `23.31`.
- Drift improves over unified O27 `1.610`: pass, final `1.420`.
- `tmag_rel <= 0.969`: pass, final `0.802`.
- Small-k caveat: `k=1/2` tdir remains high, so the remaining bottleneck is still small-baseline translation direction, not rotation.

Use:

- Small-k candidate: `checkpoints/O28_c31_smallk_anchor2_all_1200/best_joint_local_A_abs.pt`
- Small-k fallback: `checkpoints/O27_c31_smallk_anchor2_all_800/best_joint_local_A_abs.pt`
- Wide/stable baseline: `checkpoints/C31_coarse_gtmatch_tmag_detach_workers6_800/best_joint_local_A_abs.pt`

## O29 Plan

O29 is a checkpoint-selection experiment, not a model or loss change.

- Recipe: same as O28.
- New behavior: save `best_odom_drift.pt` when `odom_metric_drift` improves while `tdir_abs <= 25.0` and `tmag_rel_err <= 0.9`.
- Adoption rule: if O29 `best_odom_drift < 1.420` without violating the gates, use `checkpoints/O29_c31_smallk_anchor2_odomselect_1200/best_odom_drift.pt` as the small-k odometry candidate.

## O29 Result

O29 successfully tested the odometry-aware checkpoint selection idea. It did not change the model or loss; it only changed which checkpoint is saved during the O28-style run.

Training-time best odom checkpoint:

| checkpoint | upd | drift | tdir_abs | tmag_rel |
|---|---:|---:|---:|---:|
| `O29/best_odom_drift.pt` | 500 | 1.410 | 23.939 | 0.801 |

Unified eval-only comparison against O28:

| eval | rot | tdir_abs | local_A_abs | tmag_rel | tvec_l2 | RPE_rot | ATE | drift | norm_drift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E_O28_candidate_smallk_eval | 2.330 | 24.753 | 24.945 | 0.883 | 1.558 | 0.895 | 10.100 | 1.574 | 1.255 |
| E_O29_best_odom_smallk_eval | 2.395 | 24.865 | 24.880 | 0.918 | 1.592 | 1.070 | 8.936 | 1.410 | 1.124 |

Small-k bucket caveat for `E_O29_best_odom_smallk_eval`:

| k | rot | tdir_abs | tmag_rel |
|---:|---:|---:|---:|
| 1 | 1.20 | 29.61 | 0.827 |
| 2 | 1.05 | 26.65 | 0.873 |
| 3 | 0.95 | 25.12 | 0.930 |
| 5 | 0.97 | 22.15 | 0.961 |
| 10 | 2.13 | 20.62 | 0.977 |
| 20 | 8.14 | 25.02 | 0.940 |

Decision:

- O29 passes the odometry objective: drift improves from O28's unified eval `1.574` to `1.410`, and `tdir_abs=24.865` stays under the `25°` guard.
- O29 does not cleanly pass the scale guard under the separate eval-only protocol: `tmag_rel=0.918`, slightly above the planned `0.9` threshold. During training eval the same checkpoint had `tmag_rel=0.801`, so this difference is likely due to eval sample/protocol differences, not a code-path failure.
- Use `checkpoints/O29_c31_smallk_anchor2_odomselect_1200/best_odom_drift.pt` as the odom-drift candidate when the priority is trajectory drift.
- Keep `checkpoints/O28_c31_smallk_anchor2_all_1200/best_joint_local_A_abs.pt` as the conservative small-k candidate when the priority is pair-level scale and slightly better rotation.
- Keep `C31` as the wide/stable baseline and teacher.

Next action:

- Do not continue tuning t_mag blindly. The remaining high-value target is small-baseline translation direction, especially `k=1/2/3`, while preserving O29-level drift.

## O30 Plan

O30 keeps the O29 training recipe but adds a stricter small-k checkpoint selector:

- Saves `best_smallk_odom.pt`.
- Selection metric: `odom_metric_drift`.
- Guard buckets: `k=1/2/3`.
- Gate: weighted `k=1/2/3` `tdir_abs <= 28°`.
- Gate: weighted `k=1/2/3` `tmag_rel <= 0.95`.

Purpose:

- O29 is good for global trajectory drift, but its `k=1/2/3` translation direction remains the visible bottleneck.
- O30 does not change learning behavior; it only makes checkpoint selection more aligned with small-baseline odometry.

Adoption rule:

- Prefer O30 only if `best_smallk_odom.pt` matches or improves O29 drift while reducing the weighted `k=1/2/3` `tdir_abs`.
- If O30 improves small-k direction but worsens drift materially, keep O29 as the drift candidate and O28 as the conservative small-k candidate.

## O30 Result

O30 successfully saved both `best_odom_drift.pt` and `best_smallk_odom.pt`. In this run they point to the same eval point, `upd=500`.

Training-time selected checkpoint:

| checkpoint | upd | drift | global tdir_abs | k=1/2/3 tdir_abs | k=1/2/3 tmag_rel |
|---|---:|---:|---:|---:|---:|
| `O30/best_smallk_odom.pt` | 500 | 1.408 | 23.811 | 26.264 | 0.740 |

Unified eval-only comparison:

| eval | rot | tdir_abs | local_A_abs | tmag_rel | tvec_l2 | RPE_rot | ATE | drift | norm_drift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E_O28_candidate_smallk_eval | 2.330 | 24.753 | 24.945 | 0.883 | 1.558 | 0.895 | 10.100 | 1.574 | 1.255 |
| E_O29_best_odom_smallk_eval | 2.395 | 24.865 | 24.880 | 0.918 | 1.592 | 1.070 | 8.936 | 1.410 | 1.124 |
| E_O30_best_smallk_odom_eval | 2.382 | 24.776 | 24.794 | 0.918 | 1.592 | 1.039 | 8.922 | 1.408 | 1.122 |

Small-k buckets for `E_O30_best_smallk_odom_eval`:

| k | rot | tdir_abs | tmag_rel |
|---:|---:|---:|---:|
| 1 | 1.16 | 29.54 | 0.828 |
| 2 | 1.02 | 26.58 | 0.873 |
| 3 | 0.92 | 25.06 | 0.930 |
| 5 | 0.95 | 22.08 | 0.961 |
| 10 | 2.14 | 20.53 | 0.978 |
| 20 | 8.18 | 24.84 | 0.940 |

Decision:

- O30 is the current best odom-drift candidate: it improves O29 slightly on drift, global `tdir_abs`, local `tdir_abs`, and the `k=1/2/3` buckets.
- O30 still does not fix the eval-only scale caveat: global `tmag_rel=0.918`, so it should not replace O28 when pair-level scale is the main priority.
- Updated candidates:
  - Wide/stable baseline: `C31`.
  - Conservative small-k candidate: `O28/best_joint_local_A_abs.pt`.
  - Drift-first small-k candidate: `O30/best_smallk_odom.pt`.

## O31 Plan

O31 targets the remaining `k=1` tiny-motion outliers.

Change:

- Keep the O30 training recipe and checkpoint selection.
- Add `tdir_loss_ignore_dt_below=0.05`.
- Add `tdir_loss_ignore_weight=0.0`.
- This affects only translation-direction supervision weights.
- Rotation loss, t_mag loss, bucket metrics, and odometry eval still use the original samples.

Why:

- The `dt<0.05` bucket is weakly observable for translation direction and can produce very large `tdir_abs` outliers.
- Letting those samples dominate tdir supervision can hurt the learnable small-k regime where translation is still observable enough to be useful.

Adoption rule:

- Prefer O31 if it improves `k=1/2/3` `tdir_abs` without losing O30-level drift.
- Reject O31 if drift rises materially above O30 or global `tdir_abs` crosses `25°`.

## O31 Result

O31 tested whether fully ignoring `dt<0.05` samples for tdir supervision helps small-k odometry. It did not improve the current candidate.

Training-time selected checkpoint:

| checkpoint | upd | drift | global tdir_abs | k=1/2/3 tdir_abs | k=1/2/3 tmag_rel |
|---|---:|---:|---:|---:|---:|
| `O31/best_smallk_odom.pt` | 600 | 1.470 | 23.480 | 26.112 | 0.750 |

Unified eval-only comparison:

| eval | rot | tdir_abs | local_A_abs | tmag_rel | tvec_l2 | RPE_rot | ATE | drift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E_O30_best_smallk_odom_eval | 2.382 | 24.776 | 24.794 | 0.918 | 1.592 | 1.039 | 8.922 | 1.408 |
| E_O31_best_smallk_odom_eval | 2.294 | 25.310 | 25.552 | 0.913 | 1.588 | 0.748 | 9.307 | 1.470 |

Decision:

- Do not adopt O31.
- O31 improves rotation slightly and keeps tmag similar, but it worsens both `tdir_abs` and drift relative to O30.
- Keep O30 as the drift-first small-k candidate.
- The tiny-dt issue remains real, but hard zeroing `dt<0.05` tdir loss is too blunt. A softer alternative would be `tdir_loss_ignore_weight=0.05` or a tdir loss ramp by dt, not a hard ignore.

## O32 Plan

O32 is the softer tiny-dt variant:

- Keep the O30/O31 training recipe and checkpoint selection.
- Use `tdir_loss_ignore_dt_below=0.05`.
- Use `tdir_loss_ignore_weight=0.05` instead of `0.0`.
- This keeps weak tdir supervision for tiny-motion samples instead of fully removing it.

Adoption rule:

- Prefer O32 if it improves `k=1/2/3` `tdir_abs` or global `tdir_abs` while keeping drift near O30.
- Reject O32 if it repeats O31's pattern: `tdir_abs > 25°` or drift materially worse than O30.

## O32 Result

O32 tested the softer tiny-dt variant requested after O31: keep `dt<0.05` samples in the tdir loss with weight `0.05` instead of hard zeroing them.

Training-time selected checkpoint:

| checkpoint | upd | drift | global tdir_abs | k=1/2/3 tdir_abs | k=1/2/3 tmag_rel |
|---|---:|---:|---:|---:|---:|
| `O32/best_smallk_odom.pt` | 500 | 1.433 | 24.202 | 26.670 | 0.745 |

Unified eval-only comparison:

| eval | rot | tdir_abs | local_A_abs | tmag_rel | tvec_l2 | RPE_rot | ATE | drift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E_O30_best_smallk_odom_eval | 2.382 | 24.776 | 24.794 | 0.918 | 1.592 | 1.039 | 8.922 | 1.408 |
| E_O31_best_smallk_odom_eval | 2.294 | 25.310 | 25.552 | 0.913 | 1.588 | 0.748 | 9.307 | 1.470 |
| E_O32_best_smallk_odom_eval | 2.350 | 25.260 | 25.301 | 0.915 | 1.589 | 1.031 | 9.105 | 1.433 |

Small-k buckets for `E_O32_best_smallk_odom_eval`:

| k | rot | tdir_abs | tmag_rel |
|---:|---:|---:|---:|
| 1 | 1.16 | 29.97 | 0.824 |
| 2 | 1.01 | 27.01 | 0.867 |
| 3 | 0.90 | 25.52 | 0.927 |
| 5 | 0.91 | 22.60 | 0.959 |
| 10 | 2.08 | 21.13 | 0.977 |
| 20 | 8.12 | 25.30 | 0.937 |

Decision:

- Do not adopt O32.
- O32 is clearly better than hard-ignore O31 on drift, but it still misses O30: drift is `1.433` vs `1.408`, and eval-only `tdir_abs=25.260°` crosses the `25°` guard.
- Keep O30 as the drift-first small-k candidate.
- Keep C31 as the wide/stable baseline and teacher.
- The next tiny-dt experiment should be a continuous dt ramp, not another fixed threshold. A reasonable candidate is to ramp tdir weight from `0.05` at `dt=0.02` to the normal small-dt weight by `dt=0.10`, while keeping odom eval unchanged.

## O33 Plan

O33 implements the continuous tiny-dt tdir weighting ramp:

- Keep the O30 training recipe and odom/small-k checkpoint selectors.
- Enable `tdir_loss_dt_ramp_enable=True`.
- Use `tdir_loss_dt_ramp_start=0.02`.
- Use `tdir_loss_dt_ramp_end=0.10`.
- Use `tdir_loss_dt_ramp_start_weight=0.05`.
- Use `tdir_loss_dt_ramp_end_weight=-1.0`, meaning the ramp endpoint is the configured `small_dt_t_weight` (`0.15` in the O33 script).

Expected behavior:

- `dt<=0.02`: tdir loss weight `0.05`.
- `0.02<dt<0.10`: linearly ramps from `0.05` to `0.15`.
- `0.10<=dt<0.30`: keeps the normal small-dt weight `0.15`.
- `dt>=0.30`: full tdir weight.

Adoption rule:

- Prefer O33 if it beats O30 on drift or improves `k=1/2/3` `tdir_abs` while keeping eval-only global `tdir_abs <= 25°`.
- Reject O33 if it behaves like O32, especially if global `tdir_abs > 25°` or drift remains materially worse than O30.

## O33 Result

O33 tested the continuous tiny-dt ramp. It improved over O31/O32 in some training-time diagnostics, but did not beat the current O30 small-k candidate.

Training-time selected checkpoint:

| checkpoint | upd | drift | global tdir_abs | k=1/2/3 tdir_abs | k=1/2/3 tmag_rel |
|---|---:|---:|---:|---:|---:|
| `O33/best_smallk_odom.pt` | 500 | 1.423 | 24.145 | 26.618 | 0.743 |

Unified eval-only comparison:

| eval | rot | tdir_abs | local_A_abs | tmag_rel | tvec_l2 | RPE_rot | ATE | drift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E_O30_best_smallk_odom_eval | 2.382 | 24.776 | 24.794 | 0.918 | 1.592 | 1.039 | 8.922 | 1.408 |
| E_O32_best_smallk_odom_eval | 2.350 | 25.260 | 25.301 | 0.915 | 1.589 | 1.031 | 9.105 | 1.433 |
| E_O33_best_smallk_odom_eval | 2.396 | 25.200 | 25.218 | 0.916 | 1.591 | 1.082 | 9.030 | 1.423 |

Small-k buckets for `E_O33_best_smallk_odom_eval`:

| k | rot | tdir_abs | tmag_rel |
|---:|---:|---:|---:|
| 1 | 1.23 | 29.93 | 0.826 |
| 2 | 1.08 | 26.97 | 0.870 |
| 3 | 0.97 | 25.47 | 0.928 |
| 5 | 0.97 | 22.53 | 0.960 |
| 10 | 2.10 | 21.03 | 0.977 |
| 20 | 8.11 | 25.25 | 0.938 |

Decision:

- Do not adopt O33.
- O33 slightly improves over O32 on drift and `tdir_abs`, but still misses O30: drift is `1.423` vs `1.408`, and eval-only `tdir_abs=25.200°` still crosses the `25°` guard.
- Keep O30 as the drift-first small-k candidate.
- The tiny-dt tdir weighting family has now been tested in three forms: hard ignore, fixed weak weight, and continuous ramp. None beats O30, so the next improvement should move away from tiny-dt tdir weighting.

## O34 Plan

O34 moves away from tiny-dt tdir weighting and targets odometry/scale selection:

- Keep the O30 training recipe.
- Do not change model structure or training losses.
- Enable eval-only `t_mag` trajectory smoothing with `odom_eval_smooth_tmag_window=5`.
- Save `odom_trajectory_debug_latest.npz` for raw metric, smoothed metric, direction-only, and GT trajectories.
- Select odometry checkpoints with `odom_select_metric=odom_metric_smooth_tmag_drift`.
- Select small-k checkpoints with `smallk_select_metric=odom_metric_smooth_tmag_drift`.

Expected behavior:

- Raw `odom_metric_*` remains unchanged and is still reported.
- New `odom_metric_smooth_tmag_*` metrics show whether scale jitter is the main contributor to drift.
- If smoothed drift improves while raw pair-level metrics stay stable, O34 can identify a checkpoint better suited for odometry front-end use without adding learning risk.

Adoption rule:

- Prefer O34 only if smoothed selection also improves raw unified eval drift or provides a clearly useful inference-time smoothed trajectory.
- Reject O34 as a model candidate if it only improves smoothed drift but raw `odom_metric_drift` and pair-level metrics do not improve; in that case keep it as a diagnostic/inference option, not a new training baseline.

## O34 Probe Result

Before running a full O34 training pass, O30 was evaluated with `odom_eval_smooth_tmag_window=5` and trajectory debug enabled.

| eval | raw drift | smooth tmag drift | raw ATE | smooth tmag ATE | debug artifact |
|---|---:|---:|---:|---:|---|
| E_O30_smooth_tmag_w5_eval | 1.4078 | 1.4068 | 8.9215 | 8.9120 | `odom_trajectory_debug_latest.npz` |

Decision:

- Do not promote t_mag smoothing to a new model candidate yet.
- The improvement is real but tiny (`drift -0.0010`), so O30's drift is not mainly caused by high-frequency scale jitter.
- Keep the smoothing/debug path as an evaluation and visualization tool.
- If O34 is run later, treat it as checkpoint-selection/inference analysis, not as a replacement for O30 unless raw unified eval also improves.

## O35 Plan

O35 is a scale-bias diagnostic, not a deployable model change:

- Add optional `odom_eval_scale_fit=True`.
- During odometry eval, compute a per-chain least-squares scale factor using GT translation magnitudes and predicted metric magnitudes.
- Report `odom_metric_scale_fit_*` metrics separately from raw `odom_metric_*`.
- Save the scale-fit trajectory in `odom_trajectory_debug_latest.npz` when debug export is enabled.

Why:

- O30's t_mag smoothing barely changed drift, so scale jitter is likely not the main issue.
- Scale-fit oracle tells us whether a simple global scale bias is still a meaningful contributor.

Decision rule:

- If scale-fit drift drops substantially below O30 raw drift, pursue learned/calibrated scale correction.
- If scale-fit drift barely changes, prioritize translation direction, rotation accumulation, or trajectory composition diagnostics instead.

## O35 Probe Result

O30 was evaluated with `odom_eval_scale_fit=True`, `odom_eval_smooth_tmag_window=5`, and trajectory debug enabled.

| eval | raw drift | smooth tmag drift | scale-fit drift | raw ATE | scale-fit ATE | scale-fit factor mean |
|---|---:|---:|---:|---:|---:|---:|
| E_O30_scale_fit_oracle_eval | 1.4078 | 1.4068 | 1.2801 | 8.9215 | 10.0403 | 0.5436 |

Decision:

- Do not treat scale-fit as deployable performance; it uses GT translation magnitudes and is explicitly an oracle diagnostic.
- The result is still useful: simple t_mag smoothing does almost nothing, but oracle scale fitting reduces endpoint drift by about `9%`.
- ATE gets worse under scale-fit (`8.92 -> 10.04`), so this is not a clean win; scale bias helps endpoint drift but does not solve trajectory shape.
- Next reasonable model-side change should target calibrated scale prediction, not temporal smoothing. A low-risk candidate is adding an optional learned global `log_tmag_bias` initialized at zero, trained by the existing t_mag loss and evaluated under the same O30 protocol.

## O36 Plan

O36 tests that low-risk scale-calibration idea:

- Keep the O30 training recipe and small-k checkpoint selection.
- Add an optional learned global `log_tmag_bias`, initialized at `0.0`.
- Apply the bias to predicted `log_t_mag` before constructing `t_vec`.
- Do not change rotation, translation-direction supervision, matching losses, or odometry eval.

Purpose:

- O35 showed that a scale-fit oracle can reduce endpoint drift, so O36 checks whether a learned deployable global scale correction can capture some of that gain.
- Because the bias is optional and zero-initialized, default behavior remains unchanged when `use_tmag_global_bias=False`.

## O36 Result

O36 produced a small but consistent odometry improvement over O30 under the same eval-only protocol.

Training-time selected checkpoint:

| checkpoint | upd | drift | global tdir_abs | k=1/2/3 tdir_abs | k=1/2/3 tmag_rel | learned log bias |
|---|---:|---:|---:|---:|---:|---:|
| `O36/best_smallk_odom.pt` | 500 | 1.403 | 23.812 | 26.265 | 0.739 | -0.0009 |

Unified eval-only comparison:

| eval | rot | tdir_abs | local_A_abs | tmag_rel | tvec_l2 | RPE_rot | ATE | drift | norm_drift | scale-fit drift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E_O30_best_smallk_odom_eval | 2.382 | 24.776 | 24.794 | 0.918 | 1.592 | 1.039 | 8.922 | 1.408 | 1.122 | 1.280 |
| E_O36_best_smallk_odom_eval | 2.340 | 24.772 | 24.816 | 0.918 | 1.592 | 0.968 | 8.897 | 1.403 | 1.119 | 1.269 |

Small-k buckets for `E_O36_best_smallk_odom_eval`:

| k | rot | tdir_abs | tmag_rel |
|---:|---:|---:|---:|
| 1 | 1.06 | 29.53 | 0.828 |
| 2 | 0.92 | 26.57 | 0.874 |
| 3 | 0.83 | 25.05 | 0.930 |
| 5 | 0.90 | 22.08 | 0.961 |
| 10 | 2.16 | 20.53 | 0.978 |
| 20 | 8.23 | 24.84 | 0.940 |

Decision:

- Adopt O36 as the current drift-first small-k candidate because it improves eval-only drift from `1.408` to `1.403`, improves ATE from `8.922` to `8.897`, and keeps `tdir_abs=24.772°` under the `25°` guard.
- The gain is small. The learned bias stayed near zero, so this does not solve the deeper scale problem by itself.
- Keep C31 as the wide/stable baseline and teacher.
- Keep O28 as the conservative small-k candidate when pair-level scale cleanliness is more important than endpoint drift.
- Next improvement should not keep pushing global scale bias. The remaining bottleneck is still small-baseline direction and trajectory shape; consider trajectory-composition sanity/debug or a better scale-feature input rather than another global calibration knob.

## O37 Plan

O37 tests whether the scale head needs feature adaptation rather than only a scalar calibration:

- Keep the O36 recipe and odom/small-k checkpoint selection.
- Set `tmag_detach_features=False`.
- Keep `use_tmag_global_bias=True`, initialized at `0.0`.
- Do not change the pose losses, matching losses, anchor loss, eval protocol, or fine stage.

Why:

- O36 improved drift only slightly, and trajectory debug showed `tmag_pred` was still strongly compressed on the k=1 chain.
- Allowing t_mag gradients into the shared features is a minimal way to test whether scale prediction is bottlenecked by detached features.

## O37 Result

O37 is a clear improvement over the previous drift-first candidate.

Training-time selected checkpoint:

| checkpoint | upd | drift | global tdir_abs | k=1/2/3 tdir_abs | k=1/2/3 tmag_rel |
|---|---:|---:|---:|---:|---:|
| `O37/best_smallk_odom.pt` | 500 | 1.315 | 23.591 | 26.069 | 0.728 |

Unified eval-only comparison:

| eval | rot | tdir_abs | local_A_abs | tmag_rel | tvec_l2 | RPE_rot | ATE | drift | norm_drift | smooth drift | scale-fit drift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E_O36_best_smallk_odom_eval | 2.340 | 24.772 | 24.816 | 0.918 | 1.592 | 0.968 | 8.897 | 1.403 | 1.119 | 1.402 | 1.269 |
| E_O37_best_smallk_odom_eval | 2.392 | 24.604 | 24.627 | 0.927 | 1.601 | 1.076 | 8.321 | 1.315 | 1.049 | 1.314 | 1.281 |

Small-k buckets for `E_O37_best_smallk_odom_eval`:

| k | rot | tdir_abs | tmag_rel |
|---:|---:|---:|---:|
| 1 | 1.21 | 29.40 | 0.838 |
| 2 | 1.06 | 26.46 | 0.892 |
| 3 | 0.95 | 24.95 | 0.940 |
| 5 | 0.96 | 21.97 | 0.967 |
| 10 | 2.11 | 20.37 | 0.981 |
| 20 | 8.13 | 24.44 | 0.948 |

Decision:

- Adopt O37 as the current drift-first small-k candidate.
- O37 reduces eval-only drift from `1.403` to `1.315` and ATE from `8.897` to `8.321`, while keeping global `tdir_abs=24.604°`.
- It does not fix pair-level scale: `tmag_rel=0.927` is slightly worse than O36. The odometry gain likely comes from better accumulated trajectory behavior rather than a simple global t_mag error reduction.
- Keep C31 as the wide/stable baseline and teacher.
- Keep O28 as the conservative pair-level small-k candidate.
- Next checks should compare O37 on the wide protocol and inspect whether no-detach harms wide/stable behavior before making it a general default.

## O37 Wide Check

O37 should not replace C31 globally.

| eval | rot | tdir_abs | local_A_abs | tmag_rel | drift |
|---|---:|---:|---:|---:|---:|
| E_C31_wide_eval | 6.919 | 18.470 | 18.801 | 0.709 | 7.511 |
| E_O37_wide_eval | 7.629 | 20.684 | 20.363 | 0.936 | 7.180 |

Decision:

- O37 improves wide drift slightly, but C31 is still cleaner for wide/stable pair-level direction and scale.
- Keep the two-model interpretation: C31 is the wide/stable baseline and teacher; O37 is a small-k odometry candidate.

## O38 Plan

O38 tests whether stronger t_mag supervision can keep O37's drift gain while improving scale:

- Keep the O37 recipe.
- Set `w_tmag=0.2` instead of `0.1`.
- Keep `tmag_detach_features=False` and `use_tmag_global_bias=True`.
- Relax the odom selector's global tmag gate to `0.95`, matching the existing small-k gate.

## O38 Result

O38 is a slightly more aggressive drift-first candidate than O37, with tradeoffs.

Training-time selected checkpoint:

| checkpoint | upd | drift | global tdir_abs | k=1/2/3 tdir_abs | k=1/2/3 tmag_rel |
|---|---:|---:|---:|---:|---:|
| `O38/best_smallk_odom.pt` | 200 | 1.308 | 22.758 | 25.455 | 0.731 |

Unified eval-only comparison:

| eval | rot | tdir_abs | local_A_abs | tmag_rel | tvec_l2 | RPE_rot | ATE | drift | smooth drift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E_O37_best_smallk_odom_eval | 2.392 | 24.604 | 24.627 | 0.927 | 1.601 | 1.076 | 8.321 | 1.315 | 1.314 |
| E_O38_best_smallk_odom_eval | 2.728 | 24.772 | 25.175 | 0.921 | 1.595 | 1.710 | 8.326 | 1.308 | 1.307 |

Small-k buckets for `E_O38_best_smallk_odom_eval`:

| k | rot | tdir_abs | tmag_rel |
|---:|---:|---:|---:|
| 1 | 1.93 | 29.68 | 0.829 |
| 2 | 1.77 | 26.90 | 0.879 |
| 3 | 1.62 | 25.47 | 0.933 |
| 5 | 1.43 | 22.74 | 0.963 |
| 10 | 2.03 | 21.05 | 0.978 |
| 20 | 7.66 | 22.75 | 0.942 |

Decision:

- O38 has the best raw small-k endpoint drift so far: `1.308`.
- It is not a clean replacement for O37 because rotation and local-frame direction degrade: `rot=2.728` and `local_A_abs=25.175`.
- Use O38 only when endpoint drift is the primary metric. Keep O37 as the more balanced small-k odometry candidate.

## O39 Plan

O39 stops increasing `w_tmag` and instead tests whether stronger C31 direction anchoring can improve small-k direction without losing O37's trajectory behavior:

- Keep the O37 scale recipe: `w_tmag=0.1`, `tmag_detach_features=False`, and `use_tmag_global_bias=True`.
- Increase `w_tdir_anchor` from `2.0` to `3.0`.
- Keep the same C31 teacher/init checkpoint, small-k sampling, odom selection, and k=1/2/3 small-k selection gates.

Why:

- O38 showed that stronger scale supervision can reduce drift a little, but rotation and local-frame direction start to regress.
- O37's eval history shows a tradeoff: later checkpoints improve drift while k=1/2/3 direction worsens.
- O39 is a minimal direction/trajectory-shape probe around O37, not a new scale-head push.

## O39 Result

O39 is a direction-balanced small-k candidate. It does not beat O37's endpoint drift, but it improves global/local direction and all k=1/2/3 direction buckets while preserving nearly the same drift.

Training-time selected checkpoint:

| checkpoint | upd | drift | global tdir_abs | k=1/2/3 tdir_abs | k=1/2/3 tmag_rel |
|---|---:|---:|---:|---:|---:|
| `O39/best_smallk_odom.pt` | 500 | 1.316 | 22.860 | 25.376 | 0.728 |

Unified eval-only comparison:

| eval | rot | tdir_abs | local_A_abs | tmag_rel | tvec_l2 | RPE_rot | ATE | drift | smooth drift | scale-fit drift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E_O37_best_smallk_odom_eval | 2.392 | 24.604 | 24.627 | 0.927 | 1.601 | 1.076 | 8.321 | 1.315 | 1.314 | 1.281 |
| E_O39_best_smallk_odom_eval | 2.411 | 23.874 | 23.908 | 0.926 | 1.599 | 1.063 | 8.326 | 1.316 | 1.315 | 1.290 |

Small-k buckets for `E_O39_best_smallk_odom_eval`:

| k | rot | tdir_abs | tmag_rel |
|---:|---:|---:|---:|
| 1 | 1.29 | 28.79 | 0.837 |
| 2 | 1.16 | 25.83 | 0.889 |
| 3 | 1.05 | 24.30 | 0.939 |
| 5 | 0.97 | 21.27 | 0.966 |
| 10 | 2.15 | 19.61 | 0.980 |
| 20 | 8.22 | 23.41 | 0.949 |

Decision:

- Keep C31 as the wide/stable baseline.
- Keep O37 as the balanced small-k odometry default when the priority is the lowest combined drift/ATE among non-aggressive candidates.
- Use O39 when small-k direction quality matters more than the tiny `+0.001` drift difference versus O37.
- Keep O38 only as a drift-first probe; do not continue raising `w_tmag` from this branch because local-frame direction already regressed.
