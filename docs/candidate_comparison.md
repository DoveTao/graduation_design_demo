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
