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
