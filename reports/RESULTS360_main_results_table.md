# RESULTS360 main results table

## 1. Executive summary
- current visible comparison set: `FINAL360I`, `TRAIN360E`, `SEQ360B`, `BASE360D`, `T57b`
- current pair-level thesis main model: `FINAL360I_struct360b_final_selected`
- current trajectory evaluation path: `TRAIN360E_sequence_trajectory_export_and_ATE_eval`
- retained sequence-scale variant: `SEQ360B_lightweight_scale_smoothing_head`
- external baseline: `BASE360D` (`HKUST official 360DVO`)
- legacy external reference: `T57b`

## 2. Pair-level / component metrics

| item | split | signed_tdir_mean_deg | anti_parallel_rate | tmag_median_ratio | path_ratio | notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `T57b` legacy baseline | reference | 111.964932 | 0.674672 | 0.175156 | 0.148787 | recovered legacy ERP image-pair reference only |
| `BASE360D` official external baseline | test component | 128.402578 | 0.814895 | 0.039910 | 0.043542 | trajectory-derived component metrics from official `360DVO` output |
| `FINAL360I` main pair-level model | test | 45.264702 | 0.201699 | 0.834425 | 0.640352 | current thesis main model |
| `SEQ360B` retained sequence-scale variant | test adjacent/component | 45.325446 | 0.200158 | 1.697848 | 1.350477 | path improves, but pair-scale over-corrects |

## 3. Trajectory metrics

| item | split | ATE none | ATE se3 | ATE sim3 | trajectory_path_ratio | notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `T57b` legacy baseline | reference | 28.308558 | 4.353307 | 2.065264 | 0.148787 | external reference only; not current mainline |
| `BASE360D` official external baseline | test | 103.256252 | 79.295214 | 2.974466 | 0.043618 | official sequence pipeline, not pair-only inference |
| `TRAIN360E` composed `FINAL360I` trajectory | test | 222.568564 | 118.603779 | 27.564662 | 1.756343 | exposes trajectory drift not visible in isolated pair metrics |
| `SEQ360B` retained sequence-scale variant | test | 139.227175 | 75.946913 | 27.567211 | 1.350237 | improves path / SE3 ATE but not Sim3 shape error |

## 4. Current interpretation
- `FINAL360I` is the kept pair-level main model because it preserves the best balanced direction/scale behavior among the retained in-repo learned models.
- `TRAIN360E` is the kept trajectory evaluation path and should be read as a sequence-level stress test of composed adjacent predictions.
- `SEQ360B` is retained because it meaningfully improves trajectory path ratio and SE3 ATE, even though it over-corrects pair-level translation magnitude.
- `BASE360D` remains the official external baseline and must be interpreted with its trajectory-derived component-metric caveat.
- `T57b` remains only as a legacy recovered reference, not as a current training or deployment candidate.

## 5. Status-only diagnostic attempts
- `SEQ360A`: no_improvement; retained only as `reports/SEQ360A_status_summary.md`
- `STRUCT360C`: evaluation_failed; retained only as `reports/STRUCT360C_status_summary.md`

## 6. Compliance note
- no metric values were modified during MAINT15; this table only trims legacy comparison labels and focuses the visible results set on the kept mainline.
