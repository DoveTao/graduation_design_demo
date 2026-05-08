# Pano-ORB-VO Component Diagnostics

## Scope
This is a component-level diagnostic and does not change the baseline trajectory or reselect any candidate.
If an S5 reference row is included, it comes from S13 diagnostics and is not a same-evaluator full-coverage row.

## Trajectory-level reminder
- Pano-ORB-VO none/se3/sim3 ATE = `10.194695` / `4.137858` / `4.133322`
- Pano-ORB-VO none/se3/sim3 drift = `0.172360` / `0.173159` / `0.233378`
- Pano-ORB-VO path_ratio = `0.811334`
- S5 locked metrics: ATE=`7.352288`, drift=`1.327343`, path_ratio=`0.932379`
- SB2b caveat: S5 external TUM export is sparse diagnostic only and not used for full-coverage comparison.

## Pairwise component metrics
| method | source | num_valid_pairs | rot_mean_deg | rot_median_deg | rot_p90_deg | rot_max_deg | tdir_mean_deg | tdir_median_deg | tdir_p90_deg | tdir_max_deg | tdir_mean_cosine | tmag_mean_log_error | tmag_median_log_error | tmag_p90_log_error | tmag_max_log_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pano-ORB-VO | same-evaluator full-coverage TUM pairwise adjacent-pose diagnostic | 453 | 21.171064 | 0.705887 | 179.468812 | 179.998624 | 93.422068 | 94.202303 | 143.112335 | 172.996726 | -0.058091 | 1.513278 | 1.195670 | 3.236833 | 5.617485 |
| S5_S13_reference | S13 diagnostics reference only; not same external full-coverage evaluator | None | 20.715353 | 20.433789 | 21.371863 | 21.717051 | 72.349483 | 102.517940 | 110.918120 | 141.610521 | 0.226300 | 0.899289 | 0.793141 | 1.742507 | 1.985574 |

## Feature tracking diagnostics
- tracking_success_rate = `1.000000`
- mean_inliers = `191.668874`
- median_inliers = `88.000000`
- selected_view_yaw distribution = `{'0': 147, '90': 85, '180': 149, '270': 72}`

## Interpretation
Pano-ORB-VO provides a strong protocol-compatible classical geometry baseline at the pairwise component level. Its rotation and translation-direction diagnostics can be compared against the S5 S13 reference only qualitatively, because the S5 reference row is not a same-evaluator full-coverage trajectory export. The Pano-ORB-VO tmag diagnostics remain scale-policy-dependent, so trajectory-level path_ratio should still be used to judge scale stability. This report does not support a blanket superiority claim for S5, and it does not change the final-candidate status.

## Caveats
- panorama-derived virtual pinhole baseline
- not original fisheye baseline
- component metrics are pairwise diagnostics
- tmag diagnostics are scale-policy-dependent
- trajectory-level and pairwise metrics answer different questions
- S5 authoritative metrics remain locked clean evaluator metrics
- S5 external TUM export is sparse diagnostic only per SB2b
