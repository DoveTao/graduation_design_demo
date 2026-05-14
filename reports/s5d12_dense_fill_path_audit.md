# S5D12 Dense Fill Path Audit

## Executive summary
- final classification: `S5D12_DENSE_FILL_PROTOCOL_MISMATCH_IDENTIFIED`
- dense fill logic found in current branch: `False`
- nonselected source type: `unknown`
- dominant run source: `unknown`

## S5D11 recap
- S5D11 validation was clean: verify, project health, s6 eval-only, and unittest passed.
- non-selected edges had tmag median ratio 29.375292, p90 71.221354, p95 88.130943, max 335.512628.
- worst 20 tmag edges did not dominate path length; the dominant signal was the long run edges 72-360.

## Dense export/fill code path
- dense export script or restoration entry: `tools/restore_or_regenerate_s5_dense_tum.py`
- restore-from-branch artifact path observed: `True`
- pre-materialization available: `False`
- code search log: `logs/s5d12_dense_fill_code_search.log`

## Edge source map summary
- source map: `external_baselines/results/s5d12_dense_fill_path_audit/dense_fill_source_map.json`
- edges: `453`, selected: `132`, non-selected: `321`

## Dominant run 72-360 trace
- trace: `external_baselines/results/s5d12_dense_fill_path_audit/nonselected_run_72_360_trace.json`
- length: `289`
- estimated path: `39.022165`
- gt path: `1.557407`
- median tmag ratio: `32.228312`
- est step median/cv: `0.134033` / `0.064198`

## Pre vs post materialization metrics
- pre materialization available: `False`
- pre metrics path: `external_baselines/results/s5d12_dense_fill_path_audit/pre_materialization_edge_metrics.json`
- post metrics path: `external_baselines/results/s5d12_dense_fill_path_audit/post_materialization_edge_metrics.json`
- over-scaling stage: `unknown`

## Hypothesis check table
| hypothesis | supported | evidence |
| --- | --- | --- |
| `total_gap_translation_repeated_per_edge` | `None` | No verified pre-materialization gap translation vector is available.; run_est_step_median=0.1340332936139236; endpoint_dense_delta_norm_72_361=18.185937277201607 |
| `timestamp_unit_bug` | `False` | No timestamp-unit conversion logic was found on the restored dense artifact path.; run_tmag_ratio_median=32.22831211327435 is not a canonical ms/us factor. |
| `missing_division_by_gap_length` | `None` | Long-run behavior is consistent with gap-level materialization trouble, but the fill code/intermediate is unavailable.; run_length=289, run_tmag_ratio_median=32.22831211327435 |
| `endpoint_displacement_not_normalized` | `None` | Endpoint displacement cannot be compared to the missing pre-materialization vector.; post_materialized_est_step_cv=0.06419813865628834 |
| `selected_nonselected_protocol_mismatch` | `True` | selected_tmag_median=1.310038384597929; nonselected_tmag_median=29.375292370072714; dominant_run_tmag_median=32.22831211327435 |
| `nonselected_edges_not_direct_predictions` | `True` | S5D7 pairwise artifact contains 132 selected_k1 adjacent predictions, not all 453 adjacent edges.; Edges 72-360 are absent from the selected pairwise artifact except the boundary selected edge 71->72 and later selected run 361->427.; Current dense TUM was restored as an existing artifact, not generated from direct dense predictions in this audit. |

## Diagnosis
- nonselected_over_scaling_source: `protocol_mismatch`
- dense_fill_bug_suspected: `None`
- requires_code_change: `False`

## Recommended fixes / next actions
- Recover or instrument the original dense fill/materialization code path and log source_type per adjacent edge.
- For any gap fill, log endpoint delta, number of adjacent steps, emitted per-step delta, and dt units.
- Keep selected_k1 official pairwise metrics separate from external dense trajectory materialization metrics.
- Do not change S5 locked policy, manifest, evaluator defaults, or official metrics as part of S5D12.

## Caveats
- diagnostic only
- no official S5 result replacement
- S5 locked metrics/policy unchanged
- no GT used to generate predictions
