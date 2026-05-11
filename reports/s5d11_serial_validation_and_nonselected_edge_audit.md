# S5D11 Serial Validation and Non-selected Edge Audit

## Executive summary
- final classification: `S5D11_COMPLETE_VALIDATION_CLEAN`
- validation clean: `True`
- non-selected extreme tmag confirmed: `True`

## S5D10 recap
- selected edges: num_edges=132, path_length_fraction=0.393727, rot_mean=20.820703, tdir_mean=85.802014, tmag_median_ratio=1.310038
- non-selected edges: num_edges=321, path_length_fraction=0.606273, rot_mean=20.826980, tdir_mean=93.691277, tmag_median_ratio=29.375292
- S5D10 validation was blocked by concurrent eval-only CUDA memory pressure.

## CUDA concurrency / OOM audit
- concurrency_risk_before: `False`
- cuda_oom_status: `not_observed`
- concurrency audit artifact: `checkpoints/S5D11_validation_concurrency_audit.json`

## Serial validation guard design
- `scripts/run_serial_validation_guard.sh` wraps verify_final_candidate, project_health_check, s6 eval-only, and unittest under one `flock` lock.
- The guard sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and records nvidia-smi before/after GPU-heavy validation steps.
- It does not modify the official evaluator default behavior.

## Validation result
- verify_final_candidate: `True`
- project_health_check: `True`
- s6_eval_only: `True`
- unittest: `True`, test_count=`197`

## Selected vs non-selected edge metrics
| group | edges | path_fraction | rot_mean | tdir_mean | tdir_abs_mean | tmag_median | tmag_p90 | tmag_p95 | tmag_max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| selected | 132 | 0.393727 | 20.820703 | 85.802014 | 51.491081 | 1.310038 | 5.681601 | 5.889437 | 7.283208 |
| non-selected | 321 | 0.606273 | 20.826980 | 93.691277 | 59.506993 | 29.375292 | 71.221354 | 88.130943 | 335.512628 |

## Worst non-selected edges
- edge 289: tmag_ratio=335.512628, gt_step=0.000425, est_step=0.142510, tdir=87.297166, rot=20.585961, adjacent_to_selected=False
- edge 337: tmag_ratio=191.727295, gt_step=0.000769, est_step=0.147348, tdir=104.668028, rot=20.790782, adjacent_to_selected=False
- edge 313: tmag_ratio=176.839649, gt_step=0.000831, est_step=0.146868, tdir=30.819592, rot=20.867877, adjacent_to_selected=False
- edge 346: tmag_ratio=137.638457, gt_step=0.001063, est_step=0.146325, tdir=48.967684, rot=20.665172, adjacent_to_selected=False
- edge 286: tmag_ratio=117.039628, gt_step=0.001212, est_step=0.141896, tdir=112.696457, rot=20.827599, adjacent_to_selected=False

## Non-selected segment/run analysis
- run count: `18`
- longest run length: `289`
- top run est path length: `39.022165`
- nonselected_runs_dominate_path_length: `True`

## Diagnosis of dense path_ratio source
- most_likely_dense_pathratio_source: `long_runs`
- worst_edges_dominate_path_length: `False`
- evidence:
  - nonselected_tmag_median_ratio=29.375292370072714
  - nonselected_tmag_p95_ratio=88.13094277644502
  - nonselected_tmag_max_ratio=335.51262816809765
  - selected_path_fraction=0.3937270444892287
  - nonselected_path_fraction=0.6062729555107713
  - top20_tmag_edges_est_path_fraction=0.040395879183513395
  - longest_nonselected_run={'run_id': 13, 'start_edge_index': 72, 'end_edge_index': 360, 'run_length': 289, 'run_gt_path_length': 1.5574073003602866, 'run_est_path_length': 39.0221651562117, 'run_median_tmag_ratio': 32.22831211327435, 'run_mean_tdir_deg': 94.81600451595092, 'aligns_with_selected_k1_component_gap': True, 'bounded_by_selected_edge': True}

## Recommendations
- Keep the serial validation lock for verify/project_health/s6 eval-only runs before declaring CUDA stability.
- Audit the dense export path that fills non-selected adjacent edges; selected_k1 edges alone do not explain the large tmag median ratio.
- Add model/export diagnostics that report adjacent-edge tmag ratios before dense trajectory materialization.
- Consider a post-export consistency gate for non-selected edges, but do not change the locked S5 official evaluator or policy in S5D11.

## Caveats
- diagnostic only
- no official S5 result replacement
- S5 locked metrics/policy unchanged
- selected_k1 sparse protocol caveat
