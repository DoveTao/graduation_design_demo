# THESIS360 Results Tables

## Table 1. Pair-level Translation Component Metrics

| Method | split | signed_tdir_mean ↓ | anti_parallel_rate ↓ | tmag_median_ratio ↑/≈1 | path_ratio ↑/≈1 | notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| T57b legacy baseline | reference | 111.964932 | 0.674672 | 0.175156 | 0.148787 | recovered legacy external reference only |
| BASE360D trajectory-derived component baseline | test component | 128.402578 | 0.814895 | 0.039910 | 0.043542 | component metrics are trajectory-derived, not native pair-level predictions |
| FINAL360I main model | test | 45.264702 | 0.201699 | 0.834425 | 0.640352 | current pair-level thesis main model |
| SEQ360B scale variant | test adjacent/component | 45.325446 | 0.200158 | 1.697848 | 1.350477 | improves trajectory scale/path behavior but over-corrects pair-level magnitude |

Caveat: `BASE360D` component metrics are trajectory-derived from the official sequence pipeline rather than produced by the same pair-level forward interface as `FINAL360I`.

## Table 2. Trajectory-level Metrics

| Method | split | ATE none ↓ | ATE SE3 ↓ | ATE Sim3 ↓ | trajectory_path_ratio ≈1 | coverage | notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| TRAIN360E / FINAL360I direct composition | test | 222.568564 | 118.603779 | 27.564662 | 1.756343 | 1.00 | full trajectory export from adjacent pair composition |
| SEQ360B scale-smoothed trajectory | test | 139.227175 | 75.946913 | 27.567211 | 1.350237 | 1.00 | path and SE3 improve; Sim3 remains unchanged |
| BASE360D official 360DVO | test | 103.256252 | 79.295214 | 2.974466 | 0.043618 | partial in interface, official in pipeline | official sequence VO pipeline; not directly equivalent to pair-only inference |

## Table 3. Ablation / Diagnostic Outcomes

| Method | purpose | result | classification | kept as main? | lesson learned |
| --- | --- | --- | --- | --- | --- |
| FINAL360I | final pair-level main model selection | strongest retained balanced pair-level result | final_balanced_model | yes | pair-level match-free panoramic pose estimation is the main contribution |
| SEQ360B | lightweight scale / log_tmag correction | improves path ratio and SE3 ATE, but not Sim3 ATE | partial | no | scale drift can be reduced without solving trajectory shape |
| SEQ360A | local sequence consistency | no clear trajectory benefit | no_improvement | no | short-clip consistency alone did not reduce accumulated drift |
| STRUCT360C | rotation-aware fine refinement challenger | unstable evaluation and not promoted | evaluation_failed | no | structural refinement needs stronger stability before it can be compared fairly |

