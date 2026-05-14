# THESIS361 Final Results Tables

## Table 1. Pair-level / Component Translation Metrics

| Model | signed_tdir_mean ↓ | anti_parallel_rate ↓ | tmag_median_ratio ≈ 1 | path_ratio ≈ 1 | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| T57b legacy baseline | 111.964932 | 0.674672 | 0.175156 | 0.148787 | Recovered legacy external reference only. |
| BASE360D trajectory-derived component baseline | 128.402578 | 0.814895 | 0.039910 | 0.043542 | Derived from official sequence trajectories rather than native pair-level predictions. |
| FINAL360I main model | 45.264702006380205 | 0.20169893322797314 | 0.8344251368086006 | 0.6403519796204528 | Current pair-level thesis main model. |
| SEQ360B scale variant | 45.32544604265589 | 0.20015760441292357 | 1.6978484631707425 | 1.3504765883324317 | Improves trajectory scale/path behavior but over-corrects pair-level magnitude. |

**Table note.** `BASE360D` component metrics are trajectory-derived and therefore only partially comparable with native pair-level predictions. Pair-level component metrics and trajectory-level ATE evaluate different aspects of the system.

## Table 2. Trajectory-level Metrics

| Model | ATE none ↓ | ATE SE3 ↓ | ATE Sim3 ↓ | trajectory_path_ratio ≈ 1 | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| TRAIN360E / FINAL360I direct composition | 222.56856382785097 | 118.6037794846689 | 27.564661865900444 | 1.756343083453392 | Full reported test coverage; direct adjacent-pair composition still drifts. |
| SEQ360B scale-smoothed trajectory | 139.22717463963536 | 75.94691348103409 | 27.567211313835486 | 1.3502369615185652 | Path ratio and SE3 ATE improve, but Sim3 ATE is unchanged. |
| BASE360D official 360DVO | 103.256252 | 79.295214 | 2.974466 | 0.043618 | Official sequence VO pipeline; trajectory-level comparison is meaningful, but not interface-identical to pair-only inference. |

**Table note.** Pair-level metrics and trajectory-level ATE quantify different failure modes. Better pair-level translation components do not automatically imply better full-trajectory ATE after direct sequential composition.

## Table 3. Diagnostic / Ablation Outcomes

| Method | Purpose | Outcome | Classification | Used as final model? | Lesson |
| --- | --- | --- | --- | --- | --- |
| FINAL360I | Final pair-level main model selection | Best retained balanced pair-level result | final_balanced_model | Yes | Pair-level match-free panoramic relative pose estimation is the main contribution. |
| SEQ360B | Lightweight scale / log_tmag correction | Improves path ratio and ATE SE3, but not ATE Sim3 | partial | No | Scale/path drift can be reduced without solving trajectory shape. |
| SEQ360A | Local sequence consistency | No clear trajectory benefit | no_improvement | No | Short-clip consistency alone did not reduce accumulated drift. |
| STRUCT360C | Rotation-aware fine refinement challenger | Unstable evaluation, not promoted | evaluation_failed | No | Structural refinement requires much stronger stability before fair comparison. |
