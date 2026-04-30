# O45 Six-Hour Odometry Cycle Results

## Baseline

| metric | value |
|---|---:|
| tdir_abs | 23.8734 |
| local_A_abs | 23.9084 |
| best_smallk_odom_tdir_abs | 25.3755 |
| drift | 1.3164 |
| scale_fit_drift | 1.2904 |
| dtcalib_drift | 1.2852 |
| path_length_ratio | 1.2009 |
| turn_sum_abs_err | 4.3149 |
| tmag_rel_err | 0.9261 |

## Ranking

| experiment | arm | score | decision | tdir_abs | local_A_abs | best_smallk_odom_tdir_abs | drift | scale_fit_drift | dtcalib_drift | path_length_ratio | turn_sum_abs_err | tmag_rel_err | reasons |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| O45a_scale_affine_probe_150 | scale_affine | 4 | continue | 24.0503 | 23.5738 | 24.0503 | 1.3588 | 1.2771 | 1.2287 | 1.3600 | 4.2267 | 0.9186 | direction_ok,path_up |

## Selected

1. `O45a_scale_affine_probe_150` (scale_affine), score=4, ckpt=`/home/dovetao/graduation_design_demo/checkpoints/O45a_scale_affine_probe_150/best_joint_local_A_abs.pt`
