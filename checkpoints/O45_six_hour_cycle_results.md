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
| O45_winner_cont300_scale_affine | unknown | 3 | reject | 25.4225 | 25.2000 | 25.4225 | 1.4596 | 1.3636 | 1.3366 | 1.3735 | 4.2346 | 0.9179 | direction_fail,path_up |
| O45a_scale_affine_probe_150_cont250 | scale_affine | 3 | reject | 25.0766 | 24.5889 | 25.0766 | 1.4403 | 1.3824 | 1.3213 | 1.3289 | 4.2453 | 0.9202 | direction_fail,path_up |
| O45b_seqturn_pair_probe_150 | seqturn_pair | 3 | reject | 23.2425 | 23.2044 | 23.2425 | 1.2907 | 1.2038 | 1.2409 | 1.0833 | 4.3085 | 0.9316 | direction_ok,path_not_up,drift_down |
| O45d_curriculum_k1_probe_150 | k1_curriculum | 0 | reject | 24.9393 | 24.8776 | 24.9393 | 1.3037 | 1.2781 | 1.2770 | 1.0296 | 4.5935 | 0.9344 | direction_ok,path_not_up,drift_down,turn_worse |
| O45c_seqturn_chain_probe_150 | seqturn_chain | -2 | reject | 23.9406 | 23.9059 | 23.9406 | 1.3385 | 1.2969 | 1.2996 | 1.1278 | 4.5852 | 0.9295 | direction_ok,path_not_up,turn_worse |

## Selected

1. `O45a_scale_affine_probe_150` (scale_affine), score=4, ckpt=`/home/dovetao/graduation_design_demo/checkpoints/O45a_scale_affine_probe_150/best_joint_local_A_abs.pt`
