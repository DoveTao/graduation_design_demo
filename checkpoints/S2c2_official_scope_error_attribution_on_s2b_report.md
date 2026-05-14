# S2c2 Official-Scope Error Attribution On S2b

## S2b baseline
- base checkpoint: `checkpoints/T57b_no_dt_multiscale_tmag_head_400/final.pt`
- policy: `checkpoints/S2b_clean_fine_rot_policy.json`
- explicit-cfg / unexpected=0: True
- fine_rot / fine_tdir / fine_tmag = 0.45 / 0.0 / 0.0
- selected_k / num_pairs / num_chains = 1 / 132 / 19
- official report drift / ATE / path_ratio = 1.327402 / 7.352371 / 0.934984

## Official scope definition
- official metric field: `odom_shape_metric_mean_path_length_ratio`
- official aggregation source: `debug_chain_summaries`
- current eval config serialized `num_debug_chains = 1`
- primary official debug chain scene_seq: `('scene01', 'seq03')`
- all-chain numbers below are secondary diagnostics only and must not be compared directly to the official path_ratio.

## Official-scope oracle diagnostics
- `drift / ATE` below follow the official all-chain odom payload.
- `path_ratio` below follows the official debug-chain-scoped `debug_chain_summaries` payload.
- step / turn / tdir / tmag breakdown below is computed on the official debug chain only.
| variant | drift_all_chain | ATE_all_chain | path_ratio_debug_chain | RPE_rot_all_chain | RPE_trans_dir_all_chain | RPE_trans_mag_all_chain | mean_step_position_error_debug | mean_rot_error_debug | mean_tdir_error_debug | mean_tmag_ratio_debug | turn_error_debug | num_steps_debug | num_pairs_all_chain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| official_current | 1.327402 | 7.352371 | 0.934984 | 20.715353 | 72.349483 | 1.426862 | 6.730752 | 21.164677 | 30.699865 | 1.641434 | 15.853794 | 40 | 132 |
| oracle_R | 1.516962 | 10.590301 | 0.934986 | 0.012936 | 72.349483 | 1.426862 | 2.014190 | 0.009257 | 30.699865 | 1.641434 | 7.578068 | 40 | 132 |
| oracle_tdir | 1.375470 | 7.658319 | 0.934985 | 20.715353 | 0.000000 | 1.426862 | 6.728818 | 21.164677 | 0.000000 | 1.641434 | 17.166941 | 40 | 132 |
| oracle_R_tdir | 0.304305 | 0.911024 | 0.934986 | 0.012936 | 0.000000 | 1.426862 | 1.088204 | 0.009257 | 0.000000 | 1.641434 | 0.000061 | 40 | 132 |
| oracle_tmag | 1.250258 | 7.569829 | 0.999998 | 20.715353 | 72.349483 | 0.000000 | 7.219007 | 21.164677 | 30.699865 | 1.000000 | 15.853892 | 40 | 132 |
| oracle_all | 0.000000 | 0.000000 | 1.000000 | 0.012936 | 0.000000 | 0.000000 | 0.000000 | 0.009257 | 0.000000 | 1.000000 | 0.000000 | 40 | 132 |

## Official debug-chain step breakdown
| step_idx | frame_i | frame_j | gt_tmag | pred_tmag | tmag_ratio | rot_error | tdir_error | step_position_error | cumulative_position_error | turn_error |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 1 | 0.025608 | 0.133864 | 5.227432 | 21.025347 | 88.586082 | 0.143285 | 0.143285 | nan |
| 1 | 1 | 2 | 0.126815 | 0.174337 | 1.374739 | 21.030870 | 56.788658 | 0.208805 | 0.208805 | nan |
| 2 | 2 | 3 | 0.218640 | 0.174367 | 0.797506 | 21.059682 | 50.690519 | 0.170392 | 0.208805 | nan |
| 3 | 3 | 4 | 0.657779 | 0.531463 | 0.807966 | 21.046150 | 33.148522 | 0.356046 | 0.356046 | nan |
| 4 | 4 | 5 | 0.453456 | 0.349022 | 0.769693 | 21.087355 | 8.042539 | 0.910982 | 0.910982 | nan |
| 5 | 5 | 6 | 0.283105 | 0.174758 | 0.617291 | 21.085695 | 18.077417 | 1.324735 | 1.324735 | nan |
| 6 | 6 | 7 | 0.335102 | 0.348310 | 1.039417 | 20.940995 | 30.101580 | 1.984998 | 1.984998 | nan |
| 7 | 7 | 8 | 0.438464 | 0.340424 | 0.776402 | 20.770633 | 37.814341 | 2.706572 | 2.706572 | nan |
| 8 | 8 | 9 | 0.211877 | 0.170553 | 0.804961 | 20.943245 | 33.789881 | 3.039944 | 3.039944 | nan |
| 9 | 9 | 10 | 0.740239 | 0.488770 | 0.660287 | 21.101861 | 23.770018 | 4.057824 | 4.057824 | nan |
| 10 | 10 | 11 | 0.605411 | 0.485735 | 0.802322 | 21.200252 | 27.399612 | 4.887694 | 4.887694 | nan |
| 11 | 11 | 12 | 0.181405 | 0.161063 | 0.887862 | 21.181093 | 30.312337 | 5.119694 | 5.119694 | nan |
| 12 | 12 | 13 | 0.647502 | 0.489743 | 0.756358 | 21.225417 | 30.531599 | 5.807222 | 5.807222 | nan |
| 13 | 13 | 14 | 0.403708 | 0.318395 | 0.788677 | 21.376114 | 24.932680 | 6.144486 | 6.144486 | nan |
| 14 | 14 | 15 | 0.641616 | 0.485837 | 0.757208 | 21.326486 | 24.556314 | 6.583723 | 6.583723 | nan |
| 15 | 15 | 16 | 0.569083 | 0.482621 | 0.848067 | 21.252733 | 19.525015 | 6.836975 | 6.836975 | nan |
| 16 | 16 | 17 | 0.434640 | 0.316573 | 0.728357 | 21.279234 | 19.597075 | 7.026152 | 7.026152 | nan |
| 17 | 17 | 18 | 0.335564 | 0.310365 | 0.924906 | 21.262439 | 19.052258 | 7.123342 | 7.123342 | nan |
| 18 | 18 | 19 | 0.314384 | 0.310125 | 0.986452 | 21.251847 | 19.677379 | 7.237606 | 7.237606 | nan |
| 19 | 19 | 20 | 0.272347 | 0.156233 | 0.573654 | 21.415219 | 22.877152 | 7.429671 | 7.429671 | nan |
| 20 | 20 | 21 | 0.379244 | 0.312627 | 0.824342 | 21.316847 | 24.819822 | 7.737642 | 7.737642 | nan |
| 21 | 21 | 22 | 0.222886 | 0.157749 | 0.707757 | 21.264253 | 25.420056 | 7.973606 | 7.973606 | nan |
| 22 | 22 | 23 | 0.331108 | 0.314932 | 0.951148 | 21.252974 | 26.349267 | 8.424031 | 8.424031 | nan |
| 23 | 23 | 24 | 0.222624 | 0.158372 | 0.711387 | 21.291513 | 28.751730 | 8.729055 | 8.729055 | nan |
| 24 | 24 | 25 | 0.163964 | 0.162408 | 0.990510 | 21.267685 | 28.673305 | 9.007175 | 9.007175 | nan |
| 25 | 25 | 26 | 0.112923 | 0.162372 | 1.437890 | 21.234129 | 30.716567 | 9.246058 | 9.246058 | nan |
| 26 | 26 | 27 | 0.142785 | 0.164611 | 1.152859 | 21.086403 | 32.727585 | 9.507411 | 9.507411 | nan |
| 27 | 27 | 28 | 0.157495 | 0.164791 | 1.046327 | 21.103209 | 33.857308 | 9.760766 | 9.760766 | nan |
| 28 | 28 | 29 | 0.073394 | 0.129911 | 1.770050 | 21.294982 | 34.357365 | 9.882375 | 9.882375 | nan |
| 29 | 29 | 30 | 0.090613 | 0.129970 | 1.434339 | 21.294791 | 35.688754 | 9.989848 | 9.989848 | nan |
| 30 | 30 | 31 | 0.054941 | 0.126320 | 2.299193 | 21.029141 | 32.361638 | 10.028985 | 10.028985 | nan |
| 31 | 31 | 32 | 0.038234 | 0.126221 | 3.301264 | 21.045880 | 33.160094 | 10.022352 | 10.028985 | nan |
| 32 | 32 | 33 | 0.047914 | 0.126765 | 2.645684 | 21.191817 | 29.642894 | 9.995588 | 10.028985 | nan |
| 33 | 33 | 34 | 0.062384 | 0.126859 | 2.033518 | 21.130382 | 30.163680 | 9.971247 | 10.028985 | nan |
| 34 | 34 | 35 | 0.029265 | 0.126523 | 4.323386 | 21.178923 | 31.696634 | 9.921218 | 10.028985 | nan |
| 35 | 35 | 36 | 0.045480 | 0.127698 | 2.807805 | 21.146363 | 30.138060 | 9.904744 | 10.028985 | nan |
| 36 | 36 | 37 | 0.035010 | 0.127774 | 3.649641 | 21.147814 | 30.168339 | 9.908948 | 10.028985 | nan |
| 37 | 37 | 38 | 0.027076 | 0.129298 | 4.775385 | 21.095627 | 30.887039 | 9.944973 | 10.028985 | nan |
| 38 | 38 | 39 | 0.036330 | 0.129740 | 3.571163 | 21.149937 | 28.353923 | 10.031703 | 10.031703 | nan |
| 39 | 39 | 40 | 0.030274 | 0.130000 | 4.294139 | 21.201749 | 30.789579 | 10.142222 | 10.142222 | nan |

## Top bad steps by cumulative_position_error
- step 39 (i=39, j=40): pos_err=10.142222, rot_err=21.201749, tdir_err=30.789579
- step 38 (i=38, j=39): pos_err=10.031703, rot_err=21.149937, tdir_err=28.353923
- step 30 (i=30, j=31): pos_err=10.028985, rot_err=21.029141, tdir_err=32.361638
- step 31 (i=31, j=32): pos_err=10.022352, rot_err=21.045880, tdir_err=33.160094
- step 32 (i=32, j=33): pos_err=9.995588, rot_err=21.191817, tdir_err=29.642894
- step 29 (i=29, j=30): pos_err=9.989848, rot_err=21.294791, tdir_err=35.688754
- step 33 (i=33, j=34): pos_err=9.971247, rot_err=21.130382, tdir_err=30.163680
- step 37 (i=37, j=38): pos_err=9.944973, rot_err=21.095627, tdir_err=30.887039
- step 34 (i=34, j=35): pos_err=9.921218, rot_err=21.178923, tdir_err=31.696634
- step 36 (i=36, j=37): pos_err=9.908948, rot_err=21.147814, tdir_err=30.168339

## Top bad steps by step_position_error
- step 39 (i=39, j=40): step_pos_err=10.142222, tmag_ratio=4.294139
- step 38 (i=38, j=39): step_pos_err=10.031703, tmag_ratio=3.571163
- step 30 (i=30, j=31): step_pos_err=10.028985, tmag_ratio=2.299193
- step 31 (i=31, j=32): step_pos_err=10.022352, tmag_ratio=3.301264
- step 32 (i=32, j=33): step_pos_err=9.995588, tmag_ratio=2.645684
- step 29 (i=29, j=30): step_pos_err=9.989848, tmag_ratio=1.434339
- step 33 (i=33, j=34): step_pos_err=9.971247, tmag_ratio=2.033518
- step 37 (i=37, j=38): step_pos_err=9.944973, tmag_ratio=4.775385
- step 34 (i=34, j=35): step_pos_err=9.921218, tmag_ratio=4.323386
- step 36 (i=36, j=37): step_pos_err=9.908948, tmag_ratio=3.649641

## Top bad steps by tdir_error
- step 0 (i=0, j=1): tdir_err=88.586082, rot_err=21.025347, pos_err=0.143285
- step 1 (i=1, j=2): tdir_err=56.788658, rot_err=21.030870, pos_err=0.208805
- step 2 (i=2, j=3): tdir_err=50.690519, rot_err=21.059682, pos_err=0.170392
- step 7 (i=7, j=8): tdir_err=37.814341, rot_err=20.770633, pos_err=2.706572
- step 29 (i=29, j=30): tdir_err=35.688754, rot_err=21.294791, pos_err=9.989848
- step 28 (i=28, j=29): tdir_err=34.357365, rot_err=21.294982, pos_err=9.882375
- step 27 (i=27, j=28): tdir_err=33.857308, rot_err=21.103209, pos_err=9.760766
- step 8 (i=8, j=9): tdir_err=33.789881, rot_err=20.943245, pos_err=3.039944
- step 31 (i=31, j=32): tdir_err=33.160094, rot_err=21.045880, pos_err=10.022352
- step 3 (i=3, j=4): tdir_err=33.148522, rot_err=21.046150, pos_err=0.356046

## Top bad steps by rot_error
- step 19 (i=19, j=20): rot_err=21.415219, tdir_err=22.877152, pos_err=7.429671
- step 13 (i=13, j=14): rot_err=21.376114, tdir_err=24.932680, pos_err=6.144486
- step 14 (i=14, j=15): rot_err=21.326486, tdir_err=24.556314, pos_err=6.583723
- step 20 (i=20, j=21): rot_err=21.316847, tdir_err=24.819822, pos_err=7.737642
- step 28 (i=28, j=29): rot_err=21.294982, tdir_err=34.357365, pos_err=9.882375
- step 29 (i=29, j=30): rot_err=21.294791, tdir_err=35.688754, pos_err=9.989848
- step 23 (i=23, j=24): rot_err=21.291513, tdir_err=28.751730, pos_err=8.729055
- step 16 (i=16, j=17): rot_err=21.279234, tdir_err=19.597075, pos_err=7.026152
- step 24 (i=24, j=25): rot_err=21.267685, tdir_err=28.673305, pos_err=9.007175
- step 21 (i=21, j=22): rot_err=21.264253, tdir_err=25.420056, pos_err=7.973606

## Official-scope vs all-chain secondary diagnostic
- official debug-chain path_ratio: 0.934984
- official all-chain ATE/drift: 7.352371 / 1.327402
- all-chain secondary diagnostic: weighted path_ratio=1.184835, mean path_ratio=4.987250, chain count=19
- official metric is debug-chain scoped in this eval configuration; all-chain statistics are diagnostic and should not be compared directly as official metrics.

## Correlation analysis
- corr(ATE, tdir error) = `nan`
- corr(ATE, rot error) = `nan`
- corr(ATE, path_ratio error) = `nan`
- corr(ATE, turn error) = `nan`

## Final classification
- classification: `OFFICIAL-SCOPE-R-TDIR-COUPLED-ERROR`

## Next step
- 进入 `S2d_joint_fine_rot_tdir_policy_audit_on_s2b`，联合扫 `fine_rot = 0.35, 0.40, 0.45, 0.50` 与 `fine_tdir = 0.00, 0.05, 0.10, 0.15, 0.20`，保持 `fine_tmag = 0.0`。
