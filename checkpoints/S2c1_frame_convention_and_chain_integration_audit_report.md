# S2c1 Frame Convention And Chain Integration Audit

## Official odom metric source
- official reported path_ratio field: `odom_shape_metric_mean_path_length_ratio`
- implementation source: `train_mvp.py::eval_odometry_sequence`
- field value is assigned from `debug_metric_path_ratio`
- `debug_metric_path_ratio = _finite_mean_nested(debug_chain_summaries, 'shape_metric', 'path_length_ratio')`
- `debug_max_chains = 1` in this S2b repro output, so official path_ratio uses only the first debug chain
- official debug chain scene_seq: `('scene01', 'seq03')`
- selected_k filtering: `selected_k = 1`
- aggregation mode: mean over debug chains, not all manifest chains; weighted alternative exists as `odom_shape_metric_path_weighted_path_length_ratio`

## Official chain selection / aggregation
- manifest chains at selected_k=1: `19`
- manifest pairs at selected_k=1: `132`
- official debug chains serialized: `1`
- official payload drift/ATE come from all selected chains
- official payload path_ratio comes only from debug_chain_summaries

## Official vs S2c mismatch
- official reported path_ratio = `0.934984`
- reproduced official debug-chain path_ratio = `0.934984`
- all-chain weighted path_ratio = `1.184835`
- all-chain mean path_ratio = `4.987250`
- mismatch reason: official path_ratio is computed from debug_chain_summaries only; debug_max_chains=1, so only chain `('scene01', 'seq03')` contributes, while S2c all-chain aggregation used all 19 chains including many tiny 1-2 step residual chains.

## Official integration reproduction
- reproduced official drift = `1.327402` vs report `1.327402`
- reproduced official ATE = `7.352371` vs report `7.352371`
- reproduced official path_ratio (debug-chain) = `0.934984` vs report `0.934984`
- first divergence point between official and original S2c is the path_ratio aggregation scope, not pose composition math.

## Frame convention variants
| variant | drift | ATE | path_ratio | path_weighted_ratio | mean_path_ratio | RPE_rot | RPE_trans_dir | selected_k | num_chains | num_pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| official_current | 1.327402 | 7.352371 | 1.184835 | 1.184835 | 4.987250 | 20.715353 | 72.349483 | 1 | 19 | 132 |
| pred_R_pred_tdir | 1.327402 | 7.352371 | 1.184835 | 1.184835 | 4.987249 | 20.715353 | 72.349483 | 1 | 19 | 132 |
| pred_RT_pred_tdir | 1.356046 | 7.382440 | 1.184835 | 1.184835 | 4.987249 | 20.833085 | 72.349483 | 1 | 19 | 132 |
| pred_R_R_tdir | 1.330083 | 7.382643 | 1.184835 | 1.184835 | 4.987249 | 20.715353 | 67.689367 | 1 | 19 | 132 |
| pred_R_RT_tdir | 1.330914 | 7.326646 | 1.184835 | 1.184835 | 4.987250 | 20.715353 | 80.502216 | 1 | 19 | 132 |
| pred_RT_R_tdir | 1.395232 | 7.503425 | 1.184835 | 1.184835 | 4.987249 | 20.833085 | 67.689367 | 1 | 19 | 132 |
| pred_RT_RT_tdir | 1.318708 | 7.261537 | 1.184835 | 1.184835 | 4.987249 | 20.833085 | 80.502216 | 1 | 19 | 132 |
| invert_comp_order | 1.379866 | 7.627665 | 1.063757 | 1.063757 | 4.976716 | 20.715353 | 72.349483 | 1 | 19 | 132 |
| swap_AB_direction | 1.558574 | 8.053178 | 1.184835 | 1.184835 | 4.987249 | 20.833085 | 99.497784 | 1 | 19 | 132 |
| use_local_A_tdir | 1.330914 | 7.326646 | 1.184835 | 1.184835 | 4.987250 | 20.715353 | 80.502216 | 1 | 19 | 132 |
| use_output_frame_tdir | 1.327402 | 7.352371 | 1.184835 | 1.184835 | 4.987249 | 20.715353 | 72.349483 | 1 | 19 | 132 |

## R / tdir coupling oracle
| variant | drift | ATE | path_ratio | RPE_rot | RPE_trans_dir | RPE_trans_mag |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| oracle_R | 1.516962 | 10.590301 | 1.184835 | 0.012936 | 72.349483 | 1.426862 |
| oracle_tdir | 1.375470 | 7.658319 | 1.184836 | 20.715353 | 0.000000 | 1.426862 |
| oracle_R_tdir | 0.304305 | 0.911024 | 1.184835 | 0.012936 | 0.000000 | 1.426862 |
| oracle_tmag | 1.250258 | 7.569829 | 1.000000 | 20.715353 | 72.349483 | 0.000000 |
| oracle_all | 0.000000 | 0.000000 | 1.000000 | 0.012936 | 0.000000 | 0.000000 |
| gtR_predtdir_gtRframe | 1.647691 | 11.740566 | 1.184835 | 0.012936 | 80.449368 | 1.426862 |
| predR_gttdir_predRframe | 1.364868 | 7.645666 | 1.184835 | 20.715353 | 20.261813 | 1.426862 |
| gtR_gttdir_pred_order | 0.087561 | 0.502584 | 1.001922 | 0.012936 | 0.000000 | 0.000000 |
| predR_predtdir_gt_order | 1.379866 | 7.627665 | 1.063757 | 20.715353 | 72.349483 | 1.426862 |
| gtR_predtdir_A_local | 1.647961 | 11.749637 | 1.184835 | 0.012936 | 80.502216 | 1.426862 |

## Chain-level breakdown
| chain_id | scene_seq | num_steps | gt_path_length | pred_path_length | path_ratio | ATE | drift | mean_rot_error | mean_tdir_error | mean_tmag_ratio | mean_turn_error | max_step_error | worst_step_index |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 14 | scene01/seq03 | 67 | 13.013749 | 15.301692 | 1.175810 | 8.514647 | 12.293643 | 20.305119 | 108.815313 | 1.523941 | 18.295176 | 12.715776 | 61 |
| 0 | scene01/seq03 | 40 | 10.200697 | 9.537485 | 0.934984 | 7.544831 | 10.142222 | 21.164677 | 30.699865 | 1.641434 | 15.853794 | 10.142222 | 39 |
| 18 | scene01/seq03 | 5 | 0.131634 | 0.765756 | 5.817300 | 0.383712 | 0.543933 | 20.441778 | 65.637252 | 5.850247 | 17.267089 | 0.543933 | 4 |
| 16 | scene01/seq03 | 2 | 0.044554 | 0.308256 | 6.918752 | 0.213046 | 0.267851 | 20.408929 | 64.837226 | 6.936126 | 16.390360 | 0.267851 | 1 |
| 15 | scene01/seq03 | 1 | 0.028136 | 0.155806 | 5.537505 | 0.171752 | 0.171752 | 20.416434 | 100.855226 | 5.537505 | nan | 0.171752 | 0 |
| 1 | scene01/seq03 | 2 | 0.059278 | 0.256570 | 4.328279 | 0.168329 | 0.212860 | 21.255264 | 28.510628 | 4.333452 | 13.980722 | 0.212860 | 1 |
| 11 | scene01/seq03 | 2 | 0.044525 | 0.250193 | 5.619178 | 0.161862 | 0.203346 | 21.667990 | 21.546713 | 5.660168 | 19.037109 | 0.203346 | 1 |
| 3 | scene01/seq03 | 2 | 0.056228 | 0.250103 | 4.448030 | 0.159829 | 0.201423 | 21.319017 | 24.109372 | 4.483121 | 15.082032 | 0.201423 | 1 |
| 17 | scene01/seq03 | 1 | 0.026840 | 0.153650 | 5.724691 | 0.134571 | 0.134571 | 20.423945 | 61.412552 | 5.724692 | nan | 0.134571 | 0 |
| 6 | scene01/seq03 | 1 | 0.021284 | 0.127553 | 5.992971 | 0.107488 | 0.107488 | 21.378853 | 27.959943 | 5.992975 | nan | 0.107488 | 0 |
| 5 | scene01/seq03 | 1 | 0.023888 | 0.128590 | 5.382983 | 0.106263 | 0.106263 | 21.501832 | 29.169221 | 5.382984 | nan | 0.106263 | 0 |
| 13 | scene01/seq03 | 1 | 0.022211 | 0.126912 | 5.714023 | 0.106159 | 0.106159 | 21.470225 | 32.309924 | 5.714021 | nan | 0.106159 | 0 |
| 2 | scene01/seq03 | 1 | 0.027600 | 0.127748 | 4.628634 | 0.105067 | 0.105067 | 21.385366 | 27.893172 | 4.628634 | nan | 0.105067 | 0 |
| 12 | scene01/seq03 | 1 | 0.022285 | 0.126856 | 5.692477 | 0.105048 | 0.105048 | 21.515914 | 22.639928 | 5.692476 | nan | 0.105048 | 0 |
| 8 | scene01/seq03 | 1 | 0.021827 | 0.125987 | 5.772051 | 0.104629 | 0.104629 | 21.585068 | 18.217242 | 5.772052 | nan | 0.104629 | 0 |
| 10 | scene01/seq03 | 1 | 0.021416 | 0.124694 | 5.822346 | 0.103920 | 0.103920 | 21.578879 | 26.319392 | 5.822348 | nan | 0.103920 | 0 |
| 7 | scene01/seq03 | 1 | 0.027550 | 0.129284 | 4.692648 | 0.103659 | 0.103659 | 21.508023 | 26.259149 | 4.692649 | nan | 0.103659 | 0 |
| 4 | scene01/seq03 | 1 | 0.026721 | 0.127196 | 4.760166 | 0.103407 | 0.103407 | 21.481666 | 31.322409 | 4.760166 | nan | 0.103407 | 0 |
| 9 | scene01/seq03 | 1 | 0.021460 | 0.124360 | 5.794912 | 0.103395 | 0.103395 | 21.639034 | 20.958893 | 5.794911 | nan | 0.103395 | 0 |

## Worst chains by ATE
| chain_id | scene_seq | num_steps | gt_path_length | pred_path_length | path_ratio | ATE | drift | mean_rot_error | mean_tdir_error | mean_tmag_ratio | mean_turn_error | max_step_error | worst_step_index |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 14 | scene01/seq03 | 67 | 13.013749 | 15.301692 | 1.175810 | 8.514647 | 12.293643 | 20.305119 | 108.815313 | 1.523941 | 18.295176 | 12.715776 | 61 |
| 0 | scene01/seq03 | 40 | 10.200697 | 9.537485 | 0.934984 | 7.544831 | 10.142222 | 21.164677 | 30.699865 | 1.641434 | 15.853794 | 10.142222 | 39 |
| 18 | scene01/seq03 | 5 | 0.131634 | 0.765756 | 5.817300 | 0.383712 | 0.543933 | 20.441778 | 65.637252 | 5.850247 | 17.267089 | 0.543933 | 4 |
| 16 | scene01/seq03 | 2 | 0.044554 | 0.308256 | 6.918752 | 0.213046 | 0.267851 | 20.408929 | 64.837226 | 6.936126 | 16.390360 | 0.267851 | 1 |
| 15 | scene01/seq03 | 1 | 0.028136 | 0.155806 | 5.537505 | 0.171752 | 0.171752 | 20.416434 | 100.855226 | 5.537505 | nan | 0.171752 | 0 |
| 1 | scene01/seq03 | 2 | 0.059278 | 0.256570 | 4.328279 | 0.168329 | 0.212860 | 21.255264 | 28.510628 | 4.333452 | 13.980722 | 0.212860 | 1 |
| 11 | scene01/seq03 | 2 | 0.044525 | 0.250193 | 5.619178 | 0.161862 | 0.203346 | 21.667990 | 21.546713 | 5.660168 | 19.037109 | 0.203346 | 1 |
| 3 | scene01/seq03 | 2 | 0.056228 | 0.250103 | 4.448030 | 0.159829 | 0.201423 | 21.319017 | 24.109372 | 4.483121 | 15.082032 | 0.201423 | 1 |
| 17 | scene01/seq03 | 1 | 0.026840 | 0.153650 | 5.724691 | 0.134571 | 0.134571 | 20.423945 | 61.412552 | 5.724692 | nan | 0.134571 | 0 |
| 6 | scene01/seq03 | 1 | 0.021284 | 0.127553 | 5.992971 | 0.107488 | 0.107488 | 21.378853 | 27.959943 | 5.992975 | nan | 0.107488 | 0 |

## Worst chains by drift
| chain_id | scene_seq | num_steps | gt_path_length | pred_path_length | path_ratio | ATE | drift | mean_rot_error | mean_tdir_error | mean_tmag_ratio | mean_turn_error | max_step_error | worst_step_index |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 14 | scene01/seq03 | 67 | 13.013749 | 15.301692 | 1.175810 | 8.514647 | 12.293643 | 20.305119 | 108.815313 | 1.523941 | 18.295176 | 12.715776 | 61 |
| 0 | scene01/seq03 | 40 | 10.200697 | 9.537485 | 0.934984 | 7.544831 | 10.142222 | 21.164677 | 30.699865 | 1.641434 | 15.853794 | 10.142222 | 39 |
| 18 | scene01/seq03 | 5 | 0.131634 | 0.765756 | 5.817300 | 0.383712 | 0.543933 | 20.441778 | 65.637252 | 5.850247 | 17.267089 | 0.543933 | 4 |
| 16 | scene01/seq03 | 2 | 0.044554 | 0.308256 | 6.918752 | 0.213046 | 0.267851 | 20.408929 | 64.837226 | 6.936126 | 16.390360 | 0.267851 | 1 |
| 1 | scene01/seq03 | 2 | 0.059278 | 0.256570 | 4.328279 | 0.168329 | 0.212860 | 21.255264 | 28.510628 | 4.333452 | 13.980722 | 0.212860 | 1 |
| 11 | scene01/seq03 | 2 | 0.044525 | 0.250193 | 5.619178 | 0.161862 | 0.203346 | 21.667990 | 21.546713 | 5.660168 | 19.037109 | 0.203346 | 1 |
| 3 | scene01/seq03 | 2 | 0.056228 | 0.250103 | 4.448030 | 0.159829 | 0.201423 | 21.319017 | 24.109372 | 4.483121 | 15.082032 | 0.201423 | 1 |
| 15 | scene01/seq03 | 1 | 0.028136 | 0.155806 | 5.537505 | 0.171752 | 0.171752 | 20.416434 | 100.855226 | 5.537505 | nan | 0.171752 | 0 |
| 17 | scene01/seq03 | 1 | 0.026840 | 0.153650 | 5.724691 | 0.134571 | 0.134571 | 20.423945 | 61.412552 | 5.724692 | nan | 0.134571 | 0 |
| 6 | scene01/seq03 | 1 | 0.021284 | 0.127553 | 5.992971 | 0.107488 | 0.107488 | 21.378853 | 27.959943 | 5.992975 | nan | 0.107488 | 0 |

## Top 10 highest tdir error steps
- chain 14 `scene01/seq03` step 0 (i=361 j=362): tdir_err=141.610521, rot_err=20.288773, pos_err=0.167762
- chain 14 `scene01/seq03` step 1 (i=362 j=363): tdir_err=133.640047, rot_err=20.227973, pos_err=0.370388
- chain 14 `scene01/seq03` step 2 (i=363 j=364): tdir_err=123.313817, rot_err=20.340484, pos_err=0.562332
- chain 14 `scene01/seq03` step 3 (i=364 j=365): tdir_err=113.631374, rot_err=20.212946, pos_err=0.854375
- chain 14 `scene01/seq03` step 10 (i=371 j=372): tdir_err=112.881936, rot_err=20.352483, pos_err=2.362682
- chain 14 `scene01/seq03` step 6 (i=367 j=368): tdir_err=112.788958, rot_err=20.311738, pos_err=1.669856
- chain 14 `scene01/seq03` step 36 (i=397 j=398): tdir_err=112.335624, rot_err=20.290246, pos_err=8.511685
- chain 14 `scene01/seq03` step 26 (i=387 j=388): tdir_err=112.312826, rot_err=20.227780, pos_err=7.319690
- chain 14 `scene01/seq03` step 7 (i=368 j=369): tdir_err=111.726498, rot_err=20.359797, pos_err=1.953148
- chain 14 `scene01/seq03` step 25 (i=386 j=387): tdir_err=111.314017, rot_err=20.281618, pos_err=7.155475

## Top 10 highest rot error steps
- chain 11 `scene01/seq03` step 1 (i=66 j=67): rot_err=21.717051, tdir_err=21.030397, pos_err=0.203346
- chain 9 `scene01/seq03` step 0 (i=61 j=62): rot_err=21.639034, tdir_err=20.958893, pos_err=0.103395
- chain 11 `scene01/seq03` step 0 (i=65 j=66): rot_err=21.618929, tdir_err=22.063030, pos_err=0.105115
- chain 8 `scene01/seq03` step 0 (i=59 j=60): rot_err=21.585068, tdir_err=18.217242, pos_err=0.104629
- chain 10 `scene01/seq03` step 0 (i=63 j=64): rot_err=21.578879, tdir_err=26.319392, pos_err=0.103920
- chain 12 `scene01/seq03` step 0 (i=69 j=70): rot_err=21.515914, tdir_err=22.639928, pos_err=0.105048
- chain 7 `scene01/seq03` step 0 (i=56 j=57): rot_err=21.508023, tdir_err=26.259149, pos_err=0.103659
- chain 5 `scene01/seq03` step 0 (i=52 j=53): rot_err=21.501832, tdir_err=29.169221, pos_err=0.106263
- chain 4 `scene01/seq03` step 0 (i=50 j=51): rot_err=21.481666, tdir_err=31.322409, pos_err=0.103407
- chain 13 `scene01/seq03` step 0 (i=71 j=72): rot_err=21.470225, tdir_err=32.309924, pos_err=0.106159

## Correlation analysis
- corr(ATE, tdir error) = `0.425029`
- corr(ATE, rot error) = `-0.350974`
- corr(ATE, path_ratio_error) = `-0.902087`
- corr(ATE, turn error) = `0.231327`

## Final classification
- classification: `OFFICIAL/S2C-AGGREGATION-MISMATCH`

## Next step
- 修正 S2c 诊断工具，不改模型，然后重跑 S2c。
