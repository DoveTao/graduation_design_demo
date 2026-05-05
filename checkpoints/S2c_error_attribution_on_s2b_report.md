# S2c Error Attribution On S2b

## S2b baseline
- base checkpoint: `checkpoints/T57b_no_dt_multiscale_tmag_head_400/final.pt`
- policy: `checkpoints/S2b_clean_fine_rot_policy.json`
- explicit-cfg / unexpected=0: True
- selected_k / num_pairs / num_chains = 1 / 132 / 19
- drift = 1.327402
- ATE = 7.352371
- official reported path_ratio = 0.934984
- all-chain weighted path_ratio = 1.184835
- all-chain mean path_ratio = 4.987249
- RPE_rot = 20.715353
- RPE_trans_dir = 72.349483
- RPE_trans_mag = 0.073926

## Oracle variant comparison
| variant | drift | ATE | weighted_path_ratio | RPE_rot | RPE_trans_dir | RPE_trans_mag | mean_step_pos_err | chain_ATE | chain_drift | chain_mean_path_ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S2b current | 1.327402 | 7.352371 | 1.184835 | 20.715353 | 72.349483 | 0.073926 | 5.879424 | 0.973769 | 1.327402 | 4.987249 |
| oracle_R | 1.516962 | 10.590301 | 1.184835 | 0.000000 | 72.349483 | 0.073926 | 7.187858 | 1.033629 | 1.516962 | 4.987250 |
| oracle_tdir | 1.375470 | 7.658319 | 1.184835 | 20.715353 | 0.000000 | 0.073926 | 6.101400 | 0.996973 | 1.375470 | 4.987249 |
| oracle_R_tdir | 0.304305 | 0.911024 | 1.184835 | 0.000000 | 0.000000 | 0.073926 | 0.720172 | 0.234806 | 0.304305 | 4.987250 |
| oracle_tmag | 1.250258 | 7.569829 | 1.000000 | 20.715353 | 72.349483 | 0.000000 | 6.004989 | 0.890559 | 1.250258 | 1.000000 |
| oracle_all | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 |

## Chain-level breakdown
| chain_id | scene_seq | num_steps | gt_path_length | pred_path_length | path_ratio | ATE | drift | mean_rot_error | mean_tdir_error | mean_tmag_ratio | mean_turn_error | max_step_error | worst_step_index |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | scene01/seq03 | 40 | 10.200697 | 9.537485 | 0.934984 | 7.544831 | 10.142221 | 21.164677 | 30.699865 | 1.641434 | 15.853794 | 10.142221 | 39 |
| 1 | scene01/seq03 | 2 | 0.059278 | 0.256570 | 4.328280 | 0.168329 | 0.212860 | 21.255264 | 28.510627 | 4.333452 | 13.980722 | 0.212860 | 1 |
| 2 | scene01/seq03 | 1 | 0.027600 | 0.127748 | 4.628634 | 0.105067 | 0.105067 | 21.385366 | 27.893172 | 4.628634 | nan | 0.105067 | 0 |
| 3 | scene01/seq03 | 2 | 0.056228 | 0.250103 | 4.448030 | 0.159829 | 0.201423 | 21.319017 | 24.109371 | 4.483121 | 15.082033 | 0.201423 | 1 |
| 4 | scene01/seq03 | 1 | 0.026721 | 0.127196 | 4.760166 | 0.103407 | 0.103407 | 21.481666 | 31.322409 | 4.760166 | nan | 0.103407 | 0 |
| 5 | scene01/seq03 | 1 | 0.023888 | 0.128590 | 5.382983 | 0.106263 | 0.106263 | 21.501832 | 29.169222 | 5.382984 | nan | 0.106263 | 0 |
| 6 | scene01/seq03 | 1 | 0.021284 | 0.127553 | 5.992971 | 0.107488 | 0.107488 | 21.378853 | 27.959943 | 5.992975 | nan | 0.107488 | 0 |
| 7 | scene01/seq03 | 1 | 0.027550 | 0.129284 | 4.692647 | 0.103659 | 0.103659 | 21.508023 | 26.259149 | 4.692649 | nan | 0.103659 | 0 |
| 8 | scene01/seq03 | 1 | 0.021827 | 0.125987 | 5.772051 | 0.104629 | 0.104629 | 21.585068 | 18.217242 | 5.772052 | nan | 0.104629 | 0 |
| 9 | scene01/seq03 | 1 | 0.021460 | 0.124360 | 5.794912 | 0.103395 | 0.103395 | 21.639034 | 20.958893 | 5.794911 | nan | 0.103395 | 0 |
| 10 | scene01/seq03 | 1 | 0.021416 | 0.124694 | 5.822346 | 0.103920 | 0.103920 | 21.578879 | 26.319392 | 5.822348 | nan | 0.103920 | 0 |
| 11 | scene01/seq03 | 2 | 0.044525 | 0.250193 | 5.619178 | 0.161862 | 0.203346 | 21.667990 | 21.546713 | 5.660168 | 19.037109 | 0.203346 | 1 |
| 12 | scene01/seq03 | 1 | 0.022285 | 0.126856 | 5.692478 | 0.105048 | 0.105048 | 21.515914 | 22.639928 | 5.692476 | nan | 0.105048 | 0 |
| 13 | scene01/seq03 | 1 | 0.022211 | 0.126912 | 5.714023 | 0.106159 | 0.106159 | 21.470225 | 32.309924 | 5.714021 | nan | 0.106159 | 0 |
| 14 | scene01/seq03 | 67 | 13.013749 | 15.301692 | 1.175810 | 8.514647 | 12.293643 | 20.305119 | 108.815313 | 1.523941 | 18.295176 | 12.715776 | 61 |
| 15 | scene01/seq03 | 1 | 0.028136 | 0.155806 | 5.537505 | 0.171752 | 0.171752 | 20.416434 | 100.855226 | 5.537505 | nan | 0.171752 | 0 |
| 16 | scene01/seq03 | 2 | 0.044554 | 0.308256 | 6.918752 | 0.213046 | 0.267851 | 20.408929 | 64.837225 | 6.936126 | 16.390359 | 0.267851 | 1 |
| 17 | scene01/seq03 | 1 | 0.026840 | 0.153650 | 5.724691 | 0.134571 | 0.134571 | 20.423945 | 61.412553 | 5.724691 | nan | 0.134571 | 0 |
| 18 | scene01/seq03 | 5 | 0.131634 | 0.765756 | 5.817299 | 0.383712 | 0.543933 | 20.441778 | 65.637251 | 5.850247 | 17.267089 | 0.543933 | 4 |

## Worst cases summary
### Top 10 highest ATE chains
- chain 14 `scene01/seq03`: ATE=8.514647, drift=12.293643, path_ratio=1.175810, mean_tdir_err=108.815313, mean_rot_err=20.305119
- chain 0 `scene01/seq03`: ATE=7.544831, drift=10.142221, path_ratio=0.934984, mean_tdir_err=30.699865, mean_rot_err=21.164677
- chain 18 `scene01/seq03`: ATE=0.383712, drift=0.543933, path_ratio=5.817299, mean_tdir_err=65.637251, mean_rot_err=20.441778
- chain 16 `scene01/seq03`: ATE=0.213046, drift=0.267851, path_ratio=6.918752, mean_tdir_err=64.837225, mean_rot_err=20.408929
- chain 15 `scene01/seq03`: ATE=0.171752, drift=0.171752, path_ratio=5.537505, mean_tdir_err=100.855226, mean_rot_err=20.416434
- chain 1 `scene01/seq03`: ATE=0.168329, drift=0.212860, path_ratio=4.328280, mean_tdir_err=28.510627, mean_rot_err=21.255264
- chain 11 `scene01/seq03`: ATE=0.161862, drift=0.203346, path_ratio=5.619178, mean_tdir_err=21.546713, mean_rot_err=21.667990
- chain 3 `scene01/seq03`: ATE=0.159829, drift=0.201423, path_ratio=4.448030, mean_tdir_err=24.109371, mean_rot_err=21.319017
- chain 17 `scene01/seq03`: ATE=0.134571, drift=0.134571, path_ratio=5.724691, mean_tdir_err=61.412553, mean_rot_err=20.423945
- chain 6 `scene01/seq03`: ATE=0.107488, drift=0.107488, path_ratio=5.992971, mean_tdir_err=27.959943, mean_rot_err=21.378853

### Top 10 highest drift chains
- chain 14 `scene01/seq03`: drift=12.293643, ATE=8.514647, path_ratio=1.175810, mean_tdir_err=108.815313, mean_rot_err=20.305119
- chain 0 `scene01/seq03`: drift=10.142221, ATE=7.544831, path_ratio=0.934984, mean_tdir_err=30.699865, mean_rot_err=21.164677
- chain 18 `scene01/seq03`: drift=0.543933, ATE=0.383712, path_ratio=5.817299, mean_tdir_err=65.637251, mean_rot_err=20.441778
- chain 16 `scene01/seq03`: drift=0.267851, ATE=0.213046, path_ratio=6.918752, mean_tdir_err=64.837225, mean_rot_err=20.408929
- chain 1 `scene01/seq03`: drift=0.212860, ATE=0.168329, path_ratio=4.328280, mean_tdir_err=28.510627, mean_rot_err=21.255264
- chain 11 `scene01/seq03`: drift=0.203346, ATE=0.161862, path_ratio=5.619178, mean_tdir_err=21.546713, mean_rot_err=21.667990
- chain 3 `scene01/seq03`: drift=0.201423, ATE=0.159829, path_ratio=4.448030, mean_tdir_err=24.109371, mean_rot_err=21.319017
- chain 15 `scene01/seq03`: drift=0.171752, ATE=0.171752, path_ratio=5.537505, mean_tdir_err=100.855226, mean_rot_err=20.416434
- chain 17 `scene01/seq03`: drift=0.134571, ATE=0.134571, path_ratio=5.724691, mean_tdir_err=61.412553, mean_rot_err=20.423945
- chain 6 `scene01/seq03`: drift=0.107488, ATE=0.107488, path_ratio=5.992971, mean_tdir_err=27.959943, mean_rot_err=21.378853

### Top 10 highest tdir error steps
- chain 14 step 0 (scene01/seq03 i=361 j=362): tdir_err=141.610521, rot_err=20.288773, pos_err=0.167762
- chain 14 step 1 (scene01/seq03 i=362 j=363): tdir_err=133.640047, rot_err=20.227973, pos_err=0.370388
- chain 14 step 2 (scene01/seq03 i=363 j=364): tdir_err=123.313818, rot_err=20.340484, pos_err=0.562332
- chain 14 step 3 (scene01/seq03 i=364 j=365): tdir_err=113.631376, rot_err=20.212946, pos_err=0.854375
- chain 14 step 10 (scene01/seq03 i=371 j=372): tdir_err=112.881938, rot_err=20.352483, pos_err=2.362682
- chain 14 step 6 (scene01/seq03 i=367 j=368): tdir_err=112.788957, rot_err=20.311738, pos_err=1.669856
- chain 14 step 36 (scene01/seq03 i=397 j=398): tdir_err=112.335626, rot_err=20.290246, pos_err=8.511685
- chain 14 step 26 (scene01/seq03 i=387 j=388): tdir_err=112.312824, rot_err=20.227780, pos_err=7.319690
- chain 14 step 7 (scene01/seq03 i=368 j=369): tdir_err=111.726499, rot_err=20.359797, pos_err=1.953148
- chain 14 step 25 (scene01/seq03 i=386 j=387): tdir_err=111.314018, rot_err=20.281618, pos_err=7.155475

### Top 10 highest rot error steps
- chain 11 step 1 (scene01/seq03 i=66 j=67): rot_err=21.717051, tdir_err=21.030397, pos_err=0.203346
- chain 9 step 0 (scene01/seq03 i=61 j=62): rot_err=21.639034, tdir_err=20.958893, pos_err=0.103395
- chain 11 step 0 (scene01/seq03 i=65 j=66): rot_err=21.618929, tdir_err=22.063030, pos_err=0.105115
- chain 8 step 0 (scene01/seq03 i=59 j=60): rot_err=21.585068, tdir_err=18.217242, pos_err=0.104629
- chain 10 step 0 (scene01/seq03 i=63 j=64): rot_err=21.578879, tdir_err=26.319392, pos_err=0.103920
- chain 12 step 0 (scene01/seq03 i=69 j=70): rot_err=21.515914, tdir_err=22.639928, pos_err=0.105048
- chain 7 step 0 (scene01/seq03 i=56 j=57): rot_err=21.508023, tdir_err=26.259149, pos_err=0.103659
- chain 5 step 0 (scene01/seq03 i=52 j=53): rot_err=21.501832, tdir_err=29.169222, pos_err=0.106263
- chain 4 step 0 (scene01/seq03 i=50 j=51): rot_err=21.481666, tdir_err=31.322409, pos_err=0.103407
- chain 13 step 0 (scene01/seq03 i=71 j=72): rot_err=21.470225, tdir_err=32.309924, pos_err=0.106159

## Correlation analysis
- corr_ATE_tdir_error = 0.425029
- corr_ATE_rot_error = -0.350974
- corr_ATE_path_ratio_error = -0.902087
- corr_ATE_turn_error = 0.231327
- corr_drift_tdir_error = 0.441268
- corr_drift_rot_error = -0.360879

## Conclusion
- classification: `CHAIN/FRAME-CONVENTION-ISSUE`
- next step: `查 frame convention / integration`
