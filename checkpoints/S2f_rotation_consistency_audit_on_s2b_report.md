# S2f Rotation Consistency Audit On S2b

## S2b baseline
- policy: `checkpoints/S2b_clean_fine_rot_policy.json`
- drift = 1.327402
- ATE = 7.352371
- path_ratio = 0.934984
- RPE_rot = 20.715353
- RPE_trans_dir = 72.349483
- RPE_trans_mag = 0.073926
- rot = 21.342697
- tdir_abs = 25.143916
- tdir_local_A_abs = 25.885281
- selected_k / num_pairs / num_chains = 1 / 132 / 19
- missing / unexpected = 2 / 0

## Fine-grained global fine_rot sweep
| fine_rot | drift | ATE | path_ratio | RPE_rot | RPE_trans_dir | RPE_trans_mag | rot | tdir_abs | tdir_local_A_abs | direction_only_path_ratio | tmag_p10 | tmag_p50 | tmag_p90 | selected_k | num_pairs | num_chains | missing | unexpected |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.350 | 1.425013 | 7.933934 | 0.934986 | 15.891158 | 73.896452 | 0.073926 | 16.567489 | 24.235059 | 25.885281 | 0.999996 | 0.126998 | 0.183193 | 0.402218 | 1 | 132 | 19 | 2 | 0 |
| 0.375 | 1.407466 | 7.801761 | 0.934985 | 17.090114 | 73.493425 | 0.073926 | 17.750568 | 24.396492 | 25.885281 | 0.999995 | 0.126977 | 0.183143 | 0.402364 | 1 | 132 | 19 | 2 | 0 |
| 0.400 | 1.396358 | 7.632463 | 0.934982 | 18.294509 | 73.101235 | 0.073926 | 18.941746 | 24.605715 | 25.885281 | 0.999995 | 0.126882 | 0.183089 | 0.402144 | 1 | 132 | 19 | 2 | 0 |
| 0.425 | 1.363581 | 7.472323 | 0.934984 | 19.503266 | 72.719905 | 0.073926 | 20.139595 | 24.857266 | 25.885281 | 0.999998 | 0.126918 | 0.183170 | 0.402193 | 1 | 132 | 19 | 2 | 0 |
| 0.450 | 1.327402 | 7.352371 | 0.934984 | 20.715353 | 72.349483 | 0.073926 | 21.342697 | 25.143916 | 25.885281 | 1.000001 | 0.126974 | 0.183012 | 0.402142 | 1 | 132 | 19 | 2 | 0 |
| 0.475 | 1.331278 | 7.309820 | 0.934984 | 21.929656 | 71.989988 | 0.073926 | 22.549710 | 25.472900 | 25.885281 | 1.000005 | 0.126896 | 0.183244 | 0.402262 | 1 | 132 | 19 | 2 | 0 |
| 0.500 | 1.368983 | 7.365145 | 0.934983 | 23.145094 | 71.641407 | 0.073926 | 23.759319 | 25.841382 | 25.885281 | 1.000002 | 0.126900 | 0.183193 | 0.401970 | 1 | 132 | 19 | 2 | 0 |
| 0.525 | 1.389909 | 7.488684 | 0.934985 | 24.360566 | 71.303701 | 0.073926 | 24.970205 | 26.247096 | 25.885281 | 0.999999 | 0.127080 | 0.183169 | 0.402286 | 1 | 132 | 19 | 2 | 0 |
| 0.550 | 1.377539 | 7.618992 | 0.934984 | 25.574969 | 70.976784 | 0.073926 | 26.181108 | 26.688570 | 25.885281 | 1.000002 | 0.126939 | 0.183051 | 0.402329 | 1 | 132 | 19 | 2 | 0 |

## DT-aware / K-aware diagnostic policy table
| policy | mapping | drift | ATE | path_ratio | RPE_rot | RPE_trans_dir | RPE_trans_mag | selected_k | num_pairs | num_chains | missing | unexpected | notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| global_s2b_baseline | `{'[0.1,0.3)': 0.45, '[0.3,0.5)': 0.45, '[0.5,1.0)': 0.45}` | 1.327402 | 7.352371 | 0.934984 | 20.715353 | 72.349483 | 0.073926 | 1 | 132 | 19 | 2 | 0 | dt-aware diagnostic |
| conservative_small_dt | `{'[0.1,0.3)': 0.4, '[0.3,0.5)': 0.45, '[0.5,1.0)': 0.5}` | 1.366864 | 7.463772 | 0.934983 | 19.913663 | 73.003874 | 0.073926 | 1 | 132 | 19 | 2 | 0 | dt-aware diagnostic |
| aggressive_large_dt | `{'[0.1,0.3)': 0.45, '[0.3,0.5)': 0.5, '[0.5,1.0)': 0.55}` | 1.332847 | 7.345742 | 0.934983 | 21.341663 | 72.262563 | 0.073926 | 1 | 132 | 19 | 2 | 0 | dt-aware diagnostic |
| conservative_large_dt | `{'[0.1,0.3)': 0.45, '[0.3,0.5)': 0.45, '[0.5,1.0)': 0.4}` | 1.329883 | 7.379242 | 0.934984 | 20.604478 | 72.296798 | 0.073926 | 1 | 132 | 19 | 2 | 0 | dt-aware diagnostic |
| k_aware_diagnostic | n/a | nan | nan | nan | nan | nan | nan | 1 | 132 | 19 | 2 | 0 | Not feasible as an odom policy under official selected_k=1 scope; all evaluated odom chains use k=1. |

## Bucket-wise error breakdown
### S2b baseline dt bucket
| bucket | count | rot_error | turn_error | ATE_contribution | drift_contribution | step_position_error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [0.1,0.3) | 50 | 31.358453 | 8.384308 | 8.036063 | 6.896483 | 6.896483 |
| [0.3,0.5) | 22 | 30.672259 | 5.143029 | 5.186575 | 4.661263 | 4.661263 |
| [0.5,1.0) | 6 | 31.733193 | 5.883998 | 2.622143 | 2.436993 | 2.436993 |
| other | 54 | 31.091048 | 4.929121 | 6.705450 | 4.266733 | 4.266733 |

### S2b baseline k bucket
| bucket | count | rot_error | turn_error | ATE_contribution | drift_contribution | step_position_error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 132 | 31.151728 | 6.550312 | 6.903001 | 5.245435 | 5.245435 |

### Best diagnostic dt bucket
| bucket | count | rot_error | turn_error | ATE_contribution | drift_contribution | step_position_error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [0.1,0.3) | 50 | 31.834874 | 9.217032 | 8.074484 | 6.942548 | 6.942548 |
| [0.3,0.5) | 22 | 35.026998 | 6.758915 | 5.129202 | 4.598807 | 4.598807 |
| [0.5,1.0) | 6 | 43.160942 | 8.273215 | 2.542029 | 2.324935 | 2.324935 |
| other | 54 | 31.724646 | 6.057294 | 6.907170 | 4.516552 | 4.516552 |

### Best diagnostic k bucket
| bucket | count | rot_error | turn_error | ATE_contribution | drift_contribution | step_position_error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 132 | 32.836622 | 7.709666 | 6.992265 | 5.349580 | 5.349580 |

## Chain-level worst cases
### S2b baseline top chains by ATE
| chain_id | scene_seq | num_steps | ATE | drift | path_ratio | mean_rot_error | mean_turn_error | mean_step_position_error | max_cumulative_error |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 14 | scene01/seq03 | 67 | 9.227472 | 13.067285 | 1.175810 | 30.746916 | 5.181719 | 8.191888 | 13.186704 |
| 0 | scene01/seq03 | 40 | 3.821358 | 4.825268 | 0.934822 | 31.551859 | 9.245122 | 3.478499 | 4.916121 |
| 18 | scene01/seq03 | 5 | 0.366315 | 0.498817 | 5.817577 | 27.282086 | 4.192376 | 0.342360 | 0.498817 |
| 16 | scene01/seq03 | 2 | 0.210457 | 0.263695 | 6.921660 | 31.983419 | 1.883133 | 0.200857 | 0.263695 |
| 15 | scene01/seq03 | 1 | 0.171612 | 0.171612 | 5.535033 | 33.446686 | nan | 0.171612 | 0.171612 |
| 1 | scene01/seq03 | 2 | 0.167604 | 0.211651 | 4.328423 | 34.799291 | 8.432554 | 0.159178 | 0.211651 |
| 11 | scene01/seq03 | 2 | 0.162817 | 0.204834 | 5.617649 | 35.077634 | 6.823991 | 0.155005 | 0.204834 |
| 3 | scene01/seq03 | 2 | 0.160374 | 0.202345 | 4.446231 | 36.813224 | 3.722864 | 0.152398 | 0.202345 |
| 17 | scene01/seq03 | 1 | 0.134669 | 0.134669 | 5.729642 | 24.600799 | nan | 0.134669 | 0.134669 |
| 6 | scene01/seq03 | 1 | 0.107347 | 0.107347 | 5.987934 | 29.390609 | nan | 0.107347 | 0.107347 |

### Best diagnostic top chains by ATE
| chain_id | scene_seq | num_steps | ATE | drift | path_ratio | mean_rot_error | mean_turn_error | mean_step_position_error | max_cumulative_error |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 14 | scene01/seq03 | 67 | 9.222022 | 13.113871 | 1.175986 | 32.006124 | 7.090764 | 8.173291 | 13.113871 |
| 0 | scene01/seq03 | 40 | 4.343311 | 6.147945 | 0.934963 | 34.766534 | 9.466915 | 3.853424 | 6.166426 |
| 18 | scene01/seq03 | 5 | 0.366933 | 0.509882 | 5.820060 | 31.152315 | 7.109018 | 0.341829 | 0.509882 |
| 16 | scene01/seq03 | 2 | 0.208310 | 0.260271 | 6.918994 | 35.722323 | 2.483789 | 0.199137 | 0.260271 |
| 15 | scene01/seq03 | 1 | 0.171624 | 0.171624 | 5.536224 | 31.014615 | nan | 0.171624 | 0.171624 |
| 1 | scene01/seq03 | 2 | 0.168541 | 0.213196 | 4.325427 | 31.492434 | 0.524721 | 0.159888 | 0.213196 |
| 11 | scene01/seq03 | 2 | 0.162880 | 0.204919 | 5.620928 | 30.343682 | 1.399578 | 0.155062 | 0.204919 |
| 3 | scene01/seq03 | 2 | 0.160589 | 0.202638 | 4.445636 | 29.813608 | 1.148032 | 0.152591 | 0.202638 |
| 17 | scene01/seq03 | 1 | 0.134525 | 0.134525 | 5.722445 | 33.926619 | nan | 0.134525 | 0.134525 |
| 6 | scene01/seq03 | 1 | 0.107426 | 0.107426 | 5.991431 | 35.508350 | nan | 0.107426 | 0.107426 |

## Step-level worst cases
### Top 10 rot error steps
- chain 0 step 26 (i=26, j=27): rot_err=42.237122, turn_err=6.958579, tdir_err=10.964449, pos_err=4.763745
- chain 0 step 18 (i=18, j=19): rot_err=42.171439, turn_err=11.011879, tdir_err=12.378997, pos_err=4.258478
- chain 1 step 0 (i=41, j=42): rot_err=41.706281, turn_err=nan, tdir_err=20.511506, pos_err=0.106705
- chain 0 step 6 (i=6, j=7): rot_err=40.719658, turn_err=1.563341, tdir_err=32.300543, pos_err=1.576937
- chain 3 step 1 (i=47, j=48): rot_err=40.296578, turn_err=3.722864, tdir_err=8.881073, pos_err=0.202345
- chain 14 step 31 (i=392, j=393): rot_err=39.614115, turn_err=18.115465, tdir_err=109.491461, pos_err=8.667140
- chain 0 step 22 (i=22, j=23): rot_err=39.388907, turn_err=5.577015, tdir_err=10.211374, pos_err=4.496650
- chain 11 step 1 (i=66, j=67): rot_err=38.590430, turn_err=6.823991, tdir_err=16.374072, pos_err=0.204834
- chain 2 step 0 (i=44, j=45): rot_err=38.408074, turn_err=nan, tdir_err=16.841145, pos_err=0.105115
- chain 14 step 38 (i=399, j=400): rot_err=37.824375, turn_err=2.374114, tdir_err=110.781915, pos_err=10.559667

### Top 10 turn error steps
- chain 0 step 1 (i=1, j=2): rot_err=36.346358, turn_err=144.126319, tdir_err=57.526439, pos_err=0.209561
- chain 0 step 4 (i=4, j=5): rot_err=27.897500, turn_err=20.958551, tdir_err=9.863673, pos_err=0.955145
- chain 0 step 7 (i=7, j=8): rot_err=22.973683, turn_err=20.371128, tdir_err=39.772357, pos_err=1.807791
- chain 0 step 5 (i=5, j=6): rot_err=29.139297, turn_err=20.069856, tdir_err=21.249435, pos_err=1.200975
- chain 14 step 31 (i=392, j=393): rot_err=39.614115, turn_err=18.115465, tdir_err=109.491461, pos_err=8.667140
- chain 14 step 39 (i=400, j=401): rot_err=23.880515, turn_err=15.645804, tdir_err=113.950764, pos_err=10.696791
- chain 14 step 32 (i=393, j=394): rot_err=25.813645, turn_err=15.444457, tdir_err=112.976096, pos_err=8.799265
- chain 14 step 2 (i=363, j=364): rot_err=30.086469, turn_err=15.108716, tdir_err=110.368223, pos_err=0.565238
- chain 0 step 3 (i=3, j=4): rot_err=23.862028, turn_err=14.719026, tdir_err=29.972052, pos_err=0.569276
- chain 18 step 2 (i=450, j=451): rot_err=33.327259, turn_err=13.204603, tdir_err=68.474612, pos_err=0.368710

### Top 10 cumulative position error steps
- chain 14 step 62 (i=423, j=424): rot_err=33.178064, turn_err=5.276731, tdir_err=100.055774, pos_err=13.186704
- chain 14 step 63 (i=424, j=425): rot_err=25.362447, turn_err=6.110219, tdir_err=103.787660, pos_err=13.166782
- chain 14 step 61 (i=422, j=423): rot_err=32.811310, turn_err=8.669653, tdir_err=103.429951, pos_err=13.161081
- chain 14 step 64 (i=425, j=426): rot_err=29.384560, turn_err=2.739874, tdir_err=101.957432, pos_err=13.140660
- chain 14 step 65 (i=426, j=427): rot_err=28.855946, turn_err=1.898116, tdir_err=102.131556, pos_err=13.109525
- chain 14 step 66 (i=427, j=428): rot_err=25.750679, turn_err=3.275013, tdir_err=101.899402, pos_err=13.067285
- chain 14 step 60 (i=421, j=422): rot_err=32.440437, turn_err=3.099443, tdir_err=100.843785, pos_err=13.046783
- chain 14 step 59 (i=420, j=421): rot_err=36.072809, turn_err=8.707761, tdir_err=99.895780, pos_err=12.909754
- chain 14 step 58 (i=419, j=420): rot_err=24.677813, turn_err=3.306422, tdir_err=107.313739, pos_err=12.768475
- chain 14 step 57 (i=418, j=419): rot_err=27.035502, turn_err=5.855978, tdir_err=106.854195, pos_err=12.629398

## Correlation analysis
- corr(ATE, mean_rot_error) = `-0.081684`
- corr(ATE, mean_turn_error) = `0.155922`
- corr(ATE, path_ratio_error) = `-0.820255`
- corr(drift, mean_rot_error) = `-0.081232`
- corr(drift, mean_turn_error) = `0.131056`

## Interpretation
- best global fine_rot in this sweep = `0.475` with ATE=7.309820, drift=1.331278
- best dt-aware diagnostic policy = `aggressive_large_dt` with ATE=7.345742, drift=1.332847, path_ratio=0.934983
- exists diagnostic policy better than S2b under the positive threshold = True
- k-aware diagnostic feasibility = False (`selected_k=1` official odom scope)

## Final verdict
- `DT-AWARE-ROT-POSITIVE-DIAGNOSTIC`

## Next step
- Next: `S2g_train_cv_dt_aware_fine_rot_policy_selection_on_s2b`.
