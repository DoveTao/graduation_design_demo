# S2g Train-CV Dt-Aware Fine Rot Policy Selection On S2b

## Candidate policy definitions
- `G0_global_0p45` (global): `{'[0.1,0.3)': 0.45, '[0.3,0.5)': 0.45, '[0.5,1)': 0.45}`
- `G1_global_0p475` (global): `{'[0.1,0.3)': 0.475, '[0.3,0.5)': 0.475, '[0.5,1)': 0.475}`
- `G2_global_0p50` (global): `{'[0.1,0.3)': 0.5, '[0.3,0.5)': 0.5, '[0.5,1)': 0.5}`
- `D0_conservative_small_dt` (dt-aware): `{'[0.1,0.3)': 0.4, '[0.3,0.5)': 0.45, '[0.5,1)': 0.5}`
- `D1_aggressive_large_dt` (dt-aware): `{'[0.1,0.3)': 0.45, '[0.3,0.5)': 0.5, '[0.5,1)': 0.55}`
- `D2_mild_large_dt` (dt-aware): `{'[0.1,0.3)': 0.45, '[0.3,0.5)': 0.475, '[0.5,1)': 0.5}`
- `D3_conservative_large_dt` (dt-aware): `{'[0.1,0.3)': 0.45, '[0.3,0.5)': 0.45, '[0.5,1)': 0.4}`
- `D4_smooth_ramp` (dt-aware): `{'[0.1,0.3)': 0.425, '[0.3,0.5)': 0.475, '[0.5,1)': 0.525}`

## Train-CV setup
- train_only_leave_one_train_seq_out_cv groups: `[('scene01', 'seq01'), ('scene01', 'seq02')]`
- no model parameter training
- no test labels used for selection

## Train-CV fold table
| candidate | fold | drift | ATE | path_ratio | RPE_rot | RPE_trans_dir | RPE_trans_mag | rot | tdir_abs | tdir_local_A_abs | selected_k | num_pairs | num_chains | missing | unexpected |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| G0_global_0p45 | fold_0_scene01_seq01 | 1.118839 | 6.180546 | 1.006421 | 20.874382 | 44.056121 | 0.071867 | 23.279036 | 41.373979 | 31.185702 | 1 | 155 | 21 | 2 | 0 |
| G0_global_0p45 | fold_1_scene01_seq02 | 1.608874 | 7.127564 | 1.005017 | 20.740897 | 41.311662 | 0.073144 | 19.985931 | 17.457737 | 15.401528 | 1 | 138 | 16 | 2 | 0 |
| G1_global_0p475 | fold_0_scene01_seq01 | 1.119126 | 6.253460 | 1.006419 | 21.442172 | 44.373972 | 0.071867 | 23.747451 | 41.505448 | 31.185702 | 1 | 155 | 21 | 2 | 0 |
| G1_global_0p475 | fold_1_scene01_seq02 | 1.609547 | 7.117768 | 1.005016 | 21.419512 | 41.607993 | 0.073144 | 20.513569 | 17.662146 | 15.401528 | 1 | 138 | 16 | 2 | 0 |
| G2_global_0p50 | fold_0_scene01_seq01 | 1.117662 | 6.341522 | 1.006420 | 22.010493 | 44.694279 | 0.071867 | 24.216581 | 41.636596 | 31.185702 | 1 | 155 | 21 | 2 | 0 |
| G2_global_0p50 | fold_1_scene01_seq02 | 1.580406 | 7.083404 | 1.005015 | 22.098778 | 41.926176 | 0.073144 | 21.042710 | 17.880943 | 15.401528 | 1 | 138 | 16 | 2 | 0 |
| D0_conservative_small_dt | fold_0_scene01_seq01 | 1.102564 | 6.142037 | 1.006420 | 20.374771 | 43.733629 | 0.071867 | 23.225703 | 41.313839 | 31.185702 | 1 | 155 | 21 | 2 | 0 |
| D0_conservative_small_dt | fold_1_scene01_seq02 | 1.576545 | 7.138034 | 1.005016 | 20.020130 | 40.845670 | 0.073144 | 20.160410 | 17.537848 | 15.401528 | 1 | 138 | 16 | 2 | 0 |
| D1_aggressive_large_dt | fold_0_scene01_seq01 | 1.115875 | 6.215801 | 1.006421 | 21.508571 | 44.361340 | 0.071867 | 24.161709 | 41.574933 | 31.185702 | 1 | 155 | 21 | 2 | 0 |
| D1_aggressive_large_dt | fold_1_scene01_seq02 | 1.610688 | 7.106216 | 1.005015 | 21.374960 | 41.417609 | 0.073144 | 21.215724 | 17.969378 | 15.401528 | 1 | 138 | 16 | 2 | 0 |
| D2_mild_large_dt | fold_0_scene01_seq01 | 1.115660 | 6.196192 | 1.006420 | 21.191382 | 44.208709 | 0.071867 | 23.720129 | 41.475181 | 31.185702 | 1 | 155 | 21 | 2 | 0 |
| D2_mild_large_dt | fold_1_scene01_seq02 | 1.611092 | 7.113712 | 1.005017 | 21.057819 | 41.355082 | 0.073144 | 20.599891 | 17.698966 | 15.401528 | 1 | 138 | 16 | 2 | 0 |
| D3_conservative_large_dt | fold_0_scene01_seq01 | 1.124978 | 6.187223 | 1.006421 | 20.747059 | 44.013335 | 0.071867 | 22.972537 | 41.319830 | 31.185702 | 1 | 155 | 21 | 2 | 0 |
| D3_conservative_large_dt | fold_1_scene01_seq02 | 1.609156 | 7.138432 | 1.005016 | 20.635817 | 41.351705 | 0.073144 | 19.479603 | 17.278023 | 15.401528 | 1 | 138 | 16 | 2 | 0 |
| D4_smooth_ramp | fold_0_scene01_seq01 | 1.110025 | 6.159349 | 1.006420 | 20.941181 | 44.045748 | 0.071867 | 23.693319 | 41.444505 | 31.185702 | 1 | 155 | 21 | 2 | 0 |
| D4_smooth_ramp | fold_1_scene01_seq02 | 1.599137 | 7.113114 | 1.005016 | 20.696894 | 41.119306 | 0.073144 | 20.687376 | 17.746307 | 15.401528 | 1 | 138 | 16 | 2 | 0 |

## Mean CV ranking
| candidate | type | mapping | mean_drift | mean_ATE | mean_path_ratio | simpler_rank |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| G0_global_0p45 | global | `{'[0.1,0.3)': 0.45, '[0.3,0.5)': 0.45, '[0.5,1)': 0.45}` | 1.363857 | 6.654055 | 1.005719 | 0 |
| G1_global_0p475 | global | `{'[0.1,0.3)': 0.475, '[0.3,0.5)': 0.475, '[0.5,1)': 0.475}` | 1.364337 | 6.685614 | 1.005717 | 0 |
| G2_global_0p50 | global | `{'[0.1,0.3)': 0.5, '[0.3,0.5)': 0.5, '[0.5,1)': 0.5}` | 1.349034 | 6.712463 | 1.005717 | 0 |
| D0_conservative_small_dt | dt-aware | `{'[0.1,0.3)': 0.4, '[0.3,0.5)': 0.45, '[0.5,1)': 0.5}` | 1.339554 | 6.640035 | 1.005718 | 2 |
| D1_aggressive_large_dt | dt-aware | `{'[0.1,0.3)': 0.45, '[0.3,0.5)': 0.5, '[0.5,1)': 0.55}` | 1.363281 | 6.661008 | 1.005718 | 3 |
| D2_mild_large_dt | dt-aware | `{'[0.1,0.3)': 0.45, '[0.3,0.5)': 0.475, '[0.5,1)': 0.5}` | 1.363376 | 6.654952 | 1.005718 | 1 |
| D3_conservative_large_dt | dt-aware | `{'[0.1,0.3)': 0.45, '[0.3,0.5)': 0.45, '[0.5,1)': 0.4}` | 1.367067 | 6.662828 | 1.005718 | 2 |
| D4_smooth_ramp | dt-aware | `{'[0.1,0.3)': 0.425, '[0.3,0.5)': 0.475, '[0.5,1)': 0.525}` | 1.354581 | 6.636231 | 1.005718 | 1 |

## Selected policy
- selected policy name: `D0_conservative_small_dt`
- policy type: `dt-aware`
- selected mapping: `{'[0.1,0.3)': 0.4, '[0.3,0.5)': 0.45, '[0.5,1)': 0.5}`
- selected by train-CV only: `True`
- matches S2f diagnostic global 0.475: `False`
- matches S2f diagnostic dt-aware aggressive_large_dt: `False`

## Final test result
- drift = 1.366864
- ATE = 7.463772
- path_ratio = 0.934983
- RPE_rot = 19.913663
- RPE_trans_dir = 73.003874
- RPE_trans_mag = 0.073926
- rot = 21.409208
- tdir_abs = 29.847270
- tdir_local_A_abs = 23.152590
- tmag P10/P50/P90 = 0.126941/0.183128/0.402209
- selected_k = 1
- available_k = [1, 2, 3, 5, 10, 20]
- num_pairs / num_chains = 132 / 19
- missing / unexpected = 2 / 0

## Comparison
| method | drift | ATE | path_ratio | status |
| --- | ---: | ---: | ---: | --- |
| S1d5 | 1.396358 | 7.632463 | 0.934982 | previous clean exported baseline |
| S2b | 1.327402 | 7.352371 | 0.934984 | current clean fine-rot candidate |
| S2f diagnostic global 0.475 | 1.331278 | 7.309820 | 0.934984 | test-swept diagnostic |
| S2f diagnostic dt-aware aggressive_large_dt | 1.332847 | 7.345742 | 0.934983 | test-swept diagnostic |
| S2g selected | 1.366864 | 7.463772 | 0.934983 | train-CV selected |

## Verdict
- `FAIL`
- selected policy improves over S2b = `False`
