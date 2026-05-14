# S2b train-CV rot policy selection on S1d5

## Summary verdict

SUCCESS

## Setup
- branch: `optimize/s2-fine-refinement-on-s1d5`
- base checkpoint: `checkpoints/T57b_no_dt_multiscale_tmag_head_400/final.pt`
- fixed dt-anchor policy: `checkpoints/S1d5_clean_dt_anchor_policy.json`
- fixed alpha / bucket factors = S1d5
- no model parameter training
- train groups: [('scene01', 'seq01'), ('scene01', 'seq02')]

## Train-CV baseline
- S1d5 current clean mainline: fine_rot=0.40, drift=1.396358, ATE=7.632463, path_ratio=0.934982
- train rot-only baseline (0.40) mean drift=1.329529, mean ATE=6.663568, mean path_ratio=1.005719

## Train-CV grid mean
| fine_rot | mean_drift | mean_ATE | mean_path_ratio | mean_RPE_trans_mag | satisfies |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0.30 | 1.366094 | 6.802937 | 1.005721 | 0.072505 | True |
| 0.35 | 1.391055 | 6.711071 | 1.005721 | 0.072505 | True |
| 0.40 | 1.329529 | 6.663568 | 1.005719 | 0.072505 | True |
| 0.45 | 1.363857 | 6.654055 | 1.005719 | 0.072505 | True |
| 0.50 | 1.333527 | 6.709468 | 1.005720 | 0.072505 | True |
| 0.55 | 1.344112 | 6.655905 | 1.005719 | 0.072505 | True |

## Fold details
| fold | fine_rot | drift | ATE | path_ratio | RPE_rot | RPE_trans_dir | RPE_trans_mag | selected_k | num_pairs | num_chains | missing | unexpected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fold_0_scene01_seq01 | 0.30 | 1.118889 | 6.311907 | 1.006423 | 13.612720 | 40.903633 | 0.071867 | 1 | 155 | 21 | 2 | 0 |
| fold_1_scene01_seq02 | 0.30 | 1.613300 | 7.293967 | 1.005018 | 13.516156 | 40.178396 | 0.073144 | 1 | 138 | 16 | 2 | 0 |
| fold_0_scene01_seq01 | 0.35 | 1.157088 | 6.200383 | 1.006422 | 16.009710 | 41.908724 | 0.071867 | 1 | 155 | 21 | 2 | 0 |
| fold_1_scene01_seq02 | 0.35 | 1.625022 | 7.221760 | 1.005020 | 15.901061 | 40.424920 | 0.073144 | 1 | 138 | 16 | 2 | 0 |
| fold_0_scene01_seq01 | 0.40 | 1.111672 | 6.157829 | 1.006420 | 18.433100 | 42.962852 | 0.071867 | 1 | 155 | 21 | 2 | 0 |
| fold_1_scene01_seq02 | 0.40 | 1.547387 | 7.169306 | 1.005018 | 18.312114 | 40.806108 | 0.073144 | 1 | 138 | 16 | 2 | 0 |
| fold_0_scene01_seq01 | 0.45 | 1.118839 | 6.180546 | 1.006421 | 20.874382 | 44.056121 | 0.071867 | 1 | 155 | 21 | 2 | 0 |
| fold_1_scene01_seq02 | 0.45 | 1.608874 | 7.127564 | 1.005017 | 20.740897 | 41.311662 | 0.073144 | 1 | 138 | 16 | 2 | 0 |
| fold_0_scene01_seq01 | 0.50 | 1.109890 | 6.334728 | 1.006421 | 23.324767 | 45.179190 | 0.071867 | 1 | 155 | 21 | 2 | 0 |
| fold_1_scene01_seq02 | 0.50 | 1.557163 | 7.084209 | 1.005019 | 23.178722 | 41.931796 | 0.073144 | 1 | 138 | 16 | 2 | 0 |
| fold_0_scene01_seq01 | 0.55 | 1.142610 | 6.374574 | 1.006422 | 25.775264 | 46.323577 | 0.071867 | 1 | 155 | 21 | 2 | 0 |
| fold_1_scene01_seq02 | 0.55 | 1.545614 | 6.937237 | 1.005015 | 25.616710 | 42.657265 | 0.073144 | 1 | 138 | 16 | 2 | 0 |

## Selection
- selected fine_rot = 0.45
- selected equals S2a diagnostic best 0.45 = True
- selection used train-CV only = True

## Final test eval
- drift = 1.327402
- ATE = 7.352371
- path_ratio = 0.934984
- RPE_rot = 20.715353
- RPE_trans_dir = 72.349483
- RPE_trans_mag = 0.073926
- rot = 21.481086
- tdir_abs = 29.944641
- tdir_local_A_abs = 23.152590
- tmag P10/P50/P90 = 0.127027/0.183106/0.402155
- metric path_ratio = 0.934984
- direction_only path_ratio = 0.999998
- selected_k = 1
- available_k = [1, 2, 3, 5, 10, 20]
- num_pairs / num_chains = 132 / 19
- load missing/unexpected = 2/0

## Comparison
| row | fine_rot | drift | ATE | path_ratio | status |
| --- | ---: | ---: | ---: | ---: | --- |
| S1d5 current clean mainline | 0.40 | 1.396358 | 7.632463 | 0.934982 | current clean exported mainline |
| S2a diagnostic best | 0.45 | 1.327402 | 7.352371 | 0.934984 | test-swept diagnostic, not clean selection |
| S2b train-CV selected candidate | 0.45 | 1.327402 | 7.352371 | 0.934984 | train-CV selected |

## Stability audit
| variant | drift | ATE | path_ratio | selected_k | num_pairs | num_chains | unexpected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| default | 1.327402 | 7.352371 | 0.934984 | 1 | 132 | 19 | 0 |
| max_eval_batches_off | 1.327402 | 7.352371 | 0.934984 | 1 | 132 | 19 | 0 |
| explicit_selected_k | 1.327402 | 7.352371 | 0.934984 | 1 | 132 | 19 | 0 |

## Mainline decision
- explicit-cfg / unexpected=0 = True
- clean replacement for S1d5 = True
- F1d remains paused = True
