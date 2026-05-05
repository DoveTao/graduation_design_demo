# S1d5 freeze clean dt anchor policy export report

## Summary verdict

SUCCESS

## Exported policy

- policy json path:
  [S1d5_clean_dt_anchor_policy.json](/home/dovetao/graduation_design_demo/checkpoints/S1d5_clean_dt_anchor_policy.json)
- eval script path:
  [eval_s1d5_clean_policy.sh](/home/dovetao/graduation_design_demo/scripts/eval_s1d5_clean_policy.sh)
- eval tool path:
  [eval_clean_policy.py](/home/dovetao/graduation_design_demo/tools/eval_clean_policy.py)

## Exact exported values

- base checkpoint:
  `checkpoints/T57b_no_dt_multiscale_tmag_head_400/final.pt`
- cfg restore requirement:
  `explicit-cfg / unexpected=0`
- bucket edges:
  `[0.1, 0.3, 0.5, 1.0]`
- base S1d2 run:
  `run_13`
- base bucket factors:
  - `[0.1,0.3) = 1.2860867109362162`
  - `[0.3,0.5) = 2.48467815725585`
  - `[0.5,1) = 3.714343612316795`
- base bucket log-biases:
  - `[0.1,0.3) = 0.2516040503978729`
  - `[0.3,0.5) = 0.9101431369781494`
  - `[0.5,1) = 1.312201976776123`
- alpha:
  `1.05`
- effective bucket factors:
  - `[0.1,0.3) = 1.302368139324791`
  - `[0.3,0.5) = 2.6003610319428594`
  - `[0.5,1) = 3.9662143503139955`
- fine policy:
  - `fine_rot_fuse_strength = 0.40`
  - `fine_tdir_fuse_strength = 0.0`
  - `fine_tmag_fuse_strength = 0.0`
  - `use_geometry_refine = False`
- clean selection source:
  `train_only_leave_one_train_seq_out_cv`
- test labels used for selection:
  `false`

## Reproduction run

- reproduction output dir:
  [S1d5_clean_policy_eval](/home/dovetao/graduation_design_demo/checkpoints/S1d5_clean_policy_eval)
- summary json:
  [s1d5_policy_eval_summary.json](/home/dovetao/graduation_design_demo/checkpoints/S1d5_clean_policy_eval/s1d5_policy_eval_summary.json)

## Reproduction result

- load missing/unexpected:
  `2 / 0`
- odom_selected_k:
  `1`
- odom_available_k:
  `[1, 2, 3, 5, 10, 20]`
- num_pairs / num_chains:
  `132 / 19`
- tmag P10/P50/P90:
  `0.126966 / 0.182995 / 0.402133`
- metric path_ratio:
  `0.934982`
- direction_only path_ratio:
  `0.999997`
- drift:
  `1.396358`
- ATE:
  `7.632463`
- RPE_trans_mag:
  `0.073926`

## Comparison With S1d4

- S1d4 selected policy:
  - `alpha = 1.05`
  - `fine_rot = 0.40`
  - `fine_tdir = 0.0`
  - `fine_tmag = 0.0`
- S1d4 reported final test:
  - `drift = 1.396`
  - `ATE = 7.632`
  - `path_ratio = 0.935`
  - `RPE_trans_mag = 0.074`
- S1d5 reproduced:
  - `drift = 1.396358`
  - `ATE = 7.632463`
  - `path_ratio = 0.934982`
  - `RPE_trans_mag = 0.073926`

## Exactness

- verdict on exactness:
  numerically close, effectively identical for primary metrics
- notes:
  - `drift`, `ATE`, `path_ratio`, `RPE_trans_mag`, `selected_k`, `num_pairs`, `num_chains` match S1d4 to reporting precision
  - `tmag` quantiles are numerically very close
  - this report does not claim bitwise identity across every floating-point statistic

## Decision

- S1d5 successfully freezes S1d4 into a reusable policy artifact
- S1d5 is now the current clean exported mainline policy
- F1d remains paused
