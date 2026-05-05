# S3a1_train_cv_small_run Report

## 1. S2b baseline

- drift = `1.327402`
- ATE = `7.352371`
- path_ratio = `0.934984`

## 2. Smoke result recap

- source commit: `15dc1a0754c7698754cccc4ba2b0ba7b5ac62e19`
- drift = `1.3284205512539493`
- ATE = `7.356587799281789`
- path_ratio = `0.9350349269488233`
- tdir_before_after_max_diff = `0.0`
- tmag_before_after_max_diff = `0.0`
- optimizer only rot residual/gate = `True`

## 3. Train-CV candidate table

| candidate | heldout | lr | updates | rot_scale | gate_max | reg_w | drift | ATE | path_ratio | RPE_rot | RPE_trans_dir | RPE_trans_mag | rot | tdir_abs | tdir_local_A_abs | selected_k | num_pairs | num_chains | missing | unexpected | bad_forward | skip_updates | tdir_diff | tmag_diff | valid |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A_lr5e-05_upd100_rot0.01_gate0.02 | seq01 | 0.000050 | 100 | 0.010000 | 0.020000 | 0.010000 | 1.118591 | 6.178965 | 1.006452 | 20.872033 | 44.048090 | 0.071870 | 22.281618 | 37.046687 | 29.547427 | 1 | 155 | 21 | 14 | 0 | 0 | 0 | 0.000000 | 0.000000 | True |
| A_lr5e-05_upd100_rot0.01_gate0.02 | seq02 | 0.000050 | 100 | 0.010000 | 0.020000 | 0.010000 | 1.608224 | 7.126376 | 1.005067 | 20.738010 | 41.313790 | 0.073147 | 20.114744 | 19.434214 | 15.637897 | 1 | 138 | 16 | 14 | 0 | 0 | 0 | 0.000000 | 0.000000 | False |
| B_lr5e-05_upd200_rot0.01_gate0.02 | seq01 | 0.000050 | 200 | 0.010000 | 0.020000 | 0.010000 | 1.118591 | 6.178963 | 1.006452 | 20.871957 | 44.048064 | 0.071870 | 22.281535 | 37.046652 | 29.547427 | 1 | 155 | 21 | 14 | 0 | 0 | 0 | 0.000000 | 0.000000 | True |
| B_lr5e-05_upd200_rot0.01_gate0.02 | seq02 | 0.000050 | 200 | 0.010000 | 0.020000 | 0.010000 | 1.608225 | 7.126381 | 1.005067 | 20.737935 | 41.313779 | 0.073147 | 20.114680 | 19.434201 | 15.637897 | 1 | 138 | 16 | 14 | 0 | 0 | 0 | 0.000000 | 0.000000 | False |
| C_lr0.0001_upd100_rot0.02_gate0.05 | seq01 | 0.000100 | 100 | 0.020000 | 0.050000 | 0.010000 | 1.118615 | 6.178861 | 1.006452 | 20.867924 | 44.046629 | 0.071870 | 22.277174 | 37.044627 | 29.547427 | 1 | 155 | 21 | 14 | 0 | 0 | 0 | 0.000000 | 0.000000 | True |
| C_lr0.0001_upd100_rot0.02_gate0.05 | seq02 | 0.000100 | 100 | 0.020000 | 0.050000 | 0.010000 | 1.608232 | 7.126617 | 1.005066 | 20.734217 | 41.313215 | 0.073147 | 20.111438 | 19.433447 | 15.637897 | 1 | 138 | 16 | 14 | 0 | 0 | 0 | 0.000000 | 0.000000 | False |
| D_lr0.0001_upd200_rot0.02_gate0.05 | seq01 | 0.000100 | 200 | 0.020000 | 0.050000 | 0.010000 | 1.119441 | 6.180171 | 1.006452 | 20.829395 | 44.044946 | 0.071870 | 22.232840 | 37.034404 | 29.547427 | 1 | 155 | 21 | 14 | 0 | 0 | 0 | 0.000000 | 0.000000 | True |
| D_lr0.0001_upd200_rot0.02_gate0.05 | seq02 | 0.000100 | 200 | 0.020000 | 0.050000 | 0.010000 | 1.608921 | 7.131178 | 1.005066 | 20.691728 | 41.304665 | 0.073147 | 20.075450 | 19.428210 | 15.637897 | 1 | 138 | 16 | 14 | 0 | 0 | 0 | 0.000000 | 0.000000 | False |

Mean ranking:

| candidate | mean_ATE | mean_drift | mean_path_ratio | valid | simplicity |
| --- | ---: | ---: | ---: | --- | --- |
| A_lr5e-05_upd100_rot0.01_gate0.02 | 6.652671 | 1.363408 | 1.005759 | False | `(0.01, 0.02, 5e-05, 100)` |
| B_lr5e-05_upd200_rot0.01_gate0.02 | 6.652672 | 1.363408 | 1.005760 | False | `(0.01, 0.02, 5e-05, 200)` |
| C_lr0.0001_upd100_rot0.02_gate0.05 | 6.652739 | 1.363423 | 1.005759 | False | `(0.02, 0.05, 0.0001, 100)` |
| D_lr0.0001_upd200_rot0.02_gate0.05 | 6.655675 | 1.364181 | 1.005759 | False | `(0.02, 0.05, 0.0001, 200)` |

## 4. Selected config

- no train-CV candidate satisfied the hard gates

## 5. Final test result

- final test eval not run because no valid train-CV selection was available

## 6. Optimizer/freeze audit

- all 8 CV runs: trainable parameter count = `119076`
- all 8 CV runs: optimizer parameter count = `119076`
- all 8 CV runs: forbidden trainable params count = `0`
- all 8 CV runs: optimizer only rot residual/gate = `True`
- all 8 CV runs: trainable parameter names stayed within `coupled_pose_head.backbone.*`, `coupled_pose_head.rot_head.*`, `coupled_pose_head.gate_head.*`
- no CV run put `tdir_head`, encoder, coarse, fine, direct head, or mag head into optimizer

## 7. Tdir/Tmag invariance audit

- all 8 CV runs: `tdir before/after diff = 0.0`
- all 8 CV runs: `tmag before/after diff = 0.0`
- all 8 CV runs: `bad_forward = 0`
- all 8 CV runs: `skip_updates = 0`
- all 8 CV runs: `unexpected = 0`
- hard-gate blocker was not invariance or optimizer leakage; it was held-out `seq02` drift staying above `1.45` for every candidate

## 8. Verdict

- verdict = `FAIL`
- final ATE < S2b = `False`
- final drift <= 1.35 = `False`
- final path_ratio >= 0.90 = `False`

## 9. Whether to enter larger S3a1 run

- recommendation = `False`

## 10. Whether to replace S2b

- replace S2b = `False`
