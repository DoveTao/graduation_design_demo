# S11 Tmag Scale Consistency Report

## Executive summary

- final classification: `NO-STABLE-TMAG-CONSISTENCY-GAIN`
- S11 replaces S5: `False`
- best diagnostic candidate: `D_high_regime_weighted_consistency`
- odometry eval run: `True`

## Motivation from S4/S5/S8/S9/S10

- S4/S5 indicate that translation magnitude regime dominates current error structure.
- S8/S9/S10 suggest that post-processing, routing, and smoothing have limited clean headroom.
- S11 moves the optimization target back into training-time magnitude scale learning.

## Why training-time tmag consistency instead of post-processing

- S11 keeps residual pose heads disabled and does not add any inference-time router.
- gt_tmag is used only as a training label or train-only weighting signal.

## Candidate loss definitions

- `A`: train only tmag-related parameters with log-tmag Huber.
- `C`: same training scope plus speed / ratio / chain-sum consistency losses.
- `D`: same as `C`, plus train-only high-regime weighting on pred/gt/dt/k buckets.

## Training protocol

- mode: `cv`
- training budget: `lightweight two-fold CV, candidates=A/C/D, max_steps=100`
- folds: `heldout_scene01_seq01`, `heldout_scene01_seq02`
- candidate set: `A`, `C`, `D`
- test set was not used for candidate selection.

## Two-fold lightweight CV setting

- train split only, leave-one-seq-out style over the two available train sequences.
- fixed-pair eval and odometry eval were kept on for fold validation.
- if this stage had shown no signal at 100 updates, it would not be expanded to 200.

## Candidate table

| candidate | family | cv_tmag_log | cv_speed_log | cv_ratio_log | cv_chain_sum_log | cv_ATE | cv_drift | cv_path_ratio | eligible |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| D_high_regime_weighted_consistency | high_regime_weighted_consistency | 2.992246 | 2.992246 | 0.459631 | 2.940869 | 4.528859 | 2.661156 | nan | False |
| C_loss_only_consistency | loss_only_consistency | 3.006669 | 3.006669 | 0.459617 | 2.954968 | 4.533715 | 2.666410 | nan | False |
| A_tmag_head_only_baseline | tmag_head_only_baseline | 3.077567 | 3.077567 | 0.459594 | 3.024949 | 4.562651 | 2.692555 | nan | False |

## Fold metrics

### D_high_regime_weighted_consistency

- `heldout_scene01_seq01`: train_loss=`1.047404`, val_tmag_log=`3.535752`, val_speed_log=`3.535752`, val_ratio_log=`0.436667`, val_chain_sum_log=`3.493429`, ATE=`5.237118`, drift=`2.542570`, path_ratio=`nan`
- `heldout_scene01_seq02`: train_loss=`2.670192`, val_tmag_log=`2.448739`, val_speed_log=`2.448739`, val_ratio_log=`0.482594`, val_chain_sum_log=`2.388310`, ATE=`3.820601`, drift=`2.779742`, path_ratio=`nan`

### C_loss_only_consistency

- `heldout_scene01_seq01`: train_loss=`0.963203`, val_tmag_log=`3.550279`, val_speed_log=`3.550279`, val_ratio_log=`0.436649`, val_chain_sum_log=`3.507931`, ATE=`5.243779`, drift=`2.551665`, path_ratio=`nan`
- `heldout_scene01_seq02`: train_loss=`2.668259`, val_tmag_log=`2.463058`, val_speed_log=`2.463058`, val_ratio_log=`0.482585`, val_chain_sum_log=`2.402004`, ATE=`3.823651`, drift=`2.781155`, path_ratio=`nan`

### A_tmag_head_only_baseline

- `heldout_scene01_seq01`: train_loss=`0.861885`, val_tmag_log=`3.606479`, val_speed_log=`3.606479`, val_ratio_log=`0.436620`, val_chain_sum_log=`3.563875`, ATE=`5.269747`, drift=`2.587270`, path_ratio=`nan`
- `heldout_scene01_seq02`: train_loss=`2.662703`, val_tmag_log=`2.548655`, val_speed_log=`2.548655`, val_ratio_log=`0.482567`, val_chain_sum_log=`2.486022`, ATE=`3.855554`, drift=`2.797839`, path_ratio=`nan`

## Proxy metrics

- `val_tmag_log_error`: mean absolute log-magnitude error on held-out k=1 triplets.
- `val_speed_log_error`: held-out speed proxy error using `pred_tmag / dt`.
- `val_ratio_log_error`: adjacent-pair log-ratio consistency error.
- `val_chain_sum_log_error`: short-chain path-length proxy error.
- `val_chain_sum_ratio_absdev`: absolute deviation of `(pred_ab + pred_bc)/(gt_ab + gt_bc)` from 1.

## Whether odometry eval was run

- `True`

Note: in this lightweight S11 path, fold odometry summaries exposed ATE/drift but did not emit a stable path-ratio field into `last_eval`. The report therefore treats chain-sum proxy improvement as the main path-length safety signal, and does not promote any candidate to full eligibility without explicit path-ratio evidence.

## Hard-gate audit

- `D_high_regime_weighted_consistency`: path_ok=`False`, odom_ok=`True`, rot_tdir_safe=`True`, init_ok=`True`, eligible=`False`
- `C_loss_only_consistency`: path_ok=`False`, odom_ok=`True`, rot_tdir_safe=`True`, init_ok=`True`, eligible=`False`
- `A_tmag_head_only_baseline`: path_ok=`False`, odom_ok=`True`, rot_tdir_safe=`True`, init_ok=`True`, eligible=`False`

## Whether candidate is eligible for full S11 CV

- `D_high_regime_weighted_consistency`: `False`
- `C_loss_only_consistency`: `False`
- `A_tmag_head_only_baseline`: `False`

## Compare against S5

- locked S5 drift / ATE / path_ratio = `1.327343` / `7.352288` / `0.932379`
- S5 remains final clean candidate: `True`

## Effect on path_ratio / ATE / drift

- `D_high_regime_weighted_consistency` vs A: delta_ATE=`-0.033791`, delta_drift=`-0.031399`, delta_path_ratio=`nan`
- `C_loss_only_consistency` vs A: delta_ATE=`-0.028936`, delta_drift=`-0.026145`, delta_path_ratio=`nan`

## Effect on high tmag regime

- D applies train-only high-regime weighting; this report evaluates whether that improves held-out tmag and chain-sum proxies without unsafe path-ratio drift.

## Leakage audit

- passed: `True`
- test_used_for_selection: `False`
- gt_tmag_as_inference_feature: `False`
- gt_pose_as_inference_feature: `False`
- post_processing_router_used: `False`
- train_cv_selection_only: `True`

## Final classification

- `NO-STABLE-TMAG-CONSISTENCY-GAIN`
- rationale: `proxy improved but path/odometry safety not stable`
