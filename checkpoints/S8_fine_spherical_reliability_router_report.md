# S8 Fine/Spherical Reliability Router Diagnostic Report

## Status

- S8b baseline contract gate has already passed.
- This run switched to staged diagnostic mode because of runtime / budget limits.
- Final classification: `TOKEN-NO-ADDED-VALUE`.
- S5 remains final clean candidate: `True`.
- Final route test skipped: `True`.
- Odometry route search disabled: `True`.

## Baseline Gate

- Gate passed: `True`.
- Contract: `/home/dovetao/graduation_design_demo/checkpoints/S8b_reproduction_contract.json`.
- Locked S5 wrapper metrics: drift=`1.327343`, ATE=`7.352288`, path_ratio=`0.932379`.

## Cache And Caps

- Cache path: `/home/dovetao/graduation_design_demo/checkpoints/S8_fine_spherical_reliability_router_rows_cache.jsonl`.
- Reused cache splits: `fold_0_scene01_seq01_train, fold_0_scene01_seq01_val, fold_1_scene01_seq02_train, fold_1_scene01_seq02_val, train_full`.
- Newly written cache splits: `none`.
- Sample caps used: train_full=`512`, fold_train=`256`, fold_val=`256`.

## Feature Audit

| feature | shape | source | inference-visible | deployable-safe |
| --- | --- | --- | --- | --- |
| pred_tmag_s2b | [1] | model.forward aux['t_mag'] after S2b | True | True |
| dt_world | [1] | dataset meta['dt_world'] | True | True |
| k | [1] | dataset meta['k'] | True | True |
| fine token summary | [16] | weighted pooled Ff_t compressed to scalar stats | True | True |
| spherical token summary | [12] | weighted pooled Fc compressed to scalar stats | True | True |
| gt pose / gt_tmag / pair_pos_err | n/a | diagnostics labels only | False | False |

## Probe Families

- `regime_only`: `pred_tmag + dt + k + S5 base scale`.
- `fine_token_only`: compact fine-token summary only.
- `spherical_token_only`: compact spherical-token summary only.
- `token_plus_regime`: both token summaries plus `pred_tmag + dt + k`.
- Models used: logistic regression only.

## CV Results

| family | folds | AUC mean/std | acc mean/std | high-error recall mean/std | precision mean/std | worst-bucket recall mean/std | exceeds regime_only |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| regime_only | 2 | 0.918268/0.023734 | 0.904113/0.001769 | 0.811792/0.038208 | 0.706553/0.052707 | 1.000000/0.000000 | False |
| fine_token_only | 2 | 0.774098/0.048284 | 0.808218/0.000375 | 0.112736/0.037264 | 0.642857/0.357143 | 0.062500/0.062500 | False |
| spherical_token_only | 2 | 0.489494/0.079611 | 0.463565/0.118467 | 0.394340/0.205660 | 0.136139/0.001004 | 0.312500/0.312500 | False |
| token_plus_regime | 2 | 0.892280/0.034115 | 0.894409/0.042846 | 0.711792/0.061792 | 0.770256/0.158316 | 0.812500/0.187500 | False |

## Core Comparison

- Best token-only family: `fine_token_only`.
- token-only vs regime_only: delta_auc=`-0.144171`, delta_accuracy=`-0.095895`, delta_high_error_recall=`-0.699057`, exceeds=`False`.
- token_plus_regime vs regime_only: delta_auc=`-0.025988`, delta_accuracy=`-0.009704`, delta_high_error_recall=`-0.100000`, exceeds=`False`.

## Decision

- Evidence sufficient for S8c decision: `True`.
- Recommend S8c follow-up final router test later: `False`.
- S5 remains final clean candidate: `True`.
- Final classification: `TOKEN-NO-ADDED-VALUE`.
