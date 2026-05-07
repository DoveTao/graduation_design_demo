# S15d Train/Eval Forward Parity Fix Report

## Executive summary
- final classification: `FORWARD-PARITY-FIX-PASS`
- main root cause: `fixed train-mode preservation removed unintended train/eval parity drift`
- S15 can continue: `True`
- S5 remains final clean candidate: `yes`

## Zero-update wrapped audit
- wrapped R diff: `0.000000`
- wrapped tvec diff: `0.000000`
- wrapped tmag diff: `0.000000`
- missing/unexpected direct: `14 / 0`
- missing/unexpected training: `14 / 0`

## Deterministic forward audit
- configured mode: `mixed_train_eval_subtrees` | root_train=`True`
- eval vs configured train: R_diff=`0.000000`, tvec_diff=`0.000000`, tmag_diff=`0.000000`
- eval vs force-all-dropout-eval: R_diff=`0.000000`, tvec_diff=`0.000000`, tmag_diff=`0.000000`
- grad/no_grad eval: R_diff=`0.000000`, tvec_diff=`0.000000`, tmag_diff=`0.000000`
- grad/no_grad configured train: R_diff=`0.000000`, tvec_diff=`0.000000`, tmag_diff=`0.000000`
- grad forward mismatch: `False`

## Dropout / hidden train-call audit
- functional dropout found: `False`
- explicit training=True found: `False`
- hidden train call found: `False`
- train() matches: `1`

## One-update tiny audit
- standard mode wrapped: R_delta=`0.000000`, tvec_delta=`0.001044`, tmag_delta=`0.001129`
- standard mode raw: R_delta=`0.000000`, tdir_delta=`0.000000`, tvec_delta=`0.001129`, tmag_delta=`0.001129`
- eval-forward mode wrapped: R_delta=`0.000000`, tvec_delta=`0.001044`, tmag_delta=`0.001129`
- eval-forward mode raw: R_delta=`0.000000`, tdir_delta=`0.000000`, tvec_delta=`0.001129`, tmag_delta=`0.001129`

## Mutable state audit
- standard frozen params changed: `0`
- standard buffers changed: `0`
- standard forbidden optimizer params: `0`
- eval-forward frozen params changed: `0`
- eval-forward buffers changed: `0`
- eval-forward forbidden optimizer params: `0`

## Decision
- final classification: `FORWARD-PARITY-FIX-PASS`
- whether S15 can continue: `True`
- next step: `S15e tiny trajectory retest`
- S5 remains final clean candidate: `yes`
