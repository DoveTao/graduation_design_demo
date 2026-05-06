# Final Reproducibility Guide

## Scope
This guide reproduces the finalized clean comparison after S6 lockdown. It does not run new selection, new training, or new test sweeps.

## Environment and version
- branch: `optimize/s3a0-coupled-pose-residual-head`
- recommended python: `/home/dovetao/miniconda3/envs/pytorch/bin/python`
- final policy: `checkpoints/S5_clean_tmag_calibration_policy.json`
- eval script: `scripts/eval_s5_clean_policy.sh`
- lockdown report: `checkpoints/S6_final_clean_candidate_lockdown_audit_report.md`

## Reproduce final S5 vs S2b metrics
Run:

```bash
scripts/eval_s5_clean_policy.sh
```

This command evaluates:
1. S2b baseline policy behavior
2. S5 selected clean calibration behavior

Expected metrics:
- S2b: drift=`1.327402`, ATE=`7.352371`, path_ratio=`0.934984`
- S5: drift=`1.327343`, ATE=`7.352288`, path_ratio=`0.932379`

Expected loading diagnostics:
- missing=`14`
- unexpected=`0`

## Reproduce S2b baseline alone (optional)
Run:

```bash
scripts/eval_s2b_clean_policy.sh
```

Expected S2b metrics:
- drift=`1.327402`
- ATE=`7.352371`
- path_ratio=`0.934984`

## What not to run for final reproducibility
- do not train any model
- do not run F1d/S1d6
- do not continue residual-head training
- do not do new test sweeps or candidate reselection
- do not modify `checkpoints/S5_clean_tmag_calibration_policy.json`
- do not use test statistics or gt_tmag for policy construction/inference-time gating

## Final interpretation rule
If reproduced values match within normal float tolerance and loading stays `unexpected=0`, final candidate remains:
`S5_clean_tmag_calibration_policy` with the conclusion "clean but marginal gain."
