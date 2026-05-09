# JRT1 Branch Remote Sync Audit

## Branch

`experiment/jrt1-joint-rtdir-coupled-refiner`

## Latest commit

`355ac03 Add JRT1 final closeout summary`

## Working tree

Clean before creating this audit report.

## Remote sync

The experiment branch was pushed to `origin` with upstream tracking set.

- remote: `origin/experiment/jrt1-joint-rtdir-coupled-refiner`
- ahead: `0`
- behind: `0`

## JRT1 final classification

`NO_STABLE_JRT1_TRAJECTORY_GAIN`

## Final decision

- no final test run
- no selected candidate
- S5 remains final clean candidate

## Validation

- run_jrt1_closeout_summary: PASS
- verify_final_candidate: PASS
- project_health_check: PASS
- unittest: PASS, 60 tests

## S5 locked metrics

- ATE = 7.352288
- drift = 1.327343
- path_ratio = 0.932379

## Merge policy

Do not merge this experiment branch into `polish/final-reproducibility-guardrails` unless explicitly requested.
JRT1 is archived as a negative but informative experiment branch.
