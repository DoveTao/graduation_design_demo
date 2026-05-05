# Git Commit Summary For S2b

## Commit
- hash: `ddcf3814c6855fd67d708c1f27485dc0eac147b6`
- message: `Add S2b train-CV fine rot policy selection`

## Committed Files
- `reports/current_valid_baselines.md`
- `checkpoints/S2b_train_cv_rot_policy_selection_on_s1d5_report.md`
- `tools/s2b_train_cv_rot_policy_selection_on_s1d5.py`

## S2b Metrics
- selected fine_rot: `0.45`
- selected equals S2a diagnostic best `0.45`: `True`
- final test drift: `1.327402`
- final test ATE: `7.352371`
- final test path_ratio: `0.934984`
- final test missing/unexpected: `2 / 0`
- selected_k: `1`
- num_pairs: `132`
- num_chains: `19`
- verdict: `SUCCESS`

## Forbidden File Check
- forbidden count: `0`
- No `.pt`, `.pth`, `.ckpt`, `.npz`, `archive/*`, or `checkpoints/T57b_no_dt_multiscale_tmag_head_400/final.pt` entered this commit.

## Sanity
- `/home/dovetao/miniconda3/envs/pytorch/bin/python tools/eval_clean_policy.py --help`: passed
- `bash -n scripts/eval_s1d5_clean_policy.sh`: passed
- `bash scripts/eval_s1d5_clean_policy.sh dryrun`: passed

## current_valid_baselines
- updated: `YES`
- S2b is now recorded as a `success clean fine-rot policy candidate`.

## Post-commit `git status --short`
```text
?? reports/post_commit_workspace_cleanup_summary.md
```

## Untracked / Remaining Files
- untracked files remain: `YES`
- remaining untracked file:
  - `reports/post_commit_workspace_cleanup_summary.md`
- This file was intentionally excluded from the S2b commit because it is not part of S2b.

## Mainline Interpretation
- S2b is a clean train-CV selected candidate.
- S1d5 policy json was not modified.
- No model training was performed.
