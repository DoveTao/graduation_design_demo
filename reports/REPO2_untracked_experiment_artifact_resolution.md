# REPO2 Untracked Experiment Artifact Resolution

## Initial State

S5D2 was already pushed on `experiment/s5d2-dense-export-convention-audit`, tracked diff was empty, and the working tree contained grouped untracked experiment artifacts from MF1, ORB-SLAM3 baseline work, and S5 dense same-evaluator comparison work.

Intended target branches at the start of this audit:

- `experiment/mf1-multi-frame-chain-refiner` for MF1 chain-refiner artifacts
- `experiment/orbslam3-fisheye-strong-baseline` for ORB-SLAM3 fisheye baseline artifacts
- `experiment/orbslam3-fisheye-strong-baseline` for S5 dense same-evaluator comparison artifacts
- leave large build outputs and uncertain files uncommitted unless clearly assigned

## MF1 Resolution

- target branch: `experiment/mf1-multi-frame-chain-refiner`
- validation:
  - `bash scripts/verify_final_candidate.sh`: `PASS`
  - `bash scripts/project_health_check.sh`: `PASS`
  - `python -m unittest discover -s tests -q`: `PASS (124 tests)`
  - `bash scripts/run_mf1_closeout_summary.sh`: `PASS`
- commit hash: `ea559d5`
- commit message: `Add MF1 multi-frame chain refiner artifacts`
- push status: `pushed to origin/experiment/mf1-multi-frame-chain-refiner`
- branch sync after push: `ahead 0 / behind 0`
- remaining MF1 untracked files on the MF1 branch after push: `none`

## ORB-SLAM3 Resolution

- target branch: `experiment/orbslam3-fisheye-strong-baseline`
- validation:
  - `bash scripts/verify_final_candidate.sh`: `PASS`
  - `bash scripts/project_health_check.sh`: `PASS`
  - `python -m unittest discover -s tests -q`: `PASS (94 tests)`
  - `bash scripts/run_external_baseline_comparison.sh`: `PASS`
- commit hash: `abea042`
- commit message: `Add ORB-SLAM3 fisheye strong baseline artifacts`
- push status: `pushed to origin/experiment/orbslam3-fisheye-strong-baseline`
- branch sync after push: `ahead 0 / behind 0`
- `tools/audit_raw_fisheye_dataset.py` was committed here because it is the ORB1a raw fisheye dataset audit tool.

## S5D Dense Comparison Resolution

- target branch used: `experiment/orbslam3-fisheye-strong-baseline`
- rationale: `reports/s5_orbslam3_same_evaluator_comparison.md` explicitly compares S5 dense export against ORB-SLAM3 under the same external evaluator
- commit hash: `797865b`
- commit message: `Add S5 dense same-evaluator comparison artifacts`
- push status: `pushed to origin/experiment/orbslam3-fisheye-strong-baseline`
- branch sync after push: `ahead 0 / behind 0`

## Unknown Files

- `tools/audit_raw_fisheye_dataset.py`: committed to the ORB-SLAM3 branch after confirming it belongs to `ORB1a_fisheye_dataset_audit`

## Branch Sync Summary

| branch | upstream | ahead | behind |
|---|---|---:|---:|
| `experiment/mf1-multi-frame-chain-refiner` | `origin/experiment/mf1-multi-frame-chain-refiner` | 0 | 0 |
| `experiment/orbslam3-fisheye-strong-baseline` | `origin/experiment/orbslam3-fisheye-strong-baseline` | 0 | 0 |
| `experiment/s5d2-dense-export-convention-audit` | `origin/experiment/s5d2-dense-export-convention-audit` | 0 | 0 |

## Remaining Working Tree State

Current branch after returning to S5D2: `experiment/s5d2-dense-export-convention-audit`

`git status --short`:

```text
?? external_baselines/runners/orbslam3_mono_fisheye_runner
```

This remaining file is an ELF executable build artifact. It was intentionally left untracked and uncommitted.

## Protected Files

- S5 policy diff: `none`
- final manifest diff: `none`
- split / eval convention diff in protected checked files: `none`

## Recommended Next Actions

- Keep `external_baselines/runners/orbslam3_mono_fisheye_runner` untracked unless the user explicitly wants to version a compiled runner binary.
- Continue using branch-by-branch commits for future experiment artifacts.
- Preserve the current separation: MF1 on its own branch, ORB/S5 dense comparison on the ORB baseline branch, and S5D2 audit on the S5D2 branch.
