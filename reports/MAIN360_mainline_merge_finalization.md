# MAIN360 Mainline Merge Finalization

## Executive Summary

- merged branches:
  - `maintenance/destructive-cleanup-current-mainline-only`
  - `maintenance/trim-remaining-mainline-dependencies`
  - `maintenance/reduce-legacy-comparison-labels`
  - `thesis/experiment-section-and-result-narrative`
  - `thesis/integrate-experiment-section-into-final-template`
- integration branch: `integration/mainline-clean-thesis-final`
- conflict status: none
- final visible comparison set:
  - `FINAL360I`
  - `TRAIN360E`
  - `SEQ360B`
  - `BASE360D`
  - `T57b`
- status-only experiments:
  - `SEQ360A`
  - `STRUCT360C`
- current artifact check result: pass
- py_compile result: pass
- import smoke result: pass
- dependency health check result: `pass_with_intentional_omissions`
- metrics modified: false
- training executed: false
- checkpoint committed: false

## Merge Notes

The integration branch was created from `main` and then updated by merging the cleaned mainline maintenance branches first, followed by the thesis material branches. `MAINT14` was confirmed to be already included transitively through `MAINT15`, so an explicit follow-up merge reported `already up to date`.

The resulting tree preserves the cleaned current mainline layout, the dependency-trimmed tools, the reduced legacy comparison labeling, and the thesis-ready experiment material package from `THESIS360` and `THESIS361`.

## Validation

- `tools/current/show_mainline.py`: pass
- `tools/current/check_current_artifacts.py`: pass
- `conda run -n pytorch python -m py_compile ...`: pass
- `conda run -n pytorch` import smoke:
  - `PanoramaRelPoseModel`
  - `Config`
  - `CoarseInteraction`
  - result: pass
- `tools/maint14_dependency_health_check.py`: pass with intentional omissions
  - `current_required_missing = 0`
  - `deleted_reference_count = 0`
  - `intentional_omission_count = 12`

## Intentional Omissions

The dependency health check still reports optional omissions after cleanup. These are expected and do not block the current mainline:

- old comparison summaries removed by `MAINT13`
- optional val-side comparison artifacts not needed for delivery
- optional trajectory export directories not required for current artifact checks

## Safety Checks

- no tracked `.pt`, `.pth`, or `.ckpt` files were added
- no raw data paths were staged
- no `pred_tum.txt` or `adjacent_pair_predictions.jsonl` files were staged
- local untracked checkpoint directories remain local-only and were not committed

## Recommendation

Open a PR/MR from `integration/mainline-clean-thesis-final` into `main` and keep the merge non-force. This branch is ready for review as the stable deliverable version that combines the cleaned mainline with the final thesis experiment materials.
