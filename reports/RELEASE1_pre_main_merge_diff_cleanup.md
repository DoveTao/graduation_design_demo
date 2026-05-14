# RELEASE1 Pre-Main Merge Diff Cleanup

## Executive Summary

- cleanup executed: `true`
- ignored untracked checkpoint dirs: `true`
- checkpoint files removed from Git diff count: `245`
- raw data files removed from Git diff count: `0`
- legacy noise files removed count: `28`
- current mainline preserved: `true`
- forbidden files remaining count: `0`
- legacy noise remaining count: `0`
- current artifact check result: `pass`
- py_compile result: `pass`
- import smoke result: `pass`
- dependency health result: `pass_with_intentional_omissions`
- ready_to_merge_main: `true`

## Scope

This cleanup was performed on `integration/mainline-clean-thesis-final` to slim the merge diff against `main` before any actual main-branch merge.

Untracked local checkpoint directories were explicitly ignored as local artifacts:

- `checkpoints/FINAL360I_struct360b_final/`
- `checkpoints/STRUCT360A_implicit_spherical_cross_attention/`
- `checkpoints/STRUCT360B_match_free_coarse_to_fine/`
- `checkpoints/STRUCT360C_rotation_aware_fine_refinement/`
- `checkpoints/TRAIN360D_observability_kstep_scale/`
- `checkpoints/TRAIN360H_sweep/`

They were not deleted, moved, or committed.

## Diff Slimming Actions

1. Removed tracked `checkpoints/` artifacts from the staged integration diff.
2. Removed tracked historical external baseline output files under `external_baselines/results/base360_hkust_360dvo_official/`.
3. Removed legacy tracked noise from the staged diff:
   - `TRAIN360C/D/H`
   - `STRUCT360A`
   - `train_mvp.py`
   - `dataset_pano_only.py`
4. Restored old `main`-side legacy files when necessary so they would not continue to pollute the staged diff as deletion entries.
5. Tightened `.gitignore` to explicitly cover:
   - `checkpoints/`
   - `*.pt`
   - `*.pth`
   - `*.ckpt`
   - `data/`
   - `**/pred_tum.txt`
   - `**/adjacent_pair_predictions.jsonl`
   - `**/train_log.jsonl`
   - `backup_patches/`
   - `__pycache__/`
   - `*.pyc`

## Diff Counts

- pre-clean diff entries: `448`
- post-clean staged diff entries against `main`: `106`
- forbidden diff paths before: `245`
- forbidden diff paths after: `0`
- legacy noise paths before: `28`
- legacy noise paths after: `0`

## Preserved Deliverables

The staged diff still contains the intended deliverable content:

- cleaned current-mainline docs
- `train360/core/`
- retained DSET2C manifest-native datasets
- retained `STRUCT360B` / `SEQ360B` models and tools
- retained `FINAL360I`, `TRAIN360E`, `SEQ360B`, `BASE360D`, `RESULTS360` reports
- `SEQ360A` and `STRUCT360C` status summaries
- `MAINT13`, `MAINT14`, `MAINT15` retained cleanup-chain reports
- `THESIS360` / `THESIS361` materials
- canonical DSET2C train/val/test manifests

## Validation

- required files post-clean: `pass`
- current entrypoint:
  - `show_mainline`: `pass`
  - `check_current_artifacts`: `pass`
- `py_compile`: `pass`
- import smoke: `pass`
- dependency health: `pass_with_intentional_omissions`
  - `current_required_missing = 0`
  - `deleted_reference_count = 0`

## Remaining Blockers

None.

## Recommendation

This integration branch is now ready for the final pre-merge human review refresh and then a normal non-force merge into `main`.
