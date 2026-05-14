# MAIN361 Final Pre-Main Merge Review

## Executive Summary

- ready_to_merge_main: `false`
- blocking_issues count: `2`
- warnings count: `2`
- recommended action: `do_not_merge_blocking_issues`

This integration branch is healthy as a runnable cleaned mainline plus thesis-material branch, but it is **not** yet suitable for direct merge into `main` in its current form. The key reason is that, relative to the current `main`, the branch would reintroduce a large tracked `checkpoints/` tree and a substantial amount of legacy historical material that the cleaned deliverable was supposed to avoid.

## Branch Status

- current branch: `integration/mainline-clean-thesis-final`
- latest commit: `ae5e9a1`
- compared against: `main`
- local workspace note: untracked local checkpoint directories are still present in the working tree, but they were not staged in this review pass

## Required File Check

- result: `pass`
- source: [MAIN361_required_files_check.json](/home/dovetao/graduation_design_demo/reports/MAIN361_required_files_check.json)
- key retained files are present:
  - cleaned entry docs: `README.md`, `CURRENT_MAINLINE.md`, `PROJECT_STRUCTURE.md`
  - current tools: `tools/current/*`, `tools/final360i_retrain_and_select.py`, `tools/train360e_sequence_trajectory_export_and_ate_eval.py`, `tools/train_seq360b_lightweight_scale_smoothing.py`
  - retained datasets/models/configs
  - retained metrics reports
  - `THESIS361` final thesis files

## Current Entrypoint Check

- `show_mainline`: `pass`
- `check_current_artifacts`: `pass`
- `current_required missing = 0`
- source: [MAIN361_current_entrypoint_check.txt](/home/dovetao/graduation_design_demo/reports/MAIN361_current_entrypoint_check.txt)

The visible comparison set in the mainline entrypoint is correct:

- `FINAL360I`
- `TRAIN360E`
- `SEQ360B`
- `BASE360D`
- `T57b`

Status-only experiments are still correctly presented as archived summaries:

- `SEQ360A`
- `STRUCT360C`

## Static Validation

- `py_compile`: `pass`
- import smoke: `pass`
- dependency health: `pass_with_intentional_omissions`
- `deleted artifact still required`: `false`
- source files:
  - [MAIN361_compile_import_smoke.json](/home/dovetao/graduation_design_demo/reports/MAIN361_compile_import_smoke.json)
  - [MAIN361_dependency_health_review.md](/home/dovetao/graduation_design_demo/reports/MAIN361_dependency_health_review.md)

## Forbidden File Check

- tracked `.pt/.pth/.ckpt` files: `none detected`
- tracked raw data paths in diff: `none detected`
- tracked large trajectory outputs like `pred_tum.txt` / `adjacent_pair_predictions.jsonl` in diff: `none detected`
- **blocking issue**: tracked `checkpoints/` paths are present in the integration diff vs `main`
  - `git diff --name-only main...integration/mainline-clean-thesis-final | grep '^checkpoints/' | wc -l` = `245`
  - tracked forbidden-path scan entries under `checkpoints/` = `501`
- source: [MAIN361_large_forbidden_file_check.txt](/home/dovetao/graduation_design_demo/reports/MAIN361_large_forbidden_file_check.txt)

Interpretation:

The branch does not introduce model weight files, but it **does** carry a large tracked historical `checkpoints/` tree and many audit/result artifacts under `checkpoints/`. That is incompatible with the stated goal of keeping `main` focused on the cleaned current mainline.

## Thesis Material Check

- `THESIS361` files present: `true`
- claims/caveats checklist present: `true`
- no overclaim detected in the retained thesis package
- caveat remains explicit that `FINAL360I` does **not** fully surpass official `360DVO` as a complete VO pipeline
- source files:
  - [MAIN361_thesis_tree.txt](/home/dovetao/graduation_design_demo/reports/MAIN361_thesis_tree.txt)
  - [THESIS361_claims_and_caveats_checklist.md](/home/dovetao/graduation_design_demo/thesis/THESIS361_claims_and_caveats_checklist.md)

## Legacy Noise Check

- result: `not acceptable for direct merge to main`
- source: [MAIN361_legacy_label_scan.txt](/home/dovetao/graduation_design_demo/reports/MAIN361_legacy_label_scan.txt)

Observed status:

- some legacy mentions are acceptable inside thesis caveats, archived status summaries, and maintenance review files
- however, the branch diff vs `main` still includes many historical artifacts such as:
  - tracked `checkpoints/` audit/result trees
  - retained old reports like `TRAIN360C`, `TRAIN360D`, `TRAIN360H`, and `STRUCT360A`
  - extra historical tools such as `tools/run_base360_hkust_360dvo_official.py`

This means the integration branch is functionally clean at the entrypoint level, but structurally still too noisy for a final `main` merge if the target is a minimal deliverable repository.

## Blocking Issues

1. The integration diff against `main` includes a large tracked `checkpoints/` tree and related historical artifacts, which violates the intended no-checkpoint/no-noise finalization standard for `main`.
2. The integration diff still carries substantial legacy report/tool material (`TRAIN360C/D/H`, `STRUCT360A`, and similar historical outputs), so the cleaned mainline is not yet the only visible repository state once merged into `main`.

## Warnings

1. The working tree is not literally clean because local untracked checkpoint directories remain present. They were not staged here, but a human merge operator should still verify `git status --short` before any final `main` merge.
2. Dependency health is only `pass_with_intentional_omissions`, which is acceptable for the retained mainline but should still be documented in any release note.

## Final Recommendation

- recommendation: `do_not_merge_blocking_issues`

Suggested next step:

Create one more integration-cleanup pass that removes tracked historical `checkpoints/` artifacts and trims the remaining legacy reports/tools from the merge diff while preserving:

- cleaned current mainline entry docs
- retained current code paths
- retained current metrics
- `THESIS360/361` materials

Only after that pass should `main` receive the cleaned integration branch.
