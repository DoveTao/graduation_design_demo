# MAINT11 Current Mainline Layout And Core Refactor Phase1

## What Changed

- added `CURRENT_MAINLINE.md` as the top-level quick orientation document
- added `tools/current/README.md`, `tools/current/show_mainline.py`, and `tools/current/check_current_artifacts.py`
- created `train360/core/` for current runtime modules
- created `train360/legacy/` for preserved MVP-era modules
- replaced moved root-level files with backward-compatible wrappers
- migrated current mainline and active challenger tools/models to `train360.core.*` imports

## Compatibility

- root-level wrapper imports are preserved
- legacy `train_mvp.py` and `dataset_pano_only.py` remain importable
- current DSET2C mainline continues to use manifest-native datasets
- no checkpoints, raw data, reports, or baseline result trees were moved

## Smoke Summary

- py_compile: `True`
- root wrapper import smoke: `True`
- legacy import smoke: `True`
- FINAL360I smoke: `True`
- TRAIN360E smoke: `True`
- SEQ360B smoke: `True`
- STRUCT360C smoke: `True`
