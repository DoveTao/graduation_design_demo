# Mainline 360 Repo

## Start Here

1. Current mainline: `CURRENT_MAINLINE.md`
2. Quick terminal view: `python tools/current/show_mainline.py`
3. Current artifact check: `python tools/current/check_current_artifacts.py`

## Current Main Model

- `FINAL360I_struct360b_final_selected`

## Current Visible Comparison Set

- `FINAL360I` main pair-level model
- `TRAIN360E` trajectory / ATE evaluation path for `FINAL360I`
- `SEQ360B` retained sequence-scale variant
- `BASE360D` HKUST official `360DVO` external baseline
- `T57b` legacy recovered external reference

## Current Core Directories

- `train360/core/`
- `datasets/`
- `models/`
- `tools/`
- `configs/`

## Not Recommended

- old MVP/raw-scan entrypoints have been removed from the active tree
- `SEQ360A`: no_improvement; retained only as `reports/SEQ360A_status_summary.md`
- `STRUCT360C`: evaluation_failed; retained only as `reports/STRUCT360C_status_summary.md`
