# MAINT13 Destructive Repo Cleanup Keep Current Mainline Only

- destructive cleanup executed = true
- backup tag = `backup/pre-maint13-destructive-cleanup-20260514-174819`
- files deleted count = `1036`
- tracked files deleted count = `1027`
- untracked files deleted count = `9`
- files moved = false
- checkpoint deleted = false
- raw data deleted = false
- current mainline preserved = `true`
- show_mainline pass = `true`
- check_current_artifacts pass = `true`
- py_compile pass = `true`
- import smoke pass = `true`
- optional FINAL360I 1-batch forward smoke = `true`

## Kept Focus

- current pair-level thesis mainline: `FINAL360I_struct360b_final_selected`
- kept runtime package: `train360/core/`
- kept trajectory eval path: `tools/train360e_sequence_trajectory_export_and_ate_eval.py`
- kept non-mainline variant: `SEQ360B`
- removed old root wrappers, legacy package, old ablations, maintenance noise, and one-off historical tools/reports

## Recovery

- restore one file: `git checkout backup/pre-maint13-destructive-cleanup-20260514-174819 -- path/to/file`
- restore whole pre-cleanup state: `git checkout backup/pre-maint13-destructive-cleanup-20260514-174819`
- deleted-file manifest: `reports/MAINT13_deleted_files_manifest.json`
