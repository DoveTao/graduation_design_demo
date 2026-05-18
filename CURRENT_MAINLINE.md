# Current Mainline

## Open These First

- mainline summary: `CURRENT_MAINLINE.md`
- structure guide: `PROJECT_STRUCTURE.md`
- quick terminal entrypoint: `tools/current/show_mainline.py`
- artifact check: `tools/current/check_current_artifacts.py`

## Current Thesis Mainline

- main model: `FINAL360M_fulltrain_struct360b_thesis_main_guarded`
- main structure lineage: `STRUCT360B_match_free_coarse_to_fine`
- pair-level config: `configs/final360m_fulltrain_struct360b_thesis_main_guarded.yaml`
- pair-level trainer/selector: `tools/train_final360m_fulltrain_thesis_main.py`
- selected checkpoint: `checkpoints/FINAL360M_fulltrain_struct360b_thesis_main_guarded/best_full_val.pt`
- protocol status: `full-train`, `selected_by_full_val=true`, `test_used_for_selection=false`
- pair-level base model: `models/struct360b_match_free_coarse_to_fine.py`
- dataset path: `datasets/dset2c_manifest_dataset.py`
- trajectory evaluation refresh: `tools/final360m_trajectory_thesis_refresh.py`
- refreshed eval-only backend tools:
  - `tools/odom360a_lightweight_trajectory_fusion.py`
  - `tools/odom360b_local_pose_graph_kstep.py`
- runtime core package: `train360/core/`

## Old FINAL360I Status

- `FINAL360I_struct360b_final_selected` is no longer the thesis final main model
- audit finding: `train_subset_max=256`
- effective train sample count: `256`
- required thesis status: `subset-trained candidate only`

## Preserved Variant

- sequence scale variant: `SEQ360B`
- variant full name: `SEQ360B_lightweight_scale_smoothing_head`
- config: `configs/seq360b_lightweight_scale_smoothing.yaml`
- tool: `tools/train_seq360b_lightweight_scale_smoothing.py`
- model head: `models/seq360b_scale_smoothing_head.py`

## External Baselines

- official external baseline: `BASE360D` (`HKUST official 360DVO`)
- legacy baseline reference: `T57b`

## Current Pair-Level Reference

- `FINAL360M rot_mean = 2.322841551013553`
- `FINAL360M signed_tdir_mean = 56.07180781302633`
- `FINAL360M anti_parallel_rate = 0.20762544448834452`
- `FINAL360M tmag_median_ratio = 0.8838564229456283`
- `FINAL360M path_ratio = 0.6814703365128686`

## Current Caveat

- `FINAL360M` is protocol-qualified because it is the only full-train thesis-main result, not because it is uniformly better than old `FINAL360I`
- pair-level relation to old `FINAL360I`: `mixed`
- direct adjacent-pair trajectory composition still drifts strongly
- `FINAL360M-ODOM360A` improves `FINAL360M-direct`
- refreshed `FINAL360M` trajectory backend is still weaker than old subset-model-based `ODOM360A`
- `tdir` remains the core bottleneck for both pair-level and trajectory-level behavior

## Status-Only Experiments

- `SEQ360A`: no_improvement; retained only as `reports/SEQ360A_status_summary.md`
- `STRUCT360C`: evaluation_failed; retained only as `reports/STRUCT360C_status_summary.md`
