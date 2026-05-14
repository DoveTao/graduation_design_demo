# Current Tools Index

Use this directory when you need to orient yourself quickly.

- `show_mainline.py`: prints the current thesis mainline files and key result references.
- `check_current_artifacts.py`: checks that the expected current configs, reports, and checkpoints exist.

Current pair-level mainline:
- `tools/final360i_retrain_and_select.py`
- `configs/final360i_struct360b_final.yaml`
- `models/struct360b_match_free_coarse_to_fine.py`

Current trajectory evaluation path:
- `tools/train360e_sequence_trajectory_export_and_ate_eval.py`

Current challenger entrypoints:
- `tools/train_struct360c_rotation_aware_fine_refinement.py`
- `tools/train_seq360a_sequence_consistency_scale_drift.py`
- `tools/train360d_observability_kstep_scale.py`

