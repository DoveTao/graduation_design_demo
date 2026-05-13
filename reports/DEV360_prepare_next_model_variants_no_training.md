# DEV360 prepare next model variants no training

## Prepared variants
- `SEQ360A_sequence_consistency_scale_drift_stabilization`
- `SEQ360B_lightweight_scale_smoothing_head`
- `STRUCT360C_rotation_aware_fine_refinement`

## Problem addressed
- SEQ360A: trajectory composition consistency through contiguous manifest-native clips and k-step composition losses.
- SEQ360B: scale/path drift through bounded sequence-context `log_tmag` smoothing.
- STRUCT360C: rotation-aware fine refinement through coarse-R-conditioned spherical latent attention bias.

## Recommended future order
- Run `SEQ360B` first if disk/time is limited because it is the smallest scale-only variant.
- Run `SEQ360A` first if full sequence training budget is available.
- Run `STRUCT360C` after sequence variants because it changes the refinement architecture.

## Smoke test status
- py_compile for all new Python files: `pass`
- SEQ360A dry-run smoke: `pass`
- SEQ360B dry-run smoke: `pass`
- STRUCT360C dry-run smoke: `pass`

## Commands executed
```bash
conda run -n pytorch python -m py_compile datasets/dset2c_sequence_clip_dataset.py models/seq360b_scale_smoothing_head.py models/struct360c_rotation_aware_fine_refinement.py tools/train_seq360a_sequence_consistency_scale_drift.py tools/train_seq360b_lightweight_scale_smoothing.py tools/train_struct360c_rotation_aware_fine_refinement.py
conda run -n pytorch python tools/train_seq360a_sequence_consistency_scale_drift.py --config configs/seq360a_sequence_consistency_scale_drift.yaml --dry-run --smoke-batches 1
conda run -n pytorch python tools/train_seq360b_lightweight_scale_smoothing.py --config configs/seq360b_lightweight_scale_smoothing.yaml --dry-run --smoke-batches 1
conda run -n pytorch python tools/train_struct360c_rotation_aware_fine_refinement.py --config configs/struct360c_rotation_aware_fine_refinement.yaml --dry-run --smoke-batches 1
```

## Compliance checklist
- training executed: `false`
- fine-tune executed: `false`
- long evaluation executed: `false`
- learned weights saved: `false`
- checkpoints modified: `false`
- `.pt` committed: `false`
- explicit matching used: `false`
- match list / correspondence list output: `false`
- RANSAC / PnP / BA used: `false`
- HKUST 360DVO teacher used: `false`
- BASE360 outputs used as training input: `false`
- direct glob `data/360DVO/Sequences/*` split definition: `false`
- S5 locked metrics modified: `false`
