# MAINT11 Import Refactor Smoke

- branch: `maintenance/current-mainline-layout-and-core-refactor-phase1`
- smoke device: `cuda`
- py_compile: `True`
- root wrapper import smoke: `True`
- legacy import smoke: `True`
- FINAL360I smoke: `True`
- TRAIN360E smoke: `True`
- SEQ360B smoke: `True`
- STRUCT360C smoke: `True`

## Notes

- `SEQ360B` has no current in-repo training tool, so its smoke is recorded as artifact/checkpoint compatibility rather than a tool-driven forward pass.
- `FINAL360I` and `STRUCT360C` smokes use real checkpoint loads and 1-batch no-grad forwards on the manifest-native DSET2C loader.
- `TRAIN360E` smoke validates trajectory-export primitives and current artifact paths without running a full evaluator sweep.
