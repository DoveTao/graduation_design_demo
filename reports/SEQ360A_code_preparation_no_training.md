# SEQ360A code preparation no training

## Purpose
- Prepare sequence clip data loading and sequence consistency losses for future fine-tuning from `FINAL360I_struct360b_final_selected`.
- Address trajectory composition consistency, local scale drift, and k-step relative pose consistency without explicit matching.

## Files added
- `datasets/dset2c_sequence_clip_dataset.py`
- `configs/seq360a_sequence_consistency_scale_drift.yaml`
- `tools/train_seq360a_sequence_consistency_scale_drift.py`

## Expected future training command
```bash
conda run -n pytorch python tools/train_seq360a_sequence_consistency_scale_drift.py --config configs/seq360a_sequence_consistency_scale_drift.yaml --dry-run --smoke-batches 1
```

The current tool intentionally refuses real training by default. A later training task should explicitly add a training path after resource planning.

## Smoke test result
- command: `conda run -n pytorch python tools/train_seq360a_sequence_consistency_scale_drift.py --config configs/seq360a_sequence_consistency_scale_drift.yaml --dry-run --smoke-batches 1`
- result: `pass`
- device: `cuda`
- train clip count: `3043`
- skipped clips: `0`
- checkpoint load missing/unexpected keys: `[] / []`
- dry-run losses included adjacent pair, k-step rotation, k-step tdir, k-step log_tmag, and scale path ratio losses.

## Compliance
- no training executed: `true`
- no optimizer step executed: `true`
- no learned weights saved: `true`
- no checkpoint modified: `true`
- no explicit matching: `true`
- no match list or correspondence output: `true`
- no RANSAC / PnP / BA: `true`
- no HKUST teacher: `true`
- no BASE360 training input: `true`
- DSET2C canonical manifest-native clips: `true`
- direct raw sequence glob split: `false`
