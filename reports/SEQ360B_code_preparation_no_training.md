# SEQ360B code preparation no training

## Purpose
- Prepare a lightweight sequence scale smoothing head for future path-ratio drift correction.
- The head modulates only adjacent-pair `log_tmag` with a bounded delta and leaves rotation and translation direction unchanged.

## Files added
- `models/seq360b_scale_smoothing_head.py`
- `configs/seq360b_lightweight_scale_smoothing.yaml`
- `tools/train_seq360b_lightweight_scale_smoothing.py`

## Expected future training command
```bash
conda run -n pytorch python tools/train_seq360b_lightweight_scale_smoothing.py --config configs/seq360b_lightweight_scale_smoothing.yaml --dry-run --smoke-batches 1
```

The current tool is a dry-run preparation entry and does not train by default.

## Smoke test result
- command: `conda run -n pytorch python tools/train_seq360b_lightweight_scale_smoothing.py --config configs/seq360b_lightweight_scale_smoothing.yaml --dry-run --smoke-batches 1`
- result: `pass`
- checkpoint load missing/unexpected keys: `[] / []`
- context dimension: `768`
- adjacent pairs in smoke clip: `2`
- rotation modified: `false`
- tdir modified: `false`
- delta clamp: `[-0.3, 0.3]`

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
