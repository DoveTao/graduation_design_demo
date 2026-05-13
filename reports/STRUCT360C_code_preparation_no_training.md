# STRUCT360C code preparation no training

## Purpose
- Prepare a STRUCT360B challenger with coarse-rotation-aware spherical attention bias diagnostics.
- The added module rotates A-sphere token bearings by coarse `R0` and computes angular latent bias against B-sphere tokens.
- The bias is latent only and does not emit nearest neighbors, top-k matches, or correspondence lists.

## Files added
- `models/struct360c_rotation_aware_fine_refinement.py`
- `configs/struct360c_rotation_aware_fine_refinement.yaml`
- `tools/train_struct360c_rotation_aware_fine_refinement.py`

## Expected future training command
```bash
conda run -n pytorch python tools/train_struct360c_rotation_aware_fine_refinement.py --config configs/struct360c_rotation_aware_fine_refinement.yaml --dry-run --smoke-batches 1
```

The current tool is a dry-run preparation entry and refuses real training by default.

## Smoke test result
- command: `conda run -n pytorch python tools/train_struct360c_rotation_aware_fine_refinement.py --config configs/struct360c_rotation_aware_fine_refinement.yaml --dry-run --smoke-batches 1`
- result: `pass`
- checkpoint load missing/unexpected keys: `[] / []`
- output `R` shape: `[1, 3, 3]`
- output `tdir` shape: `[1, 3]`
- output `tmag` shape: `[1]`
- latent bias mean/std: `-4.478407382965088 / 1.935436725616455`
- correspondences output: `false`

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
- direct raw sequence glob split: `false`
