# S3b Fine Token Representation Diagnostic Report

## 1. Executive summary

- final classification: `DT/K-DOMINATED-ERROR`
- summary: frozen fine features do not generalize well, while dt / k / magnitude features explain most of the readable pair-level error structure
- fine token stable residual signal: `False`
- worth continuing head training now: `False`

## 2. Feature extraction summary

- policy path: `/home/dovetao/graduation_design_demo/checkpoints/S2b_clean_fine_rot_policy.json`
- base checkpoint: `/home/dovetao/graduation_design_demo/checkpoints/T57b_no_dt_multiscale_tmag_head_400/final.pt`
- explicit cfg / policy-path requirement satisfied: `True`
- missing / unexpected: `14` / `0`
- fine_rot / fine_tdir / fine_tmag: `0.45` / `0.0` / `0.0`
- use_geometry_refine: `False`
- train pair samples: `512`
- test pair samples: `512`
- pair sampling cap train/test: `512` / `512`
- extracted features include:
  fine pooled feature, coarse pooled feature, entropy/top1/epi-mass/confidence, predicted pose representation, dt/k/tmag, allowed-mask density, routing recall

## 3. Probe results table

| feature set | test rot R2 | test rot MAE | test tdir R2 | test tdir MAE | test rot-bucket acc | test tdir-bucket acc | test high-error acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| coarse_only | -371.6710 | 19.4034 | 0.9743 | 3.2371 | 0.3809 | 0.7910 | 0.7402 |
| fine_only | -28.7979 | 4.6516 | -14.8333 | 80.9895 | 0.2207 | 0.1406 | 0.3730 |
| coarse_fine | -42.0312 | 5.9108 | 0.8191 | 8.4928 | 0.1660 | 0.1406 | 0.3770 |
| confidence_only | -15.5660 | 3.9617 | -0.6564 | 23.1088 | 0.3809 | 0.2402 | 0.3887 |
| dt_k_tmag_only | -1.1540 | 1.2308 | -1.3811 | 26.2972 | 0.1895 | 0.2246 | 0.6914 |
| all_features | -40.3568 | 5.7592 | 0.8334 | 8.0802 | 0.1797 | 0.1406 | 0.3906 |

## 4. Generalization analysis

- all-features mean train-test R2 gap: `20.7581`
- fine feature better than coarse feature on test: `False`
- confidence-only nearly explains all-features: `True`
- dt/k/tmag-only nearly explains all-features: `True`
- overfitting concern visible: `True`

## 5. High-error clustering / nearest-neighbor summary

- high-error base rate on test: `0.3730`
- fine-feature high-error NN purity@5: `0.8084`
- high-rot-error NN purity@5: `0.5860`
- high-tdir-error NN purity@5: `0.9511`
- best k-means high-error cluster purity: `0.7217`
- bad chains share similar feature signatures: `nan` neighbor purity

## 6. Chain-level diagnostic

- test chain ATE-proxy R2: `nan`
- test chain drift-proxy R2: `nan`
- test chain path-ratio R2: `nan`
- test high-ATE chain accuracy: `1.0000`
- train/test chain counts: `1` / `1`
- interpretation: chain-level probe uses mean/final metric position error as chain ATE/drift proxies from the odometry debug chain summaries.
- limitation: the current train/test split yields only one debug chain per split at `k=1`, so chain-level generalization remains weakly identified.

## 7. Final classification

- `DT/K-DOMINATED-ERROR`

## 8. Next-step recommendation

- return to dt/k-aware evaluation or data-distribution diagnosis
