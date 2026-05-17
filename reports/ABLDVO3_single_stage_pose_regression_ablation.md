# ABLDVO3 single-stage pose regression ablation

## 1. Executive summary
- executed true/false: `true`
- model trained: `ABLDVO3_SingleStagePoseRegression`
- train subset count: `512`
- mini-val count: `1024`
- classification: `completed`
- main conclusion: `under the same subset regime as ABLDVO2, single-stage pose regression is clearly worse than FINAL360I on signed translation direction and shows severe scale/path overshoot, so the coarse-to-fine residual refinement contributes positively, especially to translation stability.`

## 2. Why ABLDVO3
- `ABLDVO2` already measured the effects of removing `spherical-aware representation` and `cross-image interaction`.
- the missing factor was whether `FINAL360I` still benefits after those two major ingredients are already kept, i.e. whether `coarse-to-fine residual refinement` matters on its own.
- `ABLDVO3` isolates that question by preserving spherical-aware encoding and cross-image interaction while removing pose-conditioned residual refinement and residual composition.

## 3. Experimental setup
- manifests:
  - train: `external_baselines/results/dset2c_360dvo_canonical/pair_manifest_train.jsonl`
  - val: `external_baselines/results/dset2c_360dvo_canonical/pair_manifest_val.jsonl`
  - test: `external_baselines/results/dset2c_360dvo_canonical/pair_manifest_test.jsonl`
- subset strategy:
  - train subset count `512`
  - mini-val count `1024`
  - seed `1`
  - same sampler rule as ABLDVO2
- training epochs: `3`
- optimizer: `AdamW`
- GPU: `NVIDIA GeForce RTX 3060 Laptop GPU`
- selection protocol: `mini-val only during training; full val and full test once for the selected best checkpoint`

## 4. Model definition
- preserved:
  - spherical-aware tokenization / spherical positional representation already used by the ABLDVO line
  - cross-image interaction through the same coarse interaction path
  - scale-aware output protocol with `R / tdir / log_tmag`
- removed:
  - coarse pose to fine residual refinement
  - pose-conditioned refinement
  - residual gate
  - residual regularization
  - final residual composition
- isolation logic:
  - this is not `PlainPairVO`
  - this is not `NoCrossImageInteraction`
  - it keeps `spherical-aware + cross-image interaction` and only removes `coarse-to-fine residual refinement`

## 5. Validation selection
- best epoch: `3`
- mini-val score: `176.9170`
- mini-val metrics:
  - `rot_mean_deg = 1.2279`
  - `signed_tdir_mean_deg = 110.9058`
  - `anti_parallel_rate = 0.7305`
  - `tmag_median_ratio = 1.6131`
  - `path_ratio = 0.9807`
- full-val metrics for selected checkpoint:
  - `rot_mean_deg = 1.2757`
  - `signed_tdir_mean_deg = 108.7306`
  - `anti_parallel_rate = 0.7076`
  - `tmag_median_ratio = 1.7021`
  - `path_ratio = 1.0139`
- test used for selection: `false`

## 6. Test result table
| model | rot_mean_deg | signed_tdir_mean_deg | anti_parallel_rate | tmag_median_ratio | path_ratio | coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ABLDVO3_SingleStagePoseRegression | 2.6584 | 51.4096 | 0.2003 | 1.3186 | 1.0248 | 1.0 |
| FINAL360I_full_model | 2.3301 | 45.2647 | 0.2017 | 0.8344 | 0.6404 | 1.0 |

## 7. Full ablation table
| model | rot_mean_deg | signed_tdir_mean_deg | anti_parallel_rate | tmag_median_ratio | path_ratio | coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ABLDVO2_PlainPairVO | 2.4218 | 49.1263 | 0.2068 | 0.8770 | 0.6855 | 1.0 |
| ABLDVO2_NoSphericalGeometry | 2.5361 | 45.1032 | 0.1997 | 0.7811 | 0.6144 | 1.0 |
| ABLDVO2_NoCrossImageInteraction | 3.6062 | 49.7586 | 0.2017 | 0.6230 | 0.4900 | 1.0 |
| ABLDVO3_SingleStagePoseRegression | 2.6584 | 51.4096 | 0.2003 | 1.3186 | 1.0248 | 1.0 |
| FINAL360I_full_model | 2.3301 | 45.2647 | 0.2017 | 0.8344 | 0.6404 | 1.0 |

## 8. Module contribution analysis
- spherical-aware representation:
  - from ABLDVO2, it mainly supports scale/path and mildly helps rotation stability
  - ABLDVO3 does not change that conclusion
- cross-image interaction:
  - still the most stable and strongest contribution from ABLDVO2
  - `NoCrossInteraction` remains much worse than both `FINAL360I` and `ABLDVO3_SingleStage`
- coarse-to-fine residual refinement:
  - `rot`: positive but limited, because `2.6584` vs `2.3301` is a modest gap
  - `signed_tdir`: clearly positive, because `51.4096` is materially worse than `45.2647`
  - `tmag/path`: clearly positive, because single-stage overshoots to `1.3186 / 1.0248`, while FINAL360I stays much closer to realistic scale/path at `0.8344 / 0.6404`
  - overall interpretation: the refinement stage looks more like a scale-and-direction stabilizer than a major rotation breakthrough
- anti-parallel caveat:
  - anti-parallel stays nearly unchanged (`0.2003` vs `0.2017`), so this problem still does not seem to be controlled by the coarse-to-fine block alone

## 9. Thesis-ready interpretation
- thesis-ready paragraph:
  - `The ABLDVO3 single-stage ablation shows that keeping spherical-aware representation and cross-image interaction is not sufficient to recover the full behaviour of FINAL360I. When the coarse-to-fine residual refinement is removed and the model regresses pose in a single stage, rotation degrades slightly, translation direction degrades more clearly, and translation magnitude/path exhibit substantial overshoot. This suggests that the refinement stage provides a meaningful stabilizing effect, especially for translation direction and scale/path behaviour, even though its contribution to rotation accuracy is comparatively limited.`
- limitations statement:
  - `This ablation still uses subset training, one seed, and a compact training budget, so it should be interpreted as strong architectural evidence rather than a fully saturated retraining result.`

## 10. Compliance checklist
- training_executed = true
- test_used_for_selection = false
- existing_metrics_modified = false
- checkpoints_committed = false
- raw_data_committed = false
- explicit_matching_used = false
- ransac_used = false
- pnp_used = false
- ba_used = false
