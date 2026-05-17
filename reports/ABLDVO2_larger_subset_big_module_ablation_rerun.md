# ABLDVO2 larger-subset big module ablation rerun

## 1. Executive summary
- rerun executed true/false: `true`
- models trained: `['ABLDVO2_PlainPairVO', 'ABLDVO2_NoSphericalGeometry', 'ABLDVO2_NoCrossImageInteraction']`
- models skipped: `['ABLDVO2_SingleStagePoseRegression']`
- train subset count: `512`
- mini-val count: `1024`
- classification: `completed`
- main conclusion: `the larger-subset rerun confirms that FINAL360I remains the most balanced pair-level model, and that cross-image interaction is the most critical large module. spherical-aware representation still helps overall, but its larger-subset benefit is concentrated more on scale/path than on signed_tdir.`

## 2. Why ABLDVO2
- `ABLVO361` was useful as a quick-turn ablation, but used only `64 / 12154` train pairs and therefore was too small for a thesis-facing main ablation table.
- `ABLDVO2` keeps the same ablation code skeleton and only changes subset strategy, selection throughput, and output naming.
- naming note: `360` stays reserved for `360DVO / DSET2C` context rather than generic rerun numbering.

## 3. Experimental setup
- manifests:
  - train: `external_baselines/results/dset2c_360dvo_canonical/pair_manifest_train.jsonl`
  - val: `external_baselines/results/dset2c_360dvo_canonical/pair_manifest_val.jsonl`
  - test: `external_baselines/results/dset2c_360dvo_canonical/pair_manifest_test.jsonl`
- train subset strategy: `512` samples, seed `1`, stratified with `uniform_sequence` weighting plus quantile `tmag` / `rot` buckets.
- mini-val strategy: `1024` samples, seed `1`, same stratification policy, used for epoch-by-epoch model selection only.
- training epochs: `3`
- seed: `1`
- optimizer: `AdamW`
- GPU: `NVIDIA GeForce RTX 3060 Laptop GPU`
- selection protocol: `mini-val only during training; full val and full test only once for each selected best checkpoint`

## 4. Subset distribution
- train subset sequence distribution: `bridge_night=87, city_driving=87, canyon_line=86, drone_racetrack=85, field=84, shanghai_street=83`
- train subset tmag buckets: `0:86, 1:92, 2:99, 3:116, 4:119`
- train subset rot buckets: `0:130, 1:106, 2:89, 3:87, 4:100`
- mini-val sequence distribution: `downhill_biking=514, mountains=510`
- mini-val tmag buckets: `0:181, 1:186, 2:194, 3:224, 4:239`
- mini-val rot buckets: `0:208, 1:217, 2:213, 3:199, 4:187`
- caveat: val/test split itself only contains `2` sequences, so sequence diversity on mini-val is fundamentally capped by the canonical split rather than by the sampler.

## 5. Model definitions
- `ABLDVO2_PlainPairVO`: plain pair-level ERP VO baseline with separate image encoding and pooled pair regression, without spherical-aware representation, explicit cross-image interaction, or coarse-to-fine refinement.
- `ABLDVO2_NoSphericalGeometry`: keep cross-image interaction and coarse-to-fine style refinement, but replace spherical-aware positional representation with planar/grid-style encoding.
- `ABLDVO2_NoCrossImageInteraction`: keep spherical-aware encoding, but remove explicit cross-image interaction and regress from simple pair feature composition.
- `FINAL360I_full_model`: current thesis mainline reference.

## 6. Validation selection table
- `ABLDVO2_PlainPairVO`: best epoch `3`, mini-val score `181.4401`, full-val `rot_mean=1.3622`, `signed_tdir_mean=116.5550`, `anti_parallel=0.7611`, `tmag_median_ratio=1.2713`, `path_ratio=0.7317`
- `ABLDVO2_NoSphericalGeometry`: best epoch `1`, mini-val score `183.4652`, full-val `rot_mean=1.2586`, `signed_tdir_mean=115.1620`, `anti_parallel=0.7359`, `tmag_median_ratio=1.4490`, `path_ratio=0.7591`
- `ABLDVO2_NoCrossImageInteraction`: best epoch `1`, mini-val score `185.4239`, full-val `rot_mean=2.0674`, `signed_tdir_mean=113.0487`, `anti_parallel=0.7387`, `tmag_median_ratio=1.5050`, `path_ratio=0.6937`
- test used for selection: `false`

## 7. Test result table
| model | rot_mean_deg | signed_tdir_mean_deg | anti_parallel_rate | tmag_median_ratio | path_ratio | coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ABLDVO2_PlainPairVO | 2.4218 | 49.1263 | 0.2068 | 0.8770 | 0.6855 | 1.0 |
| ABLDVO2_NoSphericalGeometry | 2.5361 | 45.1032 | 0.1997 | 0.7811 | 0.6144 | 1.0 |
| ABLDVO2_NoCrossImageInteraction | 3.6062 | 49.7586 | 0.2017 | 0.6230 | 0.4900 | 1.0 |
| FINAL360I_full_model | 2.3301 | 45.2647 | 0.2017 | 0.8344 | 0.6404 | 1.0 |

## 8. Comparison with ABLVO361
- `PlainPairVO` trend changed the most: `signed_tdir_mean` stayed roughly similar (`48.69 -> 49.13`), but `rot`, `tmag_median_ratio`, and `path_ratio` improved strongly, so the plain baseline is more competitive once trained on a larger subset.
- `NoSphericalGeometry` moved much closer to FINAL360I and even slightly beat it on `signed_tdir_mean` (`47.32 -> 45.10` vs full `45.26`), but still lagged on `rot`, `tmag_median_ratio`, and `path_ratio`.
- `NoCrossImageInteraction` remained the clearest regression relative to the full model and remained the weakest ablation on `rot`, `signed_tdir`, `tmag`, and `path_ratio`.
- overall trend stability: `partial`. The large-subset rerun preserves the strong importance of cross-image interaction and the overall superiority of FINAL360I over a plain pair baseline, but the spherical-aware story becomes more nuanced: it is no longer a clean signed-tdir win, and instead looks mainly like a scale/path stabilizer.

## 9. Module contribution analysis
- spherical-aware representation:
  - `signed_tdir`: contribution is `weak / unstable` in this rerun because `NoSphericalGeometry` is essentially tied with, and slightly better than, FINAL360I on this metric.
  - `rot`: FINAL360I remains slightly better (`2.3301` vs `2.5361`).
  - `tmag/path`: FINAL360I remains clearly better (`0.8344 / 0.6404` vs `0.7811 / 0.6144`), so the stronger evidence is on scale/path stability rather than direction.
- cross-image interaction:
  - removing it causes the most consistent degradation: `rot 3.6062`, `signed_tdir 49.7586`, `tmag 0.6230`, `path 0.4900`.
  - this is the most robust large-module conclusion across both ABLVO361 and ABLDVO2.
- full model vs plain pair VO:
  - FINAL360I still beats the plain baseline on `rot` and `signed_tdir`.
  - the larger-subset plain baseline narrows or even exceeds FINAL360I on `tmag/path`, so the thesis claim should be phrased as `more balanced overall` rather than `dominates every metric`.
- anti-parallel caveat:
  - anti-parallel rates remain close across all models (`0.1997` to `0.2068`), so the anti-parallel issue still does not appear to be solved by any single large structural module.

## 10. Thesis-ready interpretation
- thesis-ready paragraph:
  - `Under the canonical DSET2C pair protocol, the larger-subset ABLDVO2 rerun confirms that FINAL360I remains the most balanced pair-level model relative to a plain ERP pair-VO baseline. The strongest structural contribution comes from explicit cross-image interaction: removing it degrades rotation, translation direction, translation magnitude, and path-ratio simultaneously. In contrast, removing spherical-aware representation does not materially hurt signed translation direction, but still weakens rotation and, more clearly, scale/path behaviour, indicating that spherical-aware modelling contributes mainly to geometric consistency and scale stability rather than serving as the sole source of direction accuracy.`
- limitations statement:
  - `This rerun still uses subset training rather than the full train manifest, only one seed, and does not isolate the single-stage vs coarse-to-fine factor, so it is suitable as a thesis main ablation table with caveats, not as an exhaustive architecture study.`

## 11. Compliance checklist
- training_executed = true
- test_used_for_selection = false
- existing_metrics_modified = false
- checkpoints_committed = false
- raw_data_committed = false
- explicit_matching_used = false
- ransac_used = false
- pnp_used = false
- ba_used = false
