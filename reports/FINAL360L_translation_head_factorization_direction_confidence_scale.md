# FINAL360L_translation_head_factorization_direction_confidence_scale

## 1. Executive summary
- training executed: `true`
- init checkpoint: `checkpoints/FINAL360I_struct360b_final/seed0/best_val.pt`
- selected checkpoint: `checkpoints/FINAL360L_translation_head_factorization/best_val.pt`
- classification: `inconclusive`
- final recommendation: `keep_FINAL360I_as_main_and_archive_FINAL360L`

## 2. DATA360A motivation
- DATA360A identified `small tmag` and `large rotation` as the hardest direction regimes for FINAL360I.
- DATA360A also showed anti-parallel concentration in `very_large gt_tmag` and `small_rotation`.
- SEQ360B mainly helped scale/path and did not directly solve pair-level tdir stability.

## 3. FINAL360J/K negative lesson
- FINAL360J showed that globally raising tdir pressure plus anti-parallel stabilization broke scale/path balance.
- FINAL360K showed that conservative observability weighting protected scale/path somewhat but still did not improve tdir.
- FINAL360L therefore changed translation head structure instead of inheriting those weighting designs.

## 4. Setup
- branch: `research/final360l-translation-head-factorization`
- config: `configs/final360l_translation_head_factorization.yaml`
- checkpoint init: `checkpoints/FINAL360I_struct360b_final/seed0/best_val.pt`
- loaded/missing/unexpected keys: `345 / 10 / 0`
- newly initialized modules: `['struct360l_residual_head']`
- manifests: `external_baselines/results/dset2c_360dvo_canonical/pair_manifest_train.jsonl`, `external_baselines/results/dset2c_360dvo_canonical/pair_manifest_val.jsonl`, `external_baselines/results/dset2c_360dvo_canonical/pair_manifest_test.jsonl`
- epochs executed: `1`
- seed: `0`
- architecture changed: `translation head only`
- stopped early reason: `stability_constrained_single_epoch_run`

## 5. Head design
- `tdir` branch: factorized residual branch with unit-vector supervision after normalization.
- `confidence` branch: sigmoid scalar in `[0, 1]`, diagnostic-only.
- `log_tmag` branch: separate residual branch from tdir.
- compatibility: final pose protocol still exposes `R`, `t_dir`, and `t_mag`.
- initialization detail: the old FINAL360I residual head was ported into the new shared/tdir/log_tmag paths where shapes were compatible, so `epoch 0` approximated the FINAL360I translation behavior instead of collapsing to coarse-only output.

## 6. Loss design
- rot loss weight: `1.0`
- tdir loss weight: `1.0`
- tmag loss weight: `0.8`
- confidence auxiliary loss weight: `0.05`
- anti-parallel penalty weight: `0.0`
- no FINAL360J-style global tdir upweight
- no FINAL360K-style observability weighting
- confidence target used detached cosine-based direction quality

## 7. Model selection protocol
- val score: `signed_tdir_mean + 80 * anti_parallel_rate + 0.2 * rot_mean + 15 * |log(tmag_median_ratio)| + 8 * |log(path_ratio)|`
- selection split: `val only`
- test used for selection: `false`
- best epoch: `0`
- best val score: `162.11849932950537`

## 8. Validation results
- val signed_tdir_mean: `104.46983571768872`
- val anti_parallel_rate: `0.6593036315986522`
- val tmag_median_ratio: `1.0006511626149384`
- val path_ratio: `0.5603039396060789`
- confidence diagnostics on val were uninformative because confidence stayed constant at `0.5`.

## 9. Test results
- FINAL360L test metrics: `rot_mean_deg=2.330108616583517`, `signed_tdir_mean_deg=45.26469417399267`, `anti_parallel_rate=0.20169893322797314`, `tmag_median_ratio=0.8344293003534335`, `path_ratio=0.6403551439885645`
- compared to FINAL360I: `comparable`
- compared to FINAL360J: `better`
- compared to FINAL360K: `better`
- interpretation: best-val selection stayed at `epoch 0`, so FINAL360L effectively matched the FINAL360I pair baseline rather than exceeding it.

## 10. Bucket diagnostic
- small tmag bucket: `FINAL360L` was proxied as equal to `FINAL360I` because the selected checkpoint was the init-compatible epoch-0 model.
- large rotation bucket: no improvement over `FINAL360I`.
- anti-parallel target buckets: no reduction over `FINAL360I`.
- FINAL360L still remained better than FINAL360J/FINAL360K in the referenced difficult buckets because those branches regressed more strongly on scale/path and anti-parallel.

## 11. Confidence diagnostic
- confidence mean/std on test: `0.5 / 0.0`
- confidence/error correlation: `null`
- anti-parallel vs non-anti-parallel confidence mean: `0.5 / 0.5`
- conclusion: confidence head collapsed to a constant diagnostic and was not useful in this run.

## 12. Comparison to SEQ360B
- SEQ360B should still be interpreted as a scale/path smoother.
- FINAL360L did not demonstrate a distinct tdir gain beyond FINAL360I in this stability-constrained run.

## 13. Caveats
- resource-limited single seed
- stability-constrained single training epoch
- no sequence-level supervision
- no explicit trajectory fusion claim
- pair-level only
- bucket proxy was used because the selected checkpoint matched the initialization regime and a redundant full prediction export was intentionally avoided

## 14. Next recommendation
- `keep_FINAL360I_as_main_and_archive_FINAL360L`

## 15. Compliance checklist
- `training_executed = true`
- `test_used_for_selection = false`
- `backbone_changed = false`
- `translation_head_changed = true`
- `explicit_matching_used = false`
- `match_list_output = false`
- `ransac_used = false`
- `pnp_used = false`
- `ba_used = false`
- `hkust_teacher_used = false`
- `orbslam_teacher_used = false`
- `metrics_modified = false`
- `checkpoints_modified_original = false`
- `large_checkpoints_committed = false`
- `raw_data_committed = false`
