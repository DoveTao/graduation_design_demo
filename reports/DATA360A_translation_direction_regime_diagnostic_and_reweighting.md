# DATA360A translation direction regime diagnostic and reweighting
## 1. Executive summary
- diagnostic executed: `true`
- training executed: `false`
- checkpoint modified: `false`
- model structure changed: `false`
- splits analyzed: `val`, `test`
- models analyzed: `FINAL360I, SEQ360B`
- prediction dump generated: `true`
- main conclusion: The dominant failure mode is broad pair-level translation-direction instability, so direct tdir loss reweighting should be tried before more complex sequence-level changes.
## 2. Data sources
- canonical manifests:
  - `/home/dovetao/graduation_design_demo/external_baselines/results/dset2c_360dvo_canonical/pair_manifest_val.jsonl`
  - `/home/dovetao/graduation_design_demo/external_baselines/results/dset2c_360dvo_canonical/pair_manifest_test.jsonl`
- checkpoints:
  - `/home/dovetao/graduation_design_demo/checkpoints/FINAL360I_struct360b_final/seed0/best_val.pt`
  - `/home/dovetao/graduation_design_demo/checkpoints/SEQ360B_lightweight_scale_smoothing/best_val.pt`
- metric JSON / report files:
  - `/home/dovetao/graduation_design_demo/reports/FINAL360I_metrics_test.json`
  - `/home/dovetao/graduation_design_demo/reports/TRAIN360E_metrics_test.json`
  - `/home/dovetao/graduation_design_demo/reports/SEQ360B_metrics_test.json`
  - `/home/dovetao/graduation_design_demo/reports/SEQ360B_trajectory_metrics_test.json`
  - `/home/dovetao/graduation_design_demo/reports/BASE360D_component_metric_alignment.md`
  - `/home/dovetao/graduation_design_demo/CURRENT_MAINLINE.md`
  - `/home/dovetao/graduation_design_demo/DELIVERY_HANDOFF.md`
  - `/home/dovetao/graduation_design_demo/reports/RELEASE3_delivery_handoff.md`
  - `/home/dovetao/graduation_design_demo/thesis/THESIS361_claims_and_caveats_checklist.md`
## 3. Overall metrics recap
- FINAL360I pair-level: signed_tdir_mean=`45.2647`, anti_parallel=`0.2017`, tmag_median_ratio=`0.8344`, path_ratio=`0.6404`
- TRAIN360E trajectory: ATE none / SE3 / Sim3=`222.5686` / `118.6038` / `27.5647`, path_ratio=`1.7563`
- SEQ360B pair-level: signed_tdir_mean=`45.3254`, anti_parallel=`0.2002`, tmag_median_ratio=`1.6978`, path_ratio=`1.3505`
- SEQ360B trajectory: ATE Sim3=`27.5672`, trajectory_path_ratio=`1.3502`
- BASE360D caveat: component metrics are trajectory-derived and are not identical to native pair-level predictions.
## 4. tmag bucket analysis
The following regime tables pool `val + test` samples for diagnostic density unless otherwise noted.
| bucket | count | signed_tdir_mean | anti_parallel | rot_mean | tmag_median_ratio | path_ratio |
| --- | --- | --- | --- | --- | --- | --- |
| very_small | 1041 | 73.9103 | 0.2824 | 1.5450 | 4.0547 | 3.3367 |
| small | 1560 | 86.0150 | 0.4795 | 1.4707 | 1.7570 | 2.0534 |
| medium | 2601 | 71.2172 | 0.4041 | 1.4964 | 1.2206 | 1.2903 |
| large | 2601 | 68.2881 | 0.4241 | 1.9225 | 0.7476 | 0.7489 |
| very_large | 2601 | 81.9803 | 0.5179 | 2.2896 | 0.3146 | 0.2829 |
Conclusion: the worst gt_tmag regime by signed_tdir_mean is `small`.
## 5. rotation bucket analysis
| bucket | count | signed_tdir_mean | anti_parallel | rot_mean | tmag_median_ratio |
| --- | --- | --- | --- | --- | --- |
| small_rotation | 10267 | 75.5098 | 0.4361 | 1.4810 | 0.9221 |
| medium_rotation | 129 | 86.9072 | 0.4884 | 24.4583 | 0.6784 |
| large_rotation | 8 | 92.2461 | 0.3750 | 48.7880 | 0.3830 |
| extreme_rotation | 0 | N/A | N/A | N/A | N/A |
Conclusion: the worst gt_rot_angle regime by signed_tdir_mean is `large_rotation`.
## 6. tmag × rotation regime analysis
| tmag_bucket | rotation_bucket | count | signed_tdir_mean | anti_parallel | tmag_median_ratio |
| --- | --- | --- | --- | --- | --- |
| very_small | small_rotation | 1041 | 73.9103 | 0.2824 | 4.0547 |
| very_small | medium_rotation | 0 | N/A | N/A | N/A |
| very_small | large_rotation | 0 | N/A | N/A | N/A |
| very_small | extreme_rotation | 0 | N/A | N/A | N/A |
| small | small_rotation | 1560 | 86.0150 | 0.4795 | 1.7570 |
| small | medium_rotation | 0 | N/A | N/A | N/A |
| small | large_rotation | 0 | N/A | N/A | N/A |
| small | extreme_rotation | 0 | N/A | N/A | N/A |
| medium | small_rotation | 2588 | 71.1837 | 0.4042 | 1.2197 |
| medium | medium_rotation | 13 | 77.8858 | 0.3846 | 1.2435 |
| medium | large_rotation | 0 | N/A | N/A | N/A |
| medium | extreme_rotation | 0 | N/A | N/A | N/A |
| large | small_rotation | 2517 | 67.5346 | 0.4203 | 0.7488 |
| large | medium_rotation | 84 | 90.8653 | 0.5357 | 0.7042 |
| large | large_rotation | 0 | N/A | N/A | N/A |
| large | extreme_rotation | 0 | N/A | N/A | N/A |
| very_large | small_rotation | 2561 | 81.9707 | 0.5197 | 0.3140 |
| very_large | medium_rotation | 32 | 80.1819 | 0.4062 | 0.3350 |
| very_large | large_rotation | 8 | 92.2461 | 0.3750 | 0.3830 |
| very_large | extreme_rotation | 0 | N/A | N/A | N/A |
## 7. scene / sequence analysis
- these rankings are also pooled over `val + test` to expose stable bad regimes rather than a single-split artifact.
- worst scenes by signed_tdir_mean:
  - `downhill_biking`: signed_tdir_mean=`124.7620`, anti_parallel=`0.9577`, count=`2293`
  - `mountains`: signed_tdir_mean=`89.2092`, anti_parallel=`0.4349`, count=`3049`
  - `ridge_to_lake`: signed_tdir_mean=`85.6349`, anti_parallel=`0.4647`, count=`2197`
  - `snowmobile`: signed_tdir_mean=`14.3072`, anti_parallel=`0.0000`, count=`2865`
- worst sequences by anti_parallel_rate:
  - `downhill_biking`: anti_parallel=`0.9577`, signed_tdir_mean=`124.7620`, count=`2293`
  - `ridge_to_lake`: anti_parallel=`0.4647`, signed_tdir_mean=`85.6349`, count=`2197`
  - `mountains`: anti_parallel=`0.4349`, signed_tdir_mean=`89.2092`, count=`3049`
  - `snowmobile`: anti_parallel=`0.0000`, signed_tdir_mean=`14.3072`, count=`2865`
## 8. anti-parallel analysis
- anti_parallel count: `4543` / `10404`
- strongest tmag concentration: `very_large`
- strongest rotation concentration: `small_rotation`
- collapse share within anti_parallel: `0.0000`
- explosion share within anti_parallel: `0.0000`
## 9. FINAL360I vs SEQ360B comparison
- FINAL360I adjacent-only: signed_tdir_mean=`75.7202`, anti_parallel=`0.4360`, tmag_median_ratio=`2.3556`, path_ratio=`1.6612`
- SEQ360B adjacent-only: signed_tdir_mean=`75.7202`, anti_parallel=`0.4360`, tmag_median_ratio=`1.8107`, path_ratio=`1.2763`
- Interpretation: SEQ360B improves scale/path only = `true`
## 10. Reweighting recommendation
- Conservative:
  - Lightly down-weight very_small gt_tmag pairs in the tdir loss.
  - Keep full supervision on medium/large gt_tmag pairs.
  - Add a small anti-parallel penalty before changing the translation head.
- Aggressive:
  - Strongly down-weight very_small gt_tmag plus extreme_rotation pairs for tdir supervision.
  - Up-weight medium/large gt_tmag pairs with non-extreme rotation for tdir loss.
  - Consider explicit observability weighting or a lightweight tdir confidence branch.
- Diagnostic filtering:
  - Do not filter by default; use filtering only if a tiny subset is clearly geometry-unobservable or corrupted.
  - If filtering is used, report the exact scene/sequence subset and proportion removed.
- Recommended next experiment: `proceed_to_FINAL360J_loss_reweight`
## 11. Final recommendation
- `proceed_to_FINAL360J_loss_reweight`
## 12. Compliance checklist
- `training_executed = false`
- `fine_tune_executed = false`
- `model_structure_changed = false`
- `metrics_modified = false`
- `checkpoints_modified = false`
- `raw_data_modified = false`
- `explicit_matching_used = false`
- `ransac_used = false`
- `pnp_used = false`
- `ba_used = false`
- `large_prediction_dump_committed = false`
