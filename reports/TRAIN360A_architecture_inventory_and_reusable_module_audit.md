# TRAIN360A architecture inventory and reusable module audit

## 1. Executive summary

- Recommend `TRAIN360A` first, then `TRAIN360_spherical_pose_baseline`.
- Recommended `TRAIN360-v0` body: `hybrid`:
  T57b-style `PanoramaRelPoseModel` pair-forward skeleton plus selected S5-era transferable ideas, not legacy S5 export artifacts.
- Largest migration risk:
  reusing legacy scene01/seq01/seq02/seq03 tooling or policy objects as if they were general forward modules. The current repo has many useful submodules, but most S5E-series artifacts after S5E6 are export/refinement/diagnostic layers built around `scene01/seq03` traceable-dense products, not arbitrary ERP image-pair inference.
- Recommendation headline:
  use DSET2C canonical manifest as the only training/eval pair source; keep T57b's ERP tokenization and coarse/fine pair model skeleton; add manifest-native dataloader, bounded `log_tmag`, observability-aware weighting, and optional geometry-token refinement later.

## 2. Repository and artifact inventory

### Git state

| item | value |
|---|---|
| git status | clean (`git status --short` returned empty) |
| current branch | `experiment/dset2c-360dvo-dataset-hygiene` |
| tag at HEAD | `dset2c-canonical-train360-ready` |
| related tags found | `dset2b-train360-ready`, `dset2c-canonical-train360-ready`, `final-s5-clean-candidate-post-s9`, `s5e12-feature-gate-passed` |

### High-value artifacts

| artifact | path | purpose | risk |
|---|---|---|---|
| T57b checkpoint | `checkpoints/T57b_no_dt_multiscale_tmag_head_400/final.pt` | Only confirmed arbitrary ERP pair-forward checkpoint in repo | medium |
| T57b training summary | `checkpoints/T57b_no_dt_multiscale_tmag_head_400/final_summary.json` | Exposes selected loss/head knobs and legacy anchor usage | medium |
| T57b external export bridge | `tools/export_gen5_360dvo_t57b_predictions.py` | Confirms direct manifest-driven forward inference | low |
| T57b external eval report | `checkpoints/GEN5_360DVO_T57b_true_external_eval.json` | External generalization evidence, not training dependency | low |
| Main pair model | `model.py`, `interaction.py`, `losses.py` | Reusable ERP pair model and losses | low |
| Current training loader path | `dataset_pano_only.py`, `train_mvp.py` | Current loader stack still rooted in legacy directory scan | high |
| DSET2C hygiene checkpoint | `checkpoints/DSET2C_360DVO_dataset_hygiene.json` | Canonical split readiness and enforcement policy | low |
| DSET2C manifests | `external_baselines/results/dset2c_360dvo_canonical/pair_manifest_{train,val,test}.jsonl` | Required future pair interface | low |
| S5 clean frozen policy | `checkpoints/S5_clean_tmag_calibration_policy.json` | Inference-time legacy calibration policy on top of T57b | high |
| S5E12 feature gate | `configs/s5e12_real_correspondence_feature_gate.yaml`, `tools/s5e12_observability_gate.py` | Transferable observability-gating idea | medium |
| S5E15 candidate | `checkpoints/S5E15_scale_deunderfit_antiparallel_candidate.json`, `tools/export_s5e15_adjacent_dense_predictions.py` | Legacy internal best candidate, but export/refinement only | high |
| S5E19/S5E20 configs | `configs/s5e19_rotation_compensated_multiframe_geometry.yaml`, `configs/s5e20_true_kstep_composition_supervision.yaml` | Transferable architectural ideas | medium |

## 3. T57b architecture audit

### Model definition location

- Model class: `PanoramaRelPoseModel` in [model.py](/home/dovetao/graduation_design_demo/model.py:739)
- ERP token sampler/backbone: `Module2Sampler` in [model.py](/home/dovetao/graduation_design_demo/model.py:51)
- Coarse/fine interaction and heads: [interaction.py](/home/dovetao/graduation_design_demo/interaction.py:682), [interaction.py](/home/dovetao/graduation_design_demo/interaction.py:871)
- Losses: [losses.py](/home/dovetao/graduation_design_demo/losses.py:55), [losses.py](/home/dovetao/graduation_design_demo/losses.py:122), [losses.py](/home/dovetao/graduation_design_demo/losses.py:216)

### Checkpoint state_dict key summary

- `final.pt` is a zip-format PyTorch checkpoint and contains `model` and `cfg` payload entries.
- State key families found by read-only pickle inspection:
  `module2.grid_*`, `module2.hier.*`, `module2.patch_embed_c/*`, `module2.patch_embed_f/*`, `module2.patch_embed_c_t/*`, `module2.enc_c/*`, `module2.enc_f/*`, `coarse.*`, `fine.*`, `log_tmag_bias`.
- This is a real model checkpoint, not just an export policy or post-hoc refinement package.

### Input/output interface

- Forward signature: `forward(IA, IB, enable_depth_fusion=None, dt_world=None)` in [model.py](/home/dovetao/graduation_design_demo/model.py:847)
- Returns:
  `R_final`, translation direction tensor, and `aux`.
- Final pose outputs are standardized through `_set_transform_outputs` in [model.py](/home/dovetao/graduation_design_demo/model.py:287).
- Export bridge reads:
  `R_pred_BA`, `tdir_pred_B`, `tmag_pred`, `tvec_pred_B` from the forward path in [tools/export_gen5_360dvo_t57b_predictions.py](/home/dovetao/graduation_design_demo/tools/export_gen5_360dvo_t57b_predictions.py:177).

### Is T57b a true arbitrary ERP image-pair -> R / tdir / tmag model?

- Yes.
- Evidence:
  it is loaded directly as `PanoramaRelPoseModel` from checkpoint and run on canonical manifest image pairs in [tools/export_gen5_360dvo_t57b_predictions.py](/home/dovetao/graduation_design_demo/tools/export_gen5_360dvo_t57b_predictions.py:44).
- It does not require legacy `edge_provenance.jsonl`, `scene01/seq03` correspondence CSVs, or dense replay artifacts to produce a pair prediction.

### Backbone / multiscale / heads / losses

- Backbone:
  ERP patch sampling over spherical grids plus token encoders (`PatchEmbed`, `BearingPosEnc`, `TokenEncoder`) via `Module2Sampler`.
- ERP-aware processing:
  native spherical bearing grids and ERP patch sampling in [model.py](/home/dovetao/graduation_design_demo/model.py:55) and [model.py](/home/dovetao/graduation_design_demo/model.py:123).
- Feature interaction:
  coarse soft correspondence matrix `Wc_ab` and fine routed interaction in [interaction.py](/home/dovetao/graduation_design_demo/interaction.py:781) and [interaction.py](/home/dovetao/graduation_design_demo/interaction.py:610).
- Rotation head:
  `CoarsePoseHead.rot` and `FinePoseHead.rot`.
- Translation direction head:
  `CoarsePoseHead.t`, translation-only coarse branch `TranslationOnlyHead`, and fine-stage `t_head`.
- Translation magnitude head:
  `TranslationMagnitudeHead` with `multiscale` mode support in [interaction.py](/home/dovetao/graduation_design_demo/interaction.py:313).
- Bounded scale stabilization:
  `positive_translation_magnitude` plus `_apply_log_tmag_bias` in [interaction.py](/home/dovetao/graduation_design_demo/interaction.py:557) and [model.py](/home/dovetao/graduation_design_demo/model.py:260).
- Losses present:
  pose loss, direction loss, log-space tmag loss, speed/ratio/chain consistency, coupled residual regularization, epipolar matching losses.

### Reusable modules

- Directly reusable:
  `Module2Sampler`, `PatchEmbed`, `BearingPosEnc`, `TokenEncoder`, `CoarseInteraction`, `TranslationMagnitudeHead(multiscale)`, `positive_translation_magnitude`, `pose_loss`, `translation_direction_loss`, `translation_magnitude_loss`.
- Reusable with adaptation:
  `FineInteraction`, `CoupledPoseResidualHead`, depth fusion hooks, dt-conditioned scale hooks, evaluation export bridge.
- Not recommended for `TRAIN360-v0`:
  legacy `tdir_anchor_checkpoint` dependency from `final_summary.json`, S5 clean legacy calibration policy, any legacy scene-based train/test split assumptions.

### Why T57b is weak on 360DVO translation generalization

- External metrics are poor on translation despite acceptable rotation:
  overall external `tdir_abs_mean_deg=55.26`, `anti_parallel_rate=0.675`, `path_ratio=0.149` in `external_baselines/results/gen5_360dvo_t57b_external_eval/edge_component_metrics.json`.
- Legacy bucket metrics already show k=1 scale/direction fragility:
  `k=1 tdir_abs=35.60`, `tmag_rel_err=1.50`; `k=2 tmag_rel_err=1.74`; `k=3 tmag_rel_err=1.07` in `eval_buckets_latest.json`.
- Likely architectural causes:
  coarse matching is strong for rotation but still too appearance-driven for weakly observable translation.
  multiscale `tmag` head is global-pooled and not explicitly geometry-token aware.
  no manifest-native observability weighting by default.
  no explicit adjacent+k-step composition supervision in the T57b forward baseline.
  legacy anchor loss path (`tdir_anchor_checkpoint`) suggests training-time reliance on a legacy direction prior that should not be carried into TRAIN360.

## 4. S5clean / S5E15 / S5E series audit

### S5clean

- `S5_clean_tmag_calibration_policy.json` is a frozen inference-time policy, not a model.
- It depends on:
  base checkpoint `checkpoints/T57b_no_dt_multiscale_tmag_head_400/final.pt`
  train-split quantiles of legacy `pred_tmag`
  S2b policy factors.
- Transfer verdict:
  concept reusable, exact policy not reusable.
- Why not direct:
  thresholds are legacy train-CV calibration outputs, not architecture weights.

### S5E12 feature gate

- Main role:
  derives correspondence observability buckets and signed-direction reliability.
- Evidence:
  config hardcodes `scene01/seq03` timestamps, GT, camera yaml, and output CSV/JSON paths in [configs/s5e12_real_correspondence_feature_gate.yaml](/home/dovetao/graduation_design_demo/configs/s5e12_real_correspondence_feature_gate.yaml:1).
- Transfer verdict:
  gate logic reusable with adaptation; produced artifacts not reusable.
- Reusable part:
  observability bucket idea and weighting policy.
- Non-reusable part:
  ORB/BF/flow/essential feature dumps and scene01-specific CSVs.

### S5E15 candidate

- Main role:
  post-hoc refinement over S5E14 traceable-dense outputs, not a standalone pair-forward model.
- Hard evidence:
  [tools/export_s5e15_adjacent_dense_predictions.py](/home/dovetao/graduation_design_demo/tools/export_s5e15_adjacent_dense_predictions.py:86) reads `external_baselines/results/s5e14_traceable_dense/edge_provenance.jsonl`.
  it also reads `external_baselines/results/s5e14_traceable_dense/correspondence_refinement_weights.jsonl` and `s5e12_feature_dir/correspondence_features.csv`.
  config is explicitly `scene01`, `seq03`, timestamps, GT, and legacy result dirs in [configs/s5e15_scale_deunderfit_antiparallel.yaml](/home/dovetao/graduation_design_demo/configs/s5e15_scale_deunderfit_antiparallel.yaml:1).
  candidate metadata classifies it as `S5E15_TRAIN_PRIOR_SCALE_ONLY` with `supervised_train_refinement_available=false`.
- Transfer verdict:
  exact artifact `no`.
  anti-parallel guard idea and bounded residual idea `adapt`.

### Why S5E15 cannot be used as a 360DVO external inference model

- The dedicated audit script marks:
  `image_pair_forward_available = false`
  `external_erp_input_supported = false`
  `depends_on_scene01_artifacts = true`
  bridge classification resolves to `SCENE_SPECIFIC_EXPORT_ONLY` or `IMAGE_PAIR_FORWARD_MISSING` in [tools/audit_gen3_s5e15_inference_availability.py](/home/dovetao/graduation_design_demo/tools/audit_gen3_s5e15_inference_availability.py:42).
- Its export path consumes prior provenance and train-prior scale factors instead of taking arbitrary ERP pairs as input.
- Therefore S5E15 must not be labeled as a reusable external image-pair forward model.

### S5E6 / S5E19 / S5E20 / S5E18 / STRUCT1-family ideas

- `S5E6`:
  bounded log-magnitude and scale guard are conceptually useful, but config is legacy split specific.
- `S5E19`:
  rotation compensation, multiframe geometry, observability weighting, bounded fine scale are architecturally aligned with TRAIN360; config still binds to `scene01/seq01-03`.
- `S5E20`:
  true contiguous k-step composition supervision is valuable, but current artifact is legacy training design.
- `S5E18`:
  ERP/spherical bearing-flow idea is aligned with the 360DVO direction problem, but current implementation is a surrogate extractor over `s5e12` aggregated features, not a direct forward module.
- `STRUCT1 / ARCH2` classes in `model.py`:
  `SphericalGeometryTokenBackbone`, `SoftCorrespondenceGeometryLayer`, `GeometryTokenPoseSolver` are promising reusable code-level modules and do not appear inherently bound to scene01 artifacts.

## 5. Reusable module matrix

See `reports/TRAIN360A_reusable_module_matrix.json`.

## 6. Non-transferable artifact list

| artifact | path | reason_not_transferable | risk_if_used | replacement_for_TRAIN360 |
|---|---|---|---|---|
| S5 clean frozen calibration policy | `checkpoints/S5_clean_tmag_calibration_policy.json` | legacy train-CV threshold policy, not model weights | leaks legacy operating regime into new baseline | retrain-free idea only; re-estimate later on TRAIN360 train split if needed |
| S5E15 export pipeline | `tools/export_s5e15_adjacent_dense_predictions.py` | depends on S5E14 provenance and S5E12 feature CSV | false belief that S5E15 is a forward model | build direct pair-forward TRAIN360 model |
| S5E12 feature dumps | `external_baselines/results/s5e12_feature_gate/*` | scene01/seq03 diagnostic artifacts | introduces hidden legacy runtime dependency | manifest-native observability module computed online or from train split only |
| Legacy traceable-dense provenance | `external_baselines/results/s5e14_traceable_dense/edge_provenance.jsonl` and siblings | scene-specific export-only intermediate | impossible to generalize to DSET2C canonical pairs | none; remove from runtime |
| Legacy train/test scene split loader | `dataset_pano_only.py` raw scan path | scans `scene*/seq*` and derives split internally | violates canonical-manifest-only requirement | new manifest dataset adapter |
| T57b legacy anchor checkpoint | `checkpoints/C31_coarse_gtmatch_tmag_detach_workers6_800/best_joint_local_A_abs.pt` | legacy teacher-like anchor for tdir loss | imports legacy distribution bias into TRAIN360 | no anchor at v0 |
| Direct glob training entrance | `data/360DVO/Sequences/*` | explicitly forbidden by task and bypasses canonical split | split leakage / hygiene regression | `pair_manifest_{train,val,test}.jsonl` only |

## 7. DSET2C canonical manifest interface audit

### Manifest field structure

Current `pair_manifest_{train,val,test}.jsonl` rows expose:

- `dataset`
- `seq_id`
- `split`
- `pair_index`
- `pair_type`
- `k`
- `image_path_a`
- `image_path_b`
- `timestamp_a`
- `timestamp_b`
- `R_BA`
- `t_BA_B`
- `tdir_B`
- `tmag`
- `valid_pose`
- `valid_timestamp`

### Train/val/test split check

- Train sequences: `canyon_line`, `shanghai_street`, `bridge_night`, `drone_racetrack`, `field`, `city_driving`
- Val sequences: `mountains`, `downhill_biking`
- Test sequences: `snowmobile`, `ridge_to_lake`
- Summary explicitly records:
  `split_by_sequence=true`
  `forbid_random_pair_split=true`
  `manifest_ready=true`
  in `canonical_manifest_summary.json` and `DSET2C_360DVO_dataset_hygiene.json`.

### Dataloader requirements

For TRAIN360 pair training, required fields are already present:

- image paths: `image_path_a`, `image_path_b`
- relative pose: `R_BA`, `t_BA_B`, `tdir_B`, `tmag`
- sequence id: `seq_id`
- pair distance / k-step: `k`, `pair_type`
- split: `split`
- temporal metadata for diagnostics: `timestamp_a`, `timestamp_b`

### Existing dataloader compatibility

- Current train/eval dataloaders do not directly read these jsonl manifests.
- `train_mvp.py` still defaults to `RflyPanoPanoramaPairsMixedK` / `RflyPanoPanoramaPairsEvalFixedKList` and scans `data_root` with internal split logic in [train_mvp.py](/home/dovetao/graduation_design_demo/train_mvp.py:3566).
- `RflyPanoPanoramaPairsTrainFixedList` does accept a manifest, but the format is a legacy JSON payload with `scene`, `seq`, `i`, `j`, `k`, not canonical jsonl rows.

### Minimal dataloader adapter recommendation

- Add a new manifest-native dataset class, for example `Dset2CCanonicalPairDataset`.
- Input:
  one jsonl path per split.
- Output batch keys:
  `IA`, `IB`, `R_gt`, `t_gt_vec`, `t_gt_dir`, `t_gt_mag`, `meta`.
- `meta` should include:
  `seq_id`, `split`, `pair_index`, `pair_type`, `k`, `timestamp_a`, `timestamp_b`, `dt_world`.
- Do not infer train/test split in code.
  The file path selected by config is the split contract.
- Do not require `scene`/`seq` or frame indices.
  Those are absent from canonical rows and should not be reconstructed unless strictly needed for debugging.

### Random pair split risk

- Canonical manifests themselves are safe.
- The risk comes from reusing loaders that call `_partition_seq_keys(...)` or scan raw directories.
- Enforcement rule for TRAIN360:
  if `dataset_manifest_train/val/test` is set, bypass all `split_by`, `train_ratio`, `split_seed`, raw sequence partitioning, and random mixed-k pair generation.

### Additional interface caveat

- Current canonical rows do not contain `T_w_a` / `T_w_b`.
- `tools/evaluate_gen5_360dvo_t57b_external_eval.py` currently expects `T_w_a` for trajectory reconstruction in [tools/evaluate_gen5_360dvo_t57b_external_eval.py](/home/dovetao/graduation_design_demo/tools/evaluate_gen5_360dvo_t57b_external_eval.py:61).
- So trajectory evaluators should either:
  consume pair-level chaining only, or use a manifest variant enriched with world-pose rows. This is an evaluator-compatibility issue, not a training blocker.

## 8. TRAIN360-v0 architecture recommendation

### Backbone

- Use T57b backbone family:
  `Module2Sampler` + `PatchEmbed` + `BearingPosEnc` + token encoders.
- Keep coarse and fine token levels.
- Do not inherit legacy `tdir_anchor_checkpoint`.

### ERP-aware / spherical feature processing

- Keep ERP patch sampling on spherical grids as the v0 default.
- Add light geometry channels only if already available from `SphericalGeometryTokenBackbone`; otherwise keep T57b token stack first and defer full geometry-token backbone to later.

### Feature interaction

- Keep T57b coarse soft correspondence interaction as the main stem.
- Keep fine routed interaction, but make it optional in config.
- Add observability-weighted token pooling or confidence-aware weighting inspired by S5E12/S5E19.

### Heads for `R_BA / tdir_B / tmag`

- `R_BA`:
  keep coarse pose head plus optional fine residual rotation.
- `tdir_B`:
  keep translation-specific branch, but upgrade pooling to confidence/observability-aware pooling.
- `tmag`:
  keep bounded `log_tmag` head.
  start from T57b multiscale head or equivalent bounded log head, not S5 clean calibration policy.

### Scale stabilization

- Keep bounded `log_tmag` with clamp and positive reconstruction.
- Allow learnable global `log_tmag_bias` only if trained inside TRAIN360 and not inherited from S5 policy.
- Do not import train-prior scale factors from S5E15.

### Adjacent + k-step integration

- Train/eval pair source should come directly from canonical manifest rows with `k in {1,2,3,5}`.
- Support both adjacent and k-step rows in the same dataset.
- Add optional contiguous-window composition loss later, but do not require legacy trajectory export artifacts.

### Loss design: reuse vs drop

- Reuse:
  `pose_loss`
  `translation_direction_loss`
  `translation_magnitude_loss`
  optional pairwise ratio / chain-sum losses
  observability-weighted sampling or weighting idea.
- Use cautiously:
  epipolar / geometry matching losses if they operate only on train pairs and predicted correspondences.
- Do not use in v0:
  S5 clean frozen calibration policy
  S5E15 train-prior scale multiplication
  legacy anchor checkpoint losses
  ORB-SLAM3 or HKUST teacher terms
  scene01 traceable-dense artifacts

### Components to defer to `STRUCT360 / ARCH360`

- full `SphericalGeometryTokenBackbone`
- `SoftCorrespondenceGeometryLayer`
- `GeometryTokenPoseSolver`
- strong multiframe rotation-compensated geometry stack
- surrogate S5E18 bearing-flow modules based on feature dumps

## 9. Minimal implementation plan for next task

### Suggested file additions / modifications

- Add:
  `dataset_dset2c_manifest.py`
  `configs/train360_spherical_pose_baseline.yaml`
  `tools/evaluate_train360_pair_manifest.py`
- Modify later:
  `train_mvp.py` to allow manifest-native dataset selection
  optionally `config.py` for manifest paths.

### Dataloader adapter

- New class reads canonical jsonl directly.
- No raw split generation.
- Validate every row:
  `split` matches requested split
  `valid_pose` and `valid_timestamp` are true
  paths exist
  `k` belongs to allowed set.

### Model class

- Start with `PanoramaRelPoseModel` as the base.
- Add config switches to disable legacy-only knobs:
  anchor loss, legacy calibration policy, eval-only helpers.
- Keep optional fine stage and optional coupled residual head off by default for the very first run.

### Loss

- Start with:
  rotation geodesic
  signed `tdir`
  bounded log-`tmag`
  optional modest observability weighting from manifest-side or predicted confidence.
- Leave k-step composition as a disabled but implemented option for the next iteration.

### Evaluator

- Pair evaluator should operate on canonical val/test manifests only.
- Report:
  `rot`
  `signed_tdir`
  `tdir_abs`
  `anti_parallel_rate`
  `tmag_median_ratio`
  `tmag_p90/p95`
  `path_ratio` if adjacent chaining is available.

### Config

- Require:
  `train_manifest`
  `val_manifest`
  `test_manifest`
- Forbid:
  implicit `data_root` splitting when those are set.

### Sanity checks

- Assert no overlap between `seq_id` sets across the three manifests.
- Assert `split` field matches file identity.
- Assert no training code path reads `scene*` / `seq*` from raw directory when manifest mode is active.
- Assert outputs are `R_BA`, `tdir_B`, `tmag`.

### Expected first-run metrics

- Use these as acceptance targets, not predictions:
  better external translation than T57b overall
  `tdir_abs_mean` materially below `55.26`
  `anti_parallel_rate` materially below `0.675`
  `path_ratio` clearly closer to `1.0` than `0.149`
  no collapse of rotation quality relative to T57b external `rot_mean_deg=2.12`.

## 10. Compliance checklist

- `no_training_executed = true`
- `no_finetune_executed = true`
- `learned_weights_saved = false`
- `s5_locked_metrics_modified = false`
- `uses_eval_gt_for_training = false`
- `uses_test_gt_for_training = false`
- `uses_orbslam3_teacher = false`
- `uses_hkust_360dvo_teacher = false`
- `no_legacy_scene01_artifact_dependency_in_recommendation = true`
- `dset2c_canonical_manifest_is_required = true`
