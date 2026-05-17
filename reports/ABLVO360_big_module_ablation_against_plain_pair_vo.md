# ABLVO360 big module ablation against plain pair VO

## 1. Executive summary
- ablation executed true/false: `false`
- models trained: `[]`
- models skipped: `['ABLVO360_PlainPairVO', 'ABLVO360_NoSphericalGeometry', 'ABLVO360_NoCrossImageInteraction', 'ABLVO360_SingleStagePoseRegression']`
- selected full baseline: `FINAL360I_struct360b_final_selected`
- classification: `blocked`
- main conclusion: `code path for A/B/C/D ablations is implemented and smoke-tested, but full retraining was blocked by runtime cost of repeated full-manifest val/test evaluation on current hardware budget.`

## 2. Blocking reason
- current canonical split sizes at `384x768`: `train=12154`, `val=5342`, `test=5062`
- the training harness reached validation twice, confirming the code path is live, but full-manifest validation became the dominant runtime bottleneck
- current GPU: `RTX 3060 Laptop 6GB`; current free disk during run preparation was near the configured floor
- this task requires full val selection and final test evaluation for at least A/B/C; a quick partial run with reduced train subset would still spend most wall time in val/test and would not finish cleanly within the current execution window

## 3. What was completed
- branch created from `main`: `research/ablvo360-big-module-ablation-against-plain-pair-vo`
- canonical manifests existence checked: `pass`
- FINAL360I reports existence checked: `pass`
- CUDA smoke checked: `pass`
- current mainline artifact checks: `pass`
- new files added:
  - `configs/ablvo360_big_module_ablation.yaml`
  - `models/ablvo360_big_module_ablation.py`
  - `tools/train_ablvo360_big_module_ablation.py`
- import / py_compile smoke: `pass`
- per-variant forward smoke on real manifest batch:
  - `ABLVO360_PlainPairVO`: `pass`
  - `ABLVO360_NoSphericalGeometry`: `pass`
  - `ABLVO360_NoCrossImageInteraction`: `pass`
  - `ABLVO360_SingleStagePoseRegression`: `pass`

## 4. Recommended recovery
- reduce evaluation frequency so epoch logging does not require a full val pass every epoch
- keep final protocol strict: `train split train`, `val split select`, `test split final only`
- next practical path:
  1. run `epochs=1` smoke for A/B/C with full val/test only once per model
  2. if throughput is acceptable, rerun with `epochs=3`
  3. keep D optional until A/B/C are complete

## 5. Compliance checklist
- training_executed = false
- test_used_for_selection = false
- full_model_checkpoint_modified = false
- metrics_modified_existing = false
- checkpoints_committed = false
- raw_data_committed = false
- explicit_matching_used = false
- ransac_used = false
- pnp_used = false
- ba_used = false
