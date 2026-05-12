# MAINT1 Git Inventory

## git status --short

```text
 M .gitignore
?? configs/train360_v0_baseline.yaml
?? configs/train360_v0_manifest_sanity.yaml
?? datasets/
?? envs/
?? external_baselines/results/base360_hkust_360dvo_official/
?? reports/BASE360B_env_inventory.md
?? reports/BASE360B_fix_HKUST_360DVO_official_env_and_rerun.md
?? reports/BASE360C_cuda_ba_build_inventory.md
?? reports/BASE360C_fix_cuda_ba_extension_and_rerun_official_smoke.md
?? reports/BASE360D_component_metric_alignment.md
?? reports/BASE360D_metrics_test.json
?? reports/BASE360D_metrics_val.json
?? reports/BASE360_HKUST_360DVO_official_baseline_eval.md
?? reports/BASE360_metrics_test.json
?? reports/BASE360_metrics_val.json
?? reports/MAINT1_conda_env_inventory.json
?? reports/MAINT1_conda_env_inventory.md
?? reports/MAINT1_git_inventory.json
?? reports/MAINT1_git_inventory.md
?? reports/RESULTS360_experiment_narrative.md
?? reports/RESULTS360_main_results_table.json
?? reports/RESULTS360_main_results_table.md
?? reports/TRAIN360A_architecture_inventory_and_reusable_module_audit.md
?? reports/TRAIN360A_reusable_module_matrix.json
?? reports/TRAIN360B_forward_sanity.json
?? reports/TRAIN360B_manifest_native_dataloader_adapter_and_forward_sanity.md
?? reports/TRAIN360C_metrics_test.json
?? reports/TRAIN360C_metrics_val.json
?? reports/TRAIN360C_spherical_pose_baseline.md
?? reports/TRAIN360_vs_BASE360_vs_T57b_summary.md
?? tools/base360c_ittnotify_shim.c
?? tools/base360c_official_python
?? tools/base360d_component_metric_alignment.py
?? tools/eval_train360_pose.py
?? tools/run_base360_hkust_360dvo_official.py
?? tools/train360b_forward_sanity.py
?? tools/train360c_spherical_pose_baseline.py
?? train360_pose_losses.py
```

## git branch --show-current

```text
maintenance/env-git-cleanup
```

## git branch -vv

```text
  curation/final-report-archive                       0b8b54a [origin/curation/final-report-archive] Materialize curated reports archive
  experiment/arch2p-parameter-sensitivity-sweep       1451f52 ARCH2P: add sweep report and checkpoint
  experiment/dset1-360dvo-main-dataset-migration      e0a7f58 [origin/experiment/dset1-360dvo-main-dataset-migration] GEN5: add external evaluation report and checkpoint
  experiment/dset2-360dvo-fuller-split-expansion      3468aa4 DSET2: add fuller split report and checkpoint
  experiment/dset2b-360dvo-pair-count-completion      ed76371 DSET2B: add train360 readiness report and checkpoint
  experiment/dset2c-360dvo-dataset-hygiene            ba882f7 DSET2C: add canonical manifest report and checkpoint
  experiment/gen2-360dvo-external-adapter             016ec20 [origin/experiment/gen2-360dvo-external-adapter] GEN2: add 360DVO adapter report and checkpoint
  experiment/gen2b-360dvo-one-sequence-smoke          a6307f6 [origin/experiment/gen2b-360dvo-one-sequence-smoke: 领先 1] GEN3: add 360DVO external inference bridge
  experiment/gen3-360dvo-external-inference-bridge    1d476b2 [origin/experiment/gen3-360dvo-external-inference-bridge] GEN4: add recovery report and checkpoint
  experiment/jrt1-joint-rtdir-coupled-refiner         6765af3 [origin/experiment/jrt1-joint-rtdir-coupled-refiner] Add JRT1 branch remote sync audit
  experiment/mf1-multi-frame-chain-refiner            ea559d5 [origin/experiment/mf1-multi-frame-chain-refiner] Add MF1 multi-frame chain refiner artifacts
  experiment/orbslam3-fisheye-strong-baseline         797865b [origin/experiment/orbslam3-fisheye-strong-baseline] Add S5 dense same-evaluator comparison artifacts
  experiment/s5d2-dense-export-convention-audit       9de0632 [origin/experiment/s5d2-dense-export-convention-audit] Add curated report archive plan
  experiment/s5e1-traceable-adjacent-dense            a8e81e0 [origin/experiment/s5e1-traceable-adjacent-dense] ARCH2: add reports checkpoints and lightweight summaries
  experiment/struct1-geometry-token-pose-solver       040579f [origin/experiment/struct1-geometry-token-pose-solver] GEN1: add external dataset feasibility report and checkpoint
  main                                                bb8fadd [origin/main] Add S1d5 cleanup and staging audit reports
* maintenance/env-git-cleanup                         ba882f7 DSET2C: add canonical manifest report and checkpoint
  optimize/s10-chain-pathratio-smoother               0ae9aee [origin/optimize/s10-chain-pathratio-smoother] Add S10 chain-level path-ratio preserving smoother diagnostic
  optimize/s11-tmag-scale-consistency-training        b00eab4 [origin/optimize/s11-tmag-scale-consistency-training] Close code optimization after S10 S11 diagnostics
  optimize/s12-regime-balanced-sampling               34edd74 [origin/optimize/s12-regime-balanced-sampling] Add S13 practical usability gap analysis
  optimize/s14-local-window-pose-graph                36e68fe [origin/optimize/s14-local-window-pose-graph] Close code optimization after S14 pose graph diagnostic
  optimize/s15-trajectory-level-training-objective    8a28577 [origin/optimize/s15-trajectory-level-training-objective] Close S15 trajectory training line after stability retest
  optimize/s16-stronger-visual-backbone-feasibility   8f52076 [origin/optimize/s16-stronger-visual-backbone-feasibility] Close S16 backbone feasibility after ResNet50 probe
  optimize/s17-pose-supervision-dataset-quality-audit e020e11 [origin/optimize/s17-pose-supervision-dataset-quality-audit] Add S18 split representativeness evaluation
  optimize/s19-geometry-aware-pretraining-feasibility f90d167 [origin/optimize/s19-geometry-aware-pretraining-feasibility] Add git branch remote sync audit
  optimize/s2-fine-refinement-on-s1d5                 c1f4b36 [origin/optimize/s2-fine-refinement-on-s1d5] Add S3 coupled pose head design plan
  optimize/s3a0-coupled-pose-residual-head            f232fa3 [origin/optimize/s3a0-coupled-pose-residual-head] Close post-S5 optimization after S8 S9 diagnostics
  polish/final-reproducibility-guardrails             7cb0a2c [origin/polish/final-reproducibility-guardrails] Add repository hygiene audit before JRT1
```

## git remote -v

```text
origin	git@github.com:DoveTao/graduation_design_demo.git (fetch)
origin	git@github.com:DoveTao/graduation_design_demo.git (push)
```

## git log --oneline --decorate -n 20

```text
ba882f7 (HEAD -> maintenance/env-git-cleanup, tag: dset2c-canonical-train360-ready, origin/experiment/dset2c-360dvo-dataset-hygiene, experiment/dset2c-360dvo-dataset-hygiene) DSET2C: add canonical manifest report and checkpoint
cc5dd90 DSET2C: add 360DVO dataset hygiene tooling
ed76371 (tag: dset2b-train360-ready, origin/experiment/dset2b-360dvo-pair-count-completion, experiment/dset2b-360dvo-pair-count-completion) DSET2B: add train360 readiness report and checkpoint
4cd0a0d DSET2B: add 360DVO pair count completion tooling
3468aa4 (origin/experiment/dset2-360dvo-fuller-split-expansion, experiment/dset2-360dvo-fuller-split-expansion) DSET2: add fuller split report and checkpoint
280d8e2 DSET2: add 360DVO fuller split expansion tooling
e0a7f58 (origin/experiment/dset1-360dvo-main-dataset-migration, experiment/dset1-360dvo-main-dataset-migration) GEN5: add external evaluation report and checkpoint
7bd7cab GEN5: add 360DVO T57b external evaluation tools
f543f67 (tag: dset1-360dvo-main-dataset-ready) DSET1: add dataset migration reports and checkpoints
2125c84 DATA3: add legacy panorama quality audit
7c9aa45 DSET1: add 360DVO main dataset migration tooling
a6307f6 (experiment/gen2b-360dvo-one-sequence-smoke) GEN3: add 360DVO external inference bridge
b149aeb (origin/experiment/gen2b-360dvo-one-sequence-smoke) GEN2B: add one-sequence smoke report and checkpoint
a024872 GEN2B: add 360DVO one-sequence smoke download tooling
016ec20 (origin/experiment/gen2-360dvo-external-adapter, experiment/gen2-360dvo-external-adapter) GEN2: add 360DVO adapter report and checkpoint
6d0c186 GEN2: add 360DVO external adapter smoke tooling
040579f (origin/experiment/struct1-geometry-token-pose-solver, experiment/struct1-geometry-token-pose-solver) GEN1: add external dataset feasibility report and checkpoint
bb23c89 GEN1: add external panoramic dataset feasibility audit
7e6e2e8 DATA2: add observability report and checkpoint
8183df1 DATA2: add tdir scale observability audit
```

## git tag --list | sort | tail -n 50

```text
dset1-360dvo-main-dataset-ready
dset2b-train360-ready
dset2c-canonical-train360-ready
final-s5-clean-candidate-post-s9
s1d5-clean-mainline
s5e12-feature-gate-passed
```

## git diff --stat

```text
 .gitignore | 3 +++
 1 file changed, 3 insertions(+)
```

## git diff --name-only

```text
.gitignore
```

## git ls-files --others --exclude-standard

```text
configs/train360_v0_baseline.yaml
configs/train360_v0_manifest_sanity.yaml
datasets/__init__.py
datasets/dset2c_manifest_dataset.py
envs/base360dvo.candidate_old.lock.yml
envs/base360dvo.candidate_old.pip-freeze.txt
envs/base360dvo_rebuild.lock.yml
envs/base360dvo_rebuild.pip-freeze.txt
envs/pytorch.lock.yml
envs/pytorch.pip-freeze.txt
external_baselines/results/base360_hkust_360dvo_official/canonical_manifest_summary_snapshot.json
external_baselines/results/base360_hkust_360dvo_official/component_metrics/alignment_inventory.json
external_baselines/results/base360_hkust_360dvo_official/component_metrics/test_component_metrics.json
external_baselines/results/base360_hkust_360dvo_official/component_metrics/val_component_metrics.json
external_baselines/results/base360_hkust_360dvo_official/dset2c_hygiene_snapshot.json
external_baselines/results/base360_hkust_360dvo_official/official_method_inventory.json
external_baselines/results/base360_hkust_360dvo_official/terminal_summary.json
external_baselines/results/base360_hkust_360dvo_official/test/ridge_to_lake/image_list.txt
external_baselines/results/base360_hkust_360dvo_official/test/ridge_to_lake/input_manifest.json
external_baselines/results/base360_hkust_360dvo_official/test/ridge_to_lake/per_sequence_metrics.json
external_baselines/results/base360_hkust_360dvo_official/test/ridge_to_lake/run_metadata.json
external_baselines/results/base360_hkust_360dvo_official/test/snowmobile/image_list.txt
external_baselines/results/base360_hkust_360dvo_official/test/snowmobile/input_manifest.json
external_baselines/results/base360_hkust_360dvo_official/test/snowmobile/per_sequence_metrics.json
external_baselines/results/base360_hkust_360dvo_official/test/snowmobile/run_metadata.json
external_baselines/results/base360_hkust_360dvo_official/test/split_metrics.json
external_baselines/results/base360_hkust_360dvo_official/val/downhill_biking/image_list.txt
external_baselines/results/base360_hkust_360dvo_official/val/downhill_biking/input_manifest.json
external_baselines/results/base360_hkust_360dvo_official/val/downhill_biking/per_sequence_metrics.json
external_baselines/results/base360_hkust_360dvo_official/val/downhill_biking/run_metadata.json
external_baselines/results/base360_hkust_360dvo_official/val/mountains/image_list.txt
external_baselines/results/base360_hkust_360dvo_official/val/mountains/input_manifest.json
external_baselines/results/base360_hkust_360dvo_official/val/mountains/per_sequence_metrics.json
external_baselines/results/base360_hkust_360dvo_official/val/mountains/run_metadata.json
external_baselines/results/base360_hkust_360dvo_official/val/split_metrics.json
reports/BASE360B_env_inventory.md
reports/BASE360B_fix_HKUST_360DVO_official_env_and_rerun.md
reports/BASE360C_cuda_ba_build_inventory.md
reports/BASE360C_fix_cuda_ba_extension_and_rerun_official_smoke.md
reports/BASE360D_component_metric_alignment.md
reports/BASE360D_metrics_test.json
reports/BASE360D_metrics_val.json
reports/BASE360_HKUST_360DVO_official_baseline_eval.md
reports/BASE360_metrics_test.json
reports/BASE360_metrics_val.json
reports/MAINT1_conda_env_inventory.json
reports/MAINT1_conda_env_inventory.md
reports/MAINT1_git_inventory.json
reports/MAINT1_git_inventory.md
reports/RESULTS360_experiment_narrative.md
reports/RESULTS360_main_results_table.json
reports/RESULTS360_main_results_table.md
reports/TRAIN360A_architecture_inventory_and_reusable_module_audit.md
reports/TRAIN360A_reusable_module_matrix.json
reports/TRAIN360B_forward_sanity.json
reports/TRAIN360B_manifest_native_dataloader_adapter_and_forward_sanity.md
reports/TRAIN360C_metrics_test.json
reports/TRAIN360C_metrics_val.json
reports/TRAIN360C_spherical_pose_baseline.md
reports/TRAIN360_vs_BASE360_vs_T57b_summary.md
tools/base360c_ittnotify_shim.c
tools/base360c_official_python
tools/base360d_component_metric_alignment.py
tools/eval_train360_pose.py
tools/run_base360_hkust_360dvo_official.py
tools/train360b_forward_sanity.py
tools/train360c_spherical_pose_baseline.py
train360_pose_losses.py
```
