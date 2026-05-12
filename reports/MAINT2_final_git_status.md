# MAINT2 Final Git Status

## Final Checks

- branch: `maintenance/env-git-cleanup`
- upstream: `origin/maintenance/env-git-cleanup`
- HEAD: `315973c`
- push status: success
- working tree clean: false

## git status --short

```text
?? external_baselines/results/base360_hkust_360dvo_official/test/ridge_to_lake/image_list.txt
?? external_baselines/results/base360_hkust_360dvo_official/test/ridge_to_lake/input_manifest.json
?? external_baselines/results/base360_hkust_360dvo_official/test/ridge_to_lake/run_metadata.json
?? external_baselines/results/base360_hkust_360dvo_official/test/snowmobile/image_list.txt
?? external_baselines/results/base360_hkust_360dvo_official/test/snowmobile/input_manifest.json
?? external_baselines/results/base360_hkust_360dvo_official/test/snowmobile/run_metadata.json
?? external_baselines/results/base360_hkust_360dvo_official/val/downhill_biking/image_list.txt
?? external_baselines/results/base360_hkust_360dvo_official/val/downhill_biking/input_manifest.json
?? external_baselines/results/base360_hkust_360dvo_official/val/downhill_biking/run_metadata.json
?? external_baselines/results/base360_hkust_360dvo_official/val/mountains/image_list.txt
?? external_baselines/results/base360_hkust_360dvo_official/val/mountains/input_manifest.json
?? external_baselines/results/base360_hkust_360dvo_official/val/mountains/run_metadata.json
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
* maintenance/env-git-cleanup                         315973c [origin/maintenance/env-git-cleanup] MAINT: document experiment artifact batch commits
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

## git log --oneline --decorate -n 12

```text
315973c (HEAD -> maintenance/env-git-cleanup, origin/maintenance/env-git-cleanup) MAINT: document experiment artifact batch commits
fb5b9ad RESULTS360: add main comparison tables and experiment narrative
b840e89 BASE360: add HKUST official baseline runner and aligned metrics
8e0af9c TRAIN360: add manifest-native baseline pipeline and reports
4f40d9d MAINT: record remote push status in cleanup summary
3b1fbd9 MAINT: update cleanup summary with commit status
0f00dfd MAINT: add conda environment inventory and git cleanup notes
ba882f7 (tag: dset2c-canonical-train360-ready, origin/experiment/dset2c-360dvo-dataset-hygiene, experiment/dset2c-360dvo-dataset-hygiene) DSET2C: add canonical manifest report and checkpoint
cc5dd90 DSET2C: add 360DVO dataset hygiene tooling
ed76371 (tag: dset2b-train360-ready, origin/experiment/dset2b-360dvo-pair-count-completion, experiment/dset2b-360dvo-pair-count-completion) DSET2B: add train360 readiness report and checkpoint
4cd0a0d DSET2B: add 360DVO pair count completion tooling
3468aa4 (origin/experiment/dset2-360dvo-fuller-split-expansion, experiment/dset2-360dvo-fuller-split-expansion) DSET2: add fuller split report and checkpoint
```

## Remaining Files

- intentionally untracked local-only artifacts:
  - `external_baselines/results/base360_hkust_360dvo_official/test/ridge_to_lake/image_list.txt``: official BASE360 local provenance helper; not required for code/report reproducibility and intentionally left out of commits
  - `external_baselines/results/base360_hkust_360dvo_official/test/ridge_to_lake/input_manifest.json``: official BASE360 local provenance helper; not required for code/report reproducibility and intentionally left out of commits
  - `external_baselines/results/base360_hkust_360dvo_official/test/ridge_to_lake/run_metadata.json``: official BASE360 local provenance helper; not required for code/report reproducibility and intentionally left out of commits
  - `external_baselines/results/base360_hkust_360dvo_official/test/snowmobile/image_list.txt``: official BASE360 local provenance helper; not required for code/report reproducibility and intentionally left out of commits
  - `external_baselines/results/base360_hkust_360dvo_official/test/snowmobile/input_manifest.json``: official BASE360 local provenance helper; not required for code/report reproducibility and intentionally left out of commits
  - `external_baselines/results/base360_hkust_360dvo_official/test/snowmobile/run_metadata.json``: official BASE360 local provenance helper; not required for code/report reproducibility and intentionally left out of commits
  - `external_baselines/results/base360_hkust_360dvo_official/val/downhill_biking/image_list.txt``: official BASE360 local provenance helper; not required for code/report reproducibility and intentionally left out of commits
  - `external_baselines/results/base360_hkust_360dvo_official/val/downhill_biking/input_manifest.json``: official BASE360 local provenance helper; not required for code/report reproducibility and intentionally left out of commits
  - `external_baselines/results/base360_hkust_360dvo_official/val/downhill_biking/run_metadata.json``: official BASE360 local provenance helper; not required for code/report reproducibility and intentionally left out of commits
  - `external_baselines/results/base360_hkust_360dvo_official/val/mountains/image_list.txt``: official BASE360 local provenance helper; not required for code/report reproducibility and intentionally left out of commits
  - `external_baselines/results/base360_hkust_360dvo_official/val/mountains/input_manifest.json``: official BASE360 local provenance helper; not required for code/report reproducibility and intentionally left out of commits
  - `external_baselines/results/base360_hkust_360dvo_official/val/mountains/run_metadata.json``: official BASE360 local provenance helper; not required for code/report reproducibility and intentionally left out of commits
- intentionally ignored large artifacts: checkpoints, raw trajectories, smoke images, resized demo inputs, logs, and raw data remain outside Git according to `.gitignore`
- needs future commit: none among the TRAIN360 / BASE360 / RESULTS360 experiment artifacts requested in MAINT2
- unexpected leftovers: none beyond the untracked local provenance helpers listed above

## Recommendation

- do not create `experiment/train360d-observability-kstep-scale` yet because the working tree is not clean
- next: either ignore or relocate the remaining BASE360 provenance helpers, then branch for `TRAIN360D`
