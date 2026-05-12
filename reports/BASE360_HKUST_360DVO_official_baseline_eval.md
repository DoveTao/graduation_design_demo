# BASE360 HKUST 360DVO official baseline eval

## 1. Executive summary
- BASE360 是否成功运行: `no`
- val coverage: `0.00%`
- test coverage: `0.00%`
- 是否获得可比 metrics: `no`
- 与 TRAIN360C 相比的主要结论: 当前无法形成 BASE360 官方结果，因为 official method 在本地未完成依赖安装、CUDA 扩展构建和权重下载；TRAIN360C 仍是当前唯一完整的 DSET2C canonical val/test 结果。
- 当前最大 caveat: BASE360 真实 inference 被环境/权重阻塞，因此主表中的 BASE360 行是 blocker-aware placeholder，而不是性能数字。

## 2. Official method inventory
- official code path: `/tmp/360DVO_official`
- version / commit: `46f5ed0735d39aff29a75f28c360600340879b35`
- required dependencies: `python=3.12, pytorch=2.3.1, pytorch-cuda=12.2, pytorch-scatter=2.1.2, evo, opencv-python, viser, yacs, custom CUDA extensions via pip install ., eigen-3.4.0 zip under thirdparty/`
- config path: `/tmp/360DVO_official/config/360.yaml`
- whether method was modified: `False`
- run commands:
  - `python3 /tmp/360DVO_official/demo.py --imagedir data/360DVO/Sequences/<sequence> --config /tmp/360DVO_official/config/360.yaml --save_trajectory --name <split>_<sequence>`
- inventory blockers:
- none

## 3. DSET2C canonical split compliance
- val/test sequence list:
- `mountains`
- `downhill_biking`
- `snowmobile`
- `ridge_to_lake`
- pair counts: `val=5342`, `test=5062`
- adjacent pair counts: `val=1339`, `test=1269`
- no random split: `true`
- no direct split glob: `true`
- canonical manifest / hygiene json usage: `true`

## 4. Input adapter
- sequence-level or pair-level: `sequence-level`
- image order: recovered from canonical manifest timestamps / frame paths, not directory glob split discovery
- calibration / ERP config: official demo uses ERP intrinsics derived from image height/width; no legacy scene01 calibration was used
- symlink/list/config generation: generated `image_list.txt`, `input_manifest.json`, and `gt_tum.txt` per sequence without copying images
- failure handling: every attempted run writes `stdout.log`, `stderr.log`, and `run_metadata.json` per sequence

## 5. Inference results
- per-sequence status: `val=blocked`, `test=blocked`
- runtime summary: official demo invocation attempted for each canonical sequence and failed before prediction export
- failed frames: full sequence coverage missing because no official prediction file was produced
- coverage: `val=0.00%`, `test=0.00%`

## 6. Evaluation protocol
- evaluator used: `tools/evaluate_external_baseline_trajectory.py` for TUM trajectory metrics
- metric definitions: ATE none / SE3 / Sim3 use the existing project evaluator; path_ratio remains raw predicted path length over GT path length
- trajectory alignment modes: `none`, `se3`, `sim3`
- relative pose computation: unavailable for BASE360 because official trajectory was not generated
- known incompatibilities: T57b reference row is not on DSET2C canonical val/test and is included only as external context

## 7. Val metrics
{
  "split": "val",
  "sequence_count": 2,
  "pair_count": 5342,
  "adjacent_pair_count": 1339,
  "coverage": 0.0,
  "ate_none": null,
  "ate_se3": null,
  "ate_sim3": null,
  "path_ratio": null,
  "predicted_path_length": null,
  "gt_path_length": null,
  "rot_mean_deg": null,
  "rot_median_deg": null,
  "signed_tdir_mean_deg": null,
  "signed_tdir_median_deg": null,
  "unsigned_tdir_mean_deg": null,
  "anti_parallel_rate": null,
  "tmag_median_ratio": null,
  "tmag_mean_ratio": null,
  "log_tmag_mae": null,
  "nan_inf_count": null,
  "execution_status": "blocked",
  "per_sequence": {
    "downhill_biking": {
      "frame_count": 576,
      "pair_count": 2293,
      "adjacent_pair_count": 575,
      "image_dir": "/home/dovetao/graduation_design_demo/data/360DVO/Sequences/downhill_biking",
      "gt_tum": "/home/dovetao/graduation_design_demo/external_baselines/results/base360_hkust_360dvo_official/val/downhill_biking/gt_tum.txt",
      "pred_tum": null,
      "run_metadata": "/home/dovetao/graduation_design_demo/external_baselines/results/base360_hkust_360dvo_official/val/downhill_biking/run_metadata.json",
      "metrics": {
        "available": false,
        "coverage": 0.0,
        "failed_frames": 576,
        "trajectory_eval": {},
        "relative_metrics": {}
      }
    },
    "mountains": {
      "frame_count": 765,
      "pair_count": 3049,
      "adjacent_pair_count": 764,
      "image_dir": "/home/dovetao/graduation_design_demo/data/360DVO/Sequences/mountains",
      "gt_tum": "/home/dovetao/graduation_design_demo/external_baselines/results/base360_hkust_360dvo_official/val/mountains/gt_tum.txt",
      "pred_tum": null,
      "run_metadata": "/home/dovetao/graduation_design_demo/external_baselines/results/base360_hkust_360dvo_official/val/mountains/run_metadata.json",
      "metrics": {
        "available": false,
        "coverage": 0.0,
        "failed_frames": 765,
        "trajectory_eval": {},
        "relative_metrics": {}
      }
    }
  }
}

## 8. Test metrics
{
  "split": "test",
  "sequence_count": 2,
  "pair_count": 5062,
  "adjacent_pair_count": 1269,
  "coverage": 0.0,
  "ate_none": null,
  "ate_se3": null,
  "ate_sim3": null,
  "path_ratio": null,
  "predicted_path_length": null,
  "gt_path_length": null,
  "rot_mean_deg": null,
  "rot_median_deg": null,
  "signed_tdir_mean_deg": null,
  "signed_tdir_median_deg": null,
  "unsigned_tdir_mean_deg": null,
  "anti_parallel_rate": null,
  "tmag_median_ratio": null,
  "tmag_mean_ratio": null,
  "log_tmag_mae": null,
  "nan_inf_count": null,
  "execution_status": "blocked",
  "per_sequence": {
    "ridge_to_lake": {
      "frame_count": 552,
      "pair_count": 2197,
      "adjacent_pair_count": 551,
      "image_dir": "/home/dovetao/graduation_design_demo/data/360DVO/Sequences/ridge_to_lake",
      "gt_tum": "/home/dovetao/graduation_design_demo/external_baselines/results/base360_hkust_360dvo_official/test/ridge_to_lake/gt_tum.txt",
      "pred_tum": null,
      "run_metadata": "/home/dovetao/graduation_design_demo/external_baselines/results/base360_hkust_360dvo_official/test/ridge_to_lake/run_metadata.json",
      "metrics": {
        "available": false,
        "coverage": 0.0,
        "failed_frames": 552,
        "trajectory_eval": {},
        "relative_metrics": {}
      }
    },
    "snowmobile": {
      "frame_count": 719,
      "pair_count": 2865,
      "adjacent_pair_count": 718,
      "image_dir": "/home/dovetao/graduation_design_demo/data/360DVO/Sequences/snowmobile",
      "gt_tum": "/home/dovetao/graduation_design_demo/external_baselines/results/base360_hkust_360dvo_official/test/snowmobile/gt_tum.txt",
      "pred_tum": null,
      "run_metadata": "/home/dovetao/graduation_design_demo/external_baselines/results/base360_hkust_360dvo_official/test/snowmobile/run_metadata.json",
      "metrics": {
        "available": false,
        "coverage": 0.0,
        "failed_frames": 719,
        "trajectory_eval": {},
        "relative_metrics": {}
      }
    }
  }
}

## 9. TRAIN360C vs BASE360 vs T57b comparison
- unified table: `/home/dovetao/graduation_design_demo/reports/TRAIN360_vs_BASE360_vs_T57b_summary.md`
- TRAIN360C test signed_tdir_mean_deg: `54.956992`
- BASE360 test signed_tdir_mean_deg: `N/A`
- which model is best on which metric: only TRAIN360C has canonical DSET2C val/test metrics at this point; BASE360 official is blocked and T57b is out-of-split reference only
- whether TRAIN360C remains competitive: yes, because it remains the only completed canonical evaluation
- whether BASE360 exposes weaknesses in TRAIN360C: not yet, because no official BASE360 metrics were produced

## 10. Failure / caveat analysis
- official method repository exists externally but is not vendored in the main repo
- current Python environment is missing at least `torch`, `evo`, and `viser` for the official method
- official repo also requires local CUDA extension build via `pip install .` and Eigen download before inference
- official weights were not present locally, so even a dependency-complete run would still be blocked without weight download
- val/test difficulty difference cannot be assessed for BASE360 because coverage is zero

## 11. Next step recommendation
- `fix_BASE360_blockers`

## 12. Compliance checklist
- `base360_inference_executed = false`
- `official_method_used_as_teacher = false`
- `train360_weights_modified = false`
- `train360_training_executed = false`
- `s5_locked_metrics_modified = false`
- `legacy_scene01_artifact_dependency = false`
- `random_pair_split_used = false`
- `dset2c_canonical_split_used = true`
- `val_used_for_tuning = false`
- `test_used_for_tuning = false`
- `s5e15_included_as_external_model = false`
