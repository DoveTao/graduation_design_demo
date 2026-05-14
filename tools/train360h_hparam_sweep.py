#!/usr/bin/env python3
from __future__ import annotations

import copy
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from datasets.dset2c_manifest_dataset import Dset2CCanonicalPairDataset
from model import PanoramaRelPoseModel
from tools.eval_train360_pose import evaluate_train360_pose
from tools.miniyaml import load_yaml_like
from tools.train360d_observability_kstep_scale import _cfg_from_dict, _git, _load_model_with_status, _run_prechecks


TRAIN360D_SCRIPT = REPO_ROOT / "tools" / "train360d_observability_kstep_scale.py"
DEFAULT_SWEEP_CFG = REPO_ROOT / "configs" / "sweeps" / "train360h_sweep_space.yaml"


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _deep_update(dst: Dict[str, Any], src: Mapping[str, Any]) -> Dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, Mapping) and isinstance(dst.get(key), dict):
            _deep_update(dst[key], value)
        else:
            dst[key] = copy.deepcopy(value)
    return dst


def _yaml_dump_like(value: Any, indent: int = 0) -> List[str]:
    pad = " " * indent
    if isinstance(value, dict):
        lines: List[str] = []
        for key, val in value.items():
            if isinstance(val, dict):
                lines.append(f"{pad}{key}:")
                lines.extend(_yaml_dump_like(val, indent + 2))
            elif isinstance(val, list):
                if not val:
                    lines.append(f"{pad}{key}: []")
                else:
                    lines.append(f"{pad}{key}:")
                    lines.extend(_yaml_dump_like(val, indent + 2))
            else:
                lines.append(f"{pad}{key}: {_scalar_to_yaml(val)}")
        return lines
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, dict):
                lines.append(f"{pad}-")
                lines.extend(_yaml_dump_like(item, indent + 2))
            else:
                lines.append(f"{pad}- {_scalar_to_yaml(item)}")
        return lines
    return [f"{pad}{_scalar_to_yaml(value)}"]


def _scalar_to_yaml(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return "null"
    if isinstance(value, str):
        if value == "" or any(ch in value for ch in [":", "#", "{", "}", "[", "]"]):
            return json.dumps(value, ensure_ascii=False)
        return value
    if isinstance(value, list):
        return "[" + ", ".join(_scalar_to_yaml(v) for v in value) + "]"
    return str(value)


def _write_yaml_like(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(_yaml_dump_like(dict(payload))) + "\n", encoding="utf-8")


def _run_cmd(cmd: Sequence[str], *, cwd: Path, log_prefix: str) -> None:
    proc = subprocess.run(cmd, cwd=str(cwd), check=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{log_prefix} failed with code {proc.returncode}")


def _make_run_config(
    base_cfg: Mapping[str, Any],
    sweep_cfg: Mapping[str, Any],
    run_id: str,
    phase: str,
    overrides: Mapping[str, Any],
    *,
    run_test: bool,
) -> Dict[str, Any]:
    cfg = copy.deepcopy(dict(base_cfg))
    common_overrides = sweep_cfg.get("common_overrides", {})
    if common_overrides:
        _deep_update(cfg, common_overrides)
    outputs = {
        "checkpoint_dir": f"checkpoints/TRAIN360H_sweep/{run_id}",
        "report_path": f"checkpoints/TRAIN360H_sweep/{run_id}/run_report.md",
        "val_metrics_path": f"checkpoints/TRAIN360H_sweep/{run_id}/metrics_val.json",
        "test_metrics_path": f"checkpoints/TRAIN360H_sweep/{run_id}/metrics_test.json",
        "comparison_summary_path": f"checkpoints/TRAIN360H_sweep/{run_id}/comparison_summary.md",
        "next_recommendation": "proceed_to_TRAIN360I_final_retrain_with_best_hparams",
    }
    cfg["task_name"] = run_id
    cfg["outputs"] = outputs
    cfg["evaluation"] = copy.deepcopy(dict(cfg.get("evaluation", {})))
    cfg["evaluation"]["run_test"] = bool(run_test)
    cfg["metadata"] = {
        "phase": phase,
        "run_id": run_id,
        "selection_policy": "val_only" if not run_test else "final_candidate_eval_only",
    }
    _deep_update(cfg, overrides)
    return cfg


def _run_training(run_cfg: Mapping[str, Any], run_dir: Path) -> None:
    run_cfg_path = run_dir / "launcher_config.yaml"
    _write_yaml_like(run_cfg_path, run_cfg)
    _run_cmd([sys.executable, str(TRAIN360D_SCRIPT), str(run_cfg_path)], cwd=REPO_ROOT, log_prefix=run_cfg["task_name"])


def _iter_specs(spec_map: Mapping[str, Any]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for run_id, overrides in spec_map.items():
        items.append({"run_id": str(run_id), "overrides": copy.deepcopy(dict(overrides))})
    return items


def _build_eval_loader(manifest_path: str, split: str, image_hw: Sequence[int], batch_size: int, num_workers: int) -> DataLoader:
    dataset = Dset2CCanonicalPairDataset(
        str(REPO_ROOT / manifest_path),
        expected_split=split,
        image_hw=(int(image_hw[0]), int(image_hw[1])),
        require_paths=True,
        skip_invalid=True,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(int(num_workers) > 0),
        drop_last=False,
    )


def _evaluate_checkpoint(run_cfg: Mapping[str, Any], checkpoint_path: Path, split: str) -> Dict[str, Any]:
    cfg = run_cfg
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    inputs = cfg["inputs"]
    manifest_path = inputs["val_manifest"] if split == "val" else inputs["test_manifest"]
    loader = _build_eval_loader(
        manifest_path,
        split,
        image_hw=data_cfg["image_hw"],
        batch_size=int(data_cfg["eval_batch_size"]),
        num_workers=int(data_cfg["num_workers"]),
    )
    use_cuda = bool(model_cfg["use_cuda_if_available"]) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    payload = torch.load(str(checkpoint_path), map_location=device)
    ckpt_cfg = _cfg_from_dict(payload["cfg"])
    model = PanoramaRelPoseModel(ckpt_cfg, device).to(device)
    model.load_state_dict(payload["model"], strict=False)
    model.eval()
    metrics = evaluate_train360_pose(
        model,
        loader,
        device,
        enable_depth_fusion=bool(model_cfg["enable_depth_fusion"]),
        tmag_epsilon=float(data_cfg["tmag_epsilon"]),
        max_batches=None,
    )
    return {
        "task_name": str(cfg["task_name"]),
        "checkpoint_used": str(checkpoint_path),
        "metrics": metrics,
    }


def _score_from_metrics(metrics_payload: Mapping[str, Any]) -> float:
    return float(metrics_payload.get("val_score") or metrics_payload.get("val_score_of_selected_checkpoint") or float("inf"))


def _load_run_result(run_dir: Path) -> Dict[str, Any]:
    val_payload = _read_json(run_dir / "metrics_val.json")
    test_payload = _read_json(run_dir / "metrics_test.json")
    metadata = _read_json(run_dir / "metadata.json")
    cfg = load_yaml_like(run_dir / "config.yaml")
    return {
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "config": cfg,
        "metadata": metadata,
        "val_payload": val_payload,
        "test_payload": test_payload,
        "val_metrics": val_payload.get("metrics", {}),
        "test_metrics": test_payload.get("metrics", {}),
        "val_score": float(val_payload.get("val_score", float("inf"))),
        "phase": metadata.get("phase", cfg.get("metadata", {}).get("phase")),
    }


def _candidate_rows(results: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in results:
        val = item.get("val_metrics", {})
        test = item.get("test_metrics", {}) if isinstance(item.get("test_metrics"), Mapping) else {}
        rows.append(
            {
                "run_id": item["run_id"],
                "phase": item.get("phase"),
                "val_score": item.get("val_score"),
                "val_signed_tdir_mean_deg": val.get("signed_tdir_mean_deg"),
                "val_anti_parallel_rate": val.get("anti_parallel_rate"),
                "val_tmag_median_ratio": val.get("tmag_median_ratio"),
                "val_path_ratio": val.get("path_ratio"),
                "test_signed_tdir_mean_deg": test.get("signed_tdir_mean_deg"),
                "test_anti_parallel_rate": test.get("anti_parallel_rate"),
                "test_tmag_median_ratio": test.get("tmag_median_ratio"),
                "test_path_ratio": test.get("path_ratio"),
                "checkpoint": str(Path(item["run_dir"]) / "best_val.pt"),
            }
        )
    return rows


def _generate_refined_runs(best_result: Mapping[str, Any], sweep_cfg: Mapping[str, Any]) -> List[Dict[str, Any]]:
    best_cfg = best_result["config"]
    base_lr = float(best_cfg["training"]["lr"])
    base_tdir = float(best_cfg["loss"]["tdir_weight"])
    base_tmag = float(best_cfg["loss"]["tmag_weight"])
    base_scale = float(best_cfg["loss"]["scale_stability_weight"])
    base_grad_clip = float(best_cfg["training"]["grad_clip_norm"])
    return [
        {
            "run_id": "TRAIN360H_R01_lower_lr_more_tdir_seed0",
            "overrides": {
                "training": {"lr": max(base_lr * 0.6, 1.0e-5), "epochs": int(sweep_cfg["phases"]["refined"]["epochs"]), "seed": 3407, "grad_clip_norm": base_grad_clip},
                "data": {"train_subset_max": int(sweep_cfg["phases"]["refined"]["train_subset_max"])},
                "loss": {"tdir_weight": base_tdir + 0.5, "tmag_weight": base_tmag, "scale_stability_weight": base_scale},
            },
        },
        {
            "run_id": "TRAIN360H_R02_more_scale_seed0",
            "overrides": {
                "training": {"lr": base_lr, "epochs": int(sweep_cfg["phases"]["refined"]["epochs"]), "seed": 3407, "grad_clip_norm": base_grad_clip},
                "data": {"train_subset_max": int(sweep_cfg["phases"]["refined"]["train_subset_max"])},
                "loss": {"tdir_weight": base_tdir, "tmag_weight": base_tmag + 0.25, "scale_stability_weight": base_scale + 0.05},
            },
        },
        {
            "run_id": "TRAIN360H_R03_tighter_clip_seed0",
            "overrides": {
                "training": {"lr": base_lr, "epochs": int(sweep_cfg["phases"]["refined"]["epochs"]), "seed": 3407, "grad_clip_norm": max(0.5, base_grad_clip / 2.0)},
                "data": {"train_subset_max": int(sweep_cfg["phases"]["refined"]["train_subset_max"])},
                "loss": {"tdir_weight": base_tdir + 0.25, "tmag_weight": base_tmag + 0.1, "scale_stability_weight": base_scale},
            },
        },
    ]


def _seed_variants(best_result: Mapping[str, Any], sweep_cfg: Mapping[str, Any]) -> List[Dict[str, Any]]:
    best_cfg = best_result["config"]
    runs = []
    for seed in sweep_cfg["phases"]["seed_robustness"]["seeds"]:
        runs.append(
            {
                "run_id": f"TRAIN360H_S{seed}_best_seed{seed}",
                "overrides": {
                    "training": {
                        "lr": float(best_cfg["training"]["lr"]),
                        "epochs": int(sweep_cfg["phases"]["seed_robustness"]["epochs"]),
                        "seed": int(seed),
                        "weight_decay": float(best_cfg["training"]["weight_decay"]),
                        "grad_clip_norm": float(best_cfg["training"]["grad_clip_norm"]),
                        "scheduler": best_cfg["training"]["scheduler"],
                        "min_lr": float(best_cfg["training"]["min_lr"]),
                    },
                    "data": {"train_subset_max": int(sweep_cfg["phases"]["seed_robustness"]["train_subset_max"])},
                    "loss": {
                        "rot_weight": float(best_cfg["loss"]["rot_weight"]),
                        "tdir_weight": float(best_cfg["loss"]["tdir_weight"]),
                        "tmag_weight": float(best_cfg["loss"]["tmag_weight"]),
                        "scale_stability_weight": float(best_cfg["loss"]["scale_stability_weight"]),
                        "k_step_balancing": best_cfg["loss"]["k_step_balancing"],
                        "observability": best_cfg["loss"]["observability"],
                        "scale_stability": best_cfg["loss"]["scale_stability"],
                        "components": best_cfg["loss"].get("components", {}),
                    },
                    "evaluation": {"selection_score": best_cfg["evaluation"]["selection_score"]},
                },
            }
        )
    return runs


def _mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def _select_final_candidate(seed_results: List[Mapping[str, Any]], base_result: Mapping[str, Any]) -> Mapping[str, Any]:
    grouped: Dict[str, List[float]] = {}
    sources: Dict[str, Mapping[str, Any]] = {}
    for result in [base_result, *seed_results]:
        prefix = result["run_id"].split("_seed")[0]
        grouped.setdefault(prefix, []).append(float(result["val_score"]))
        if prefix not in sources or result["run_id"].endswith("seed0"):
            sources[prefix] = result
    ranked = sorted(grouped.items(), key=lambda kv: (_mean(kv[1]) if _mean(kv[1]) is not None else float("inf")))
    return sources[ranked[0][0]]


def main() -> None:
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SWEEP_CFG
    sweep_cfg = load_yaml_like(cfg_path)
    base_cfg = load_yaml_like(REPO_ROOT / sweep_cfg["base_train360d_config"])
    precheck = _run_prechecks(base_cfg, REPO_ROOT / sweep_cfg["base_train360d_config"])
    blockers = list(precheck["blockers"])
    if precheck["branch"] != sweep_cfg["expected_branch"]:
        blockers.append(f"branch mismatch: {precheck['branch']} != {sweep_cfg['expected_branch']}")
    if not (REPO_ROOT / sweep_cfg["required_train360d_checkpoint"]).is_file():
        blockers.append(f"missing TRAIN360D best checkpoint: {REPO_ROOT / sweep_cfg['required_train360d_checkpoint']}")
    if blockers:
        blocker_report = {
            "task_name": sweep_cfg["task_name"],
            "sweep_executed": False,
            "blockers": blockers,
            "precheck": precheck,
        }
        _json_dump(REPO_ROOT / "reports" / "TRAIN360H_best_config_val.json", blocker_report)
        _json_dump(REPO_ROOT / "reports" / "TRAIN360H_best_config_test.json", blocker_report)
        raise RuntimeError("TRAIN360H precheck failed:\n- " + "\n- ".join(blockers))

    sweep_root = REPO_ROOT / "checkpoints" / "TRAIN360H_sweep"
    sweep_root.mkdir(parents=True, exist_ok=True)

    phase_results: Dict[str, List[Dict[str, Any]]] = {"ablation": [], "coarse": [], "refined": [], "seed": [], "final_eval": []}
    run_registry: List[Dict[str, Any]] = []

    def execute_run(spec: Mapping[str, Any], phase: str, run_test: bool) -> Dict[str, Any]:
        run_id = str(spec["run_id"])
        overrides = copy.deepcopy(dict(spec["overrides"]))
        run_dir = sweep_root / run_id
        run_cfg = _make_run_config(base_cfg, sweep_cfg, run_id, phase, overrides, run_test=run_test)
        _run_training(run_cfg, run_dir)
        result = _load_run_result(run_dir)
        result["phase"] = phase
        run_registry.append(result)
        phase_results[phase].append(result)
        return result

    start = time.time()
    for spec in _iter_specs(sweep_cfg["ablation_runs"]):
        execute_run(spec, "ablation", run_test=False)

    for result in list(phase_results["ablation"]):
        run_cfg = load_yaml_like(Path(result["run_dir"]) / "config.yaml")
        ckpt = Path(result["run_dir"]) / "best_val.pt"
        test_payload = _evaluate_checkpoint(run_cfg, ckpt, "test")
        _json_dump(Path(result["run_dir"]) / "metrics_test.json", test_payload)
        result.update(_load_run_result(Path(result["run_dir"])))

    for spec in _iter_specs(sweep_cfg["coarse_runs"]):
        execute_run(spec, "coarse", run_test=False)

    best_coarse = min(phase_results["coarse"], key=lambda x: x["val_score"])
    refined_specs = _generate_refined_runs(best_coarse, sweep_cfg)
    for spec in refined_specs:
        execute_run(spec, "refined", run_test=False)

    all_val_candidates = phase_results["coarse"] + phase_results["refined"]
    best_refined = min(all_val_candidates, key=lambda x: x["val_score"])
    for spec in _seed_variants(best_refined, sweep_cfg):
        execute_run(spec, "seed", run_test=False)

    final_candidate = _select_final_candidate(phase_results["seed"], best_refined)
    final_candidates = [final_candidate]

    for idx, cand in enumerate(final_candidates, start=1):
        run_id = f"TRAIN360H_F0{idx}_{cand['run_id']}"
        spec = {
            "run_id": run_id,
            "overrides": {
                "training": cand["config"]["training"],
                "data": {**cand["config"]["data"], "train_subset_max": int(sweep_cfg["phases"]["final_eval"]["train_subset_max"])},
                "loss": cand["config"]["loss"],
                "evaluation": {**cand["config"]["evaluation"], "max_val_batches": None, "max_test_batches": None},
            },
        }
        result = execute_run(spec, "final_eval", run_test=True)
        result["selection_source_run_id"] = cand["run_id"]

    runtime_sec = float(time.time() - start)
    summary = {
        "task_name": sweep_cfg["task_name"],
        "sweep_executed": True,
        "runtime_sec": runtime_sec,
        "phases": {k: [r["run_id"] for r in v] for k, v in phase_results.items()},
        "best_coarse_run_id": best_coarse["run_id"],
        "best_val_selected_run_id": final_candidate["run_id"],
        "final_eval_run_ids": [r["run_id"] for r in phase_results["final_eval"]],
    }
    _json_dump(sweep_root / "sweep_summary.json", summary)

    subprocess.run([sys.executable, str(REPO_ROOT / "tools" / "summarize_train360h_sweep.py"), str(cfg_path)], cwd=str(REPO_ROOT), check=True)


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    main()
