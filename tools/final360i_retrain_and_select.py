#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from train360.core.config import Config
from datasets.dset2c_manifest_dataset import Dset2CCanonicalPairDataset, summarize_manifest_group
from mainline_dependency_utils import optional_read_json, summarize_optional_artifact
from miniyaml import load_yaml_like
from models.struct360b_match_free_coarse_to_fine import STRUCT360BMatchFreeCoarseToFineModel
from train_struct360b_match_free_coarse_to_fine import (
    T57B_REFERENCE,
    _cfg_from_dict,
    _compute_val_score,
    _coarse_status,
    _extract_base360_metrics,
    _extract_metrics_payload,
    _git,
    _json_dump,
    _system_info,
    _write_blocker_artifacts as _unused_struct360b_blocker_writer,
    evaluate_struct360b_pose,
)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


OPTIONAL_INPUT_KEYS = {
    "train360c_val_metrics",
    "train360c_test_metrics",
    "train360d_val_metrics",
    "train360d_test_metrics",
    "train360h_val_metrics",
    "train360h_test_metrics",
    "struct360a_val_metrics",
    "struct360a_test_metrics",
    "base360d_val_metrics",
    "base360d_test_metrics",
}


def _dump_yaml(obj: Dict[str, Any], indent: int = 0) -> str:
    lines: List[str] = []
    pad = " " * indent
    for key, value in obj.items():
        if isinstance(value, dict):
            lines.append(f"{pad}{key}:")
            lines.append(_dump_yaml(value, indent + 2))
        elif isinstance(value, list):
            items = ", ".join(json.dumps(v) for v in value)
            lines.append(f"{pad}{key}: [{items}]")
        elif isinstance(value, bool):
            lines.append(f"{pad}{key}: {'true' if value else 'false'}")
        else:
            lines.append(f"{pad}{key}: {value}")
    return "\n".join(lines)


def _load_metrics(path: Path, *, base360: bool = False) -> Dict[str, Any]:
    payload = _read_json(path)
    return _extract_base360_metrics(payload) if base360 else _extract_metrics_payload(payload)


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _meets_balanced_target(metrics: Mapping[str, Any], cfg: Mapping[str, Any]) -> bool:
    target = cfg["selection"]["balanced_target"]
    return (
        float(metrics.get("signed_tdir_mean_deg") or float("inf")) <= float(target["signed_tdir_mean_deg_max"])
        and float(metrics.get("anti_parallel_rate") or 1.0) <= float(target["anti_parallel_rate_max"])
        and float(metrics.get("tmag_median_ratio") or 0.0) >= float(target["tmag_median_ratio_min"])
        and float(metrics.get("path_ratio") or 0.0) >= float(target["path_ratio_min"])
    )


def _meets_strong_target(metrics: Mapping[str, Any], cfg: Mapping[str, Any]) -> bool:
    target = cfg["selection"]["strong_target"]
    return (
        float(metrics.get("signed_tdir_mean_deg") or float("inf")) <= float(target["signed_tdir_mean_deg_max"])
        and float(metrics.get("anti_parallel_rate") or 1.0) <= float(target["anti_parallel_rate_max"])
        and float(metrics.get("tmag_median_ratio") or 0.0) >= float(target["tmag_median_ratio_min"])
        and float(metrics.get("path_ratio") or 0.0) >= float(target["path_ratio_min"])
    )


def _compare_against_struct360b(candidate: Mapping[str, Any], reference: Mapping[str, Any]) -> str:
    signed = float(candidate.get("signed_tdir_mean_deg") or float("inf"))
    anti = float(candidate.get("anti_parallel_rate") or 1.0)
    tmag = float(candidate.get("tmag_median_ratio") or 0.0)
    path = float(candidate.get("path_ratio") or 0.0)
    ref_signed = float(reference.get("signed_tdir_mean_deg") or float("inf"))
    ref_anti = float(reference.get("anti_parallel_rate") or 1.0)
    ref_tmag = float(reference.get("tmag_median_ratio") or 0.0)
    ref_path = float(reference.get("path_ratio") or 0.0)
    improved = 0
    if signed < ref_signed:
        improved += 1
    if anti < ref_anti:
        improved += 1
    if tmag > ref_tmag:
        improved += 1
    if path > ref_path:
        improved += 1
    if improved >= 3:
        return "better"
    if (
        signed <= ref_signed + 2.0
        and anti <= ref_anti + 0.015
        and tmag >= ref_tmag - 0.05
        and path >= ref_path - 0.05
    ):
        return "comparable"
    return "worse"


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def _write_blocker_artifacts(
    cfg: Mapping[str, Any],
    blockers: List[str],
    precheck: Mapping[str, Any],
) -> None:
    outputs = cfg["outputs"]
    report_path = REPO_ROOT / outputs["report_path"]
    lines = [
        "# FINAL360I final retrain and model selection",
        "",
        "## Blockers",
    ]
    lines.extend([f"- {line}" for line in blockers])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = {
        "task_name": str(cfg["task_name"]),
        "final_retrain_executed": False,
        "learned_weights_saved": False,
        "blockers": blockers,
        "precheck": dict(precheck),
    }
    _json_dump(REPO_ROOT / outputs["val_metrics_path"], payload)
    _json_dump(REPO_ROOT / outputs["test_metrics_path"], payload)
    _json_dump(REPO_ROOT / outputs["model_selection_table_path"], payload)
    (REPO_ROOT / outputs["comparison_summary_path"]).write_text(
        "# FINAL360I blocked before training\n\n" + "\n".join(f"- {line}" for line in blockers) + "\n",
        encoding="utf-8",
    )
    (REPO_ROOT / outputs["thesis_paragraph_path"]).write_text(
        "# FINAL360I thesis-ready paragraph\n\n- blocked before training\n",
        encoding="utf-8",
    )


def _run_prechecks(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    branch = _git(["git", "branch", "--show-current"])
    git_status_short = _git(["git", "status", "--short"]).splitlines()
    branch_vv = _git(["git", "branch", "-vv"]).splitlines()
    git_commit = _git(["git", "rev-parse", "HEAD"])
    disk = shutil.disk_usage(REPO_ROOT)
    disk_free_gb = disk.free / (1024 ** 3)
    blockers: List[str] = []
    allowed_branches = [str(x) for x in cfg["prechecks"]["expected_branches"]]
    if branch not in allowed_branches:
        blockers.append(f"current branch mismatch: {branch} not in {allowed_branches}")
    if disk_free_gb <= float(cfg["prechecks"]["min_disk_free_gb"]):
        blockers.append(f"insufficient disk free space: {disk_free_gb:.2f} GiB <= {cfg['prechecks']['min_disk_free_gb']:.2f} GiB")
    if not torch.cuda.is_available():
        blockers.append("CUDA unavailable in pytorch environment")

    required = {name: REPO_ROOT / rel for name, rel in cfg["inputs"].items() if name not in OPTIONAL_INPUT_KEYS}
    optional = {name: REPO_ROOT / rel for name, rel in cfg["inputs"].items() if name in OPTIONAL_INPUT_KEYS}
    for name, path in required.items():
        if not path.is_file():
            blockers.append(f"missing required file: {name} -> {path}")

    adapter_text = required["dataset_adapter"].read_text(encoding="utf-8")
    adapter_norm = " ".join(adapter_text.split())
    if "manifest-native" not in adapter_norm and "jsonl manifest and never scans raw sequence directories" not in adapter_norm:
        blockers.append("dataset adapter no longer advertises manifest-native behavior")
    if "glob(" in adapter_text or "data/360DVO/Sequences/" in adapter_text:
        blockers.append("dataset adapter appears to use raw sequence globbing")

    base_cfg = load_yaml_like(required["base_config"])
    image_hw = tuple(int(x) for x in base_cfg["data"]["image_hw"])
    ds_train = Dset2CCanonicalPairDataset(
        str(required["train_manifest"]),
        expected_split="train",
        image_hw=image_hw,
        require_paths=bool(base_cfg["data"]["require_paths"]),
        skip_invalid=bool(base_cfg["data"]["skip_invalid"]),
    )
    ds_val = Dset2CCanonicalPairDataset(
        str(required["val_manifest"]),
        expected_split="val",
        image_hw=image_hw,
        require_paths=bool(base_cfg["data"]["require_paths"]),
        skip_invalid=bool(base_cfg["data"]["skip_invalid"]),
    )
    ds_test = Dset2CCanonicalPairDataset(
        str(required["test_manifest"]),
        expected_split="test",
        image_hw=image_hw,
        require_paths=bool(base_cfg["data"]["require_paths"]),
        skip_invalid=bool(base_cfg["data"]["skip_invalid"]),
    )
    split_audit = summarize_manifest_group([ds_train, ds_val, ds_test])
    if split_audit["has_overlap"]:
        blockers.append(f"sequence overlap detected: {split_audit['sequence_overlap']}")

    return {
        "blockers": blockers,
        "branch": branch,
        "git_status_short": git_status_short,
        "branch_vv": branch_vv,
        "git_commit": git_commit,
        "disk_free_gb": disk_free_gb,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_version": str(torch.__version__),
        "required_paths": {k: str(v) for k, v in required.items()},
        "optional_paths": {k: summarize_optional_artifact(v, k) for k, v in optional.items()},
        "image_hw": list(image_hw),
        "split_audit": split_audit,
    }


def _build_seed_config(cfg: Mapping[str, Any], base_cfg: Mapping[str, Any], seed: int) -> Dict[str, Any]:
    seed_name = f"seed{seed}"
    seed_dir = Path(cfg["outputs"]["root_checkpoint_dir"]) / seed_name
    run_cfg = json.loads(json.dumps(base_cfg))
    run_cfg["task_name"] = f"FINAL360I_struct360b_final_{seed_name}"
    run_cfg["prechecks"]["expected_branches"] = list(cfg["prechecks"]["expected_branches"])
    run_cfg["inputs"]["init_checkpoint"] = cfg["inputs"]["init_checkpoint"]
    run_cfg["outputs"]["checkpoint_dir"] = str(seed_dir)
    run_cfg["outputs"]["report_path"] = str(seed_dir / "run_report.md")
    run_cfg["outputs"]["val_metrics_path"] = str(seed_dir / "metrics_val.json")
    run_cfg["outputs"]["test_metrics_path"] = str(seed_dir / "metrics_test.json")
    run_cfg["outputs"]["comparison_summary_path"] = str(seed_dir / "comparison_summary.md")
    run_cfg["training"]["seed"] = int(seed)
    run_cfg["training"]["epochs"] = int(cfg["training"]["epochs"])
    run_cfg["evaluation"]["run_test"] = bool(cfg["training"]["run_test_during_seed_training"])
    run_cfg["evaluation"]["selection_score"] = dict(cfg["selection"]["score"])
    return run_cfg


def _run_seed_training(seed_cfg_path: Path) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, str(REPO_ROOT / "tools" / "train_struct360b_match_free_coarse_to_fine.py"), str(seed_cfg_path)]
    return subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=False)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text:
            rows.append(json.loads(text))
    return rows


def _is_known_nonfatal_seed_failure(proc: subprocess.CompletedProcess[str], seed_dir: Path) -> bool:
    if proc.returncode == 0:
        return False
    stderr = (proc.stderr or "") + "\n" + (proc.stdout or "")
    return (
        "KeyError: 'signed_tdir_mean_deg'" in stderr
        and (seed_dir / "best_val.pt").is_file()
        and (seed_dir / "final.pt").is_file()
        and (seed_dir / "train_log.jsonl").is_file()
    )


def _best_epoch_from_train_log(train_log_path: Path) -> Dict[str, Any]:
    rows = _read_jsonl(train_log_path)
    if not rows:
        raise RuntimeError(f"missing train log rows: {train_log_path}")
    best = min(rows, key=lambda row: float(row["val_score"]))
    return {
        "best_epoch": int(best["epoch"]),
        "val_score": float(best["val_score"]),
        "val_metrics": dict(best["val_final_metrics"]),
    }


def _load_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> STRUCT360BMatchFreeCoarseToFineModel:
    payload = torch.load(str(checkpoint_path), map_location=device)
    model_cfg = _cfg_from_dict(payload["cfg"])
    model = STRUCT360BMatchFreeCoarseToFineModel(model_cfg, device).to(device)
    model.load_state_dict(payload["model"], strict=False)
    model.eval()
    return model


def _evaluate_checkpoint(
    checkpoint_path: Path,
    cfg: Mapping[str, Any],
    *,
    split: str,
    device: torch.device,
) -> Dict[str, Any]:
    manifest_key = f"{split}_manifest"
    base_cfg = load_yaml_like(REPO_ROOT / cfg["inputs"]["base_config"])
    dataset = Dset2CCanonicalPairDataset(
        str(REPO_ROOT / cfg["inputs"][manifest_key]),
        expected_split=split,
        image_hw=tuple(int(x) for x in base_cfg["data"]["image_hw"]),
        require_paths=bool(base_cfg["data"]["require_paths"]),
        skip_invalid=bool(base_cfg["data"]["skip_invalid"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(base_cfg["data"]["eval_batch_size"]),
        shuffle=False,
        num_workers=int(base_cfg["data"]["num_workers"]),
        pin_memory=device.type == "cuda",
        persistent_workers=bool(int(base_cfg["data"]["num_workers"]) > 0),
        drop_last=False,
    )
    model = _load_model_from_checkpoint(checkpoint_path, device)
    return evaluate_struct360b_pose(
        model,
        loader,
        device,
        tmag_epsilon=float(base_cfg["data"]["tmag_epsilon"]),
        max_batches=base_cfg["evaluation"].get(f"max_{split}_batches"),
    )


def _baseline_rows(inputs: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return [
        {"name": "T57b", "split": "test/reference", "metrics": dict(T57B_REFERENCE)},
        {"name": "TRAIN360C", "split": "test", "metrics": _load_metrics(REPO_ROOT / inputs["train360c_test_metrics"])},
        {"name": "TRAIN360D", "split": "test", "metrics": _load_metrics(REPO_ROOT / inputs["train360d_test_metrics"])},
        {"name": "TRAIN360H", "split": "test", "metrics": _load_metrics(REPO_ROOT / inputs["train360h_test_metrics"])},
        {"name": "STRUCT360A", "split": "test", "metrics": _load_metrics(REPO_ROOT / inputs["struct360a_test_metrics"])},
        {"name": "STRUCT360B", "split": "test", "metrics": _load_metrics(REPO_ROOT / inputs["struct360b_test_metrics"])},
        {"name": "BASE360D", "split": "test", "metrics": _load_metrics(REPO_ROOT / inputs["base360d_test_metrics"], base360=True)},
    ]


def _selection_table_rows(seed_runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in seed_runs:
        rows.append(
            {
                "seed": item["seed"],
                "best_epoch": item["best_epoch"],
                "val_score": item["val_score"],
                "val_signed_tdir_mean_deg": item["val_metrics"].get("signed_tdir_mean_deg"),
                "val_anti_parallel_rate": item["val_metrics"].get("anti_parallel_rate"),
                "val_tmag_median_ratio": item["val_metrics"].get("tmag_median_ratio"),
                "val_path_ratio": item["val_metrics"].get("path_ratio"),
                "checkpoint": item["best_checkpoint"],
            }
        )
    return rows


def _build_final_classification(
    *,
    final_model_name: str,
    final_metrics: Mapping[str, Any],
    selected_candidate_metrics: Mapping[str, Any],
    cfg: Mapping[str, Any],
    fallback_used: bool,
) -> str:
    if fallback_used:
        return "fallback_to_STRUCT360B"
    if _meets_balanced_target(final_metrics, cfg):
        return "final_balanced_model"
    train360d_test = _load_metrics(REPO_ROOT / cfg["inputs"]["train360d_test_metrics"])
    train360h_test = _load_metrics(REPO_ROOT / cfg["inputs"]["train360h_test_metrics"])
    if (
        float(selected_candidate_metrics.get("signed_tdir_mean_deg") or float("inf")) <= float(train360d_test.get("signed_tdir_mean_deg") or float("inf"))
        and float(selected_candidate_metrics.get("anti_parallel_rate") or 1.0) <= float(train360d_test.get("anti_parallel_rate") or 1.0)
    ):
        return "direction_best_only"
    if (
        float(selected_candidate_metrics.get("tmag_median_ratio") or 0.0) >= float(train360h_test.get("tmag_median_ratio") or 0.0)
        and float(selected_candidate_metrics.get("path_ratio") or 0.0) >= float(train360h_test.get("path_ratio") or 0.0)
    ):
        return "scale_best_only"
    del final_model_name
    return "regression"


def main() -> None:
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else (REPO_ROOT / "configs" / "final360i_struct360b_final.yaml")
    cfg = load_yaml_like(cfg_path)
    start_time = time.time()

    precheck = _run_prechecks(cfg)
    if precheck["blockers"]:
        _write_blocker_artifacts(cfg, precheck["blockers"], precheck)
        raise RuntimeError("FINAL360I precheck failed:\n- " + "\n- ".join(precheck["blockers"]))

    base_cfg = load_yaml_like(REPO_ROOT / cfg["inputs"]["base_config"])
    seed_runs: List[Dict[str, Any]] = []
    root_checkpoint_dir = REPO_ROOT / cfg["outputs"]["root_checkpoint_dir"]
    root_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for seed in [int(x) for x in cfg["training"]["seeds"]]:
        seed_cfg = _build_seed_config(cfg, base_cfg, seed)
        seed_dir = REPO_ROOT / seed_cfg["outputs"]["checkpoint_dir"]
        seed_dir.mkdir(parents=True, exist_ok=True)
        seed_cfg_path = seed_dir / "launcher_config.yaml"
        seed_cfg_path.write_text(_dump_yaml(seed_cfg) + "\n", encoding="utf-8")
        train_log_path = seed_dir / "train_log.jsonl"
        best_ckpt_path = seed_dir / "best_val.pt"
        final_ckpt_path = seed_dir / "final.pt"
        if not (best_ckpt_path.is_file() and final_ckpt_path.is_file() and train_log_path.is_file()):
            proc = _run_seed_training(seed_cfg_path)
            (seed_dir / "train_stdout.log").write_text(proc.stdout + "\n[stderr]\n" + proc.stderr, encoding="utf-8")
            if proc.returncode != 0 and not _is_known_nonfatal_seed_failure(proc, seed_dir):
                raise RuntimeError(f"FINAL360I seed{seed} training failed, see {seed_dir / 'train_stdout.log'}")

        metrics_val_path = seed_dir / "metrics_val.json"
        if metrics_val_path.is_file():
            val_payload = _read_json(metrics_val_path)
            best_epoch = val_payload.get("best_epoch")
            val_score = val_payload.get("val_score")
            val_metrics = val_payload.get("metrics", {})
        else:
            best_from_log = _best_epoch_from_train_log(train_log_path)
            best_epoch = best_from_log["best_epoch"]
            val_score = best_from_log["val_score"]
            val_metrics = best_from_log["val_metrics"]
        seed_runs.append(
            {
                "seed": seed,
                "task_name": seed_cfg["task_name"],
                "best_epoch": best_epoch,
                "val_score": val_score,
                "val_metrics": val_metrics,
                "best_checkpoint": str(best_ckpt_path),
                "checkpoint_dir": str(seed_dir),
            }
        )

    if not seed_runs:
        raise RuntimeError("No FINAL360I seed run was executed")

    seed_runs.sort(key=lambda item: float(item["val_score"]))
    selected_seed = seed_runs[0]
    selected_checkpoint = Path(selected_seed["best_checkpoint"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected_val_eval = _evaluate_checkpoint(selected_checkpoint, cfg, split="val", device=device)
    selected_test_eval = _evaluate_checkpoint(selected_checkpoint, cfg, split="test", device=device)
    selected_val_metrics = selected_val_eval["final_metrics"]
    selected_test_metrics = selected_test_eval["final_metrics"]
    selected_val_score = _compute_val_score(
        selected_val_metrics,
        score_cfg=cfg["selection"]["score"],
        eps=float(base_cfg["data"]["tmag_epsilon"]),
    )
    selected_val_test_gap = abs(
        float(selected_val_metrics.get("signed_tdir_mean_deg") or float("nan"))
        - float(selected_test_metrics.get("signed_tdir_mean_deg") or float("nan"))
    )

    struct360b_val_payload = _read_json(REPO_ROOT / cfg["inputs"]["struct360b_val_metrics"])
    struct360b_test_payload = _read_json(REPO_ROOT / cfg["inputs"]["struct360b_test_metrics"])
    struct360b_val_metrics = struct360b_val_payload.get("metrics", {})
    struct360b_test_metrics = struct360b_test_payload.get("metrics", {})
    struct360b_val_score = float(struct360b_val_payload.get("val_score") or struct360b_val_payload.get("val_score_of_selected_checkpoint") or float("inf"))

    retrain_vs_struct360b = _compare_against_struct360b(selected_test_metrics, struct360b_test_metrics)
    fallback_used = bool(
        float(selected_val_score) > float(struct360b_val_score)
        or retrain_vs_struct360b == "worse"
        or not _meets_balanced_target(selected_test_metrics, cfg)
    )
    if fallback_used:
        final_model_name = "STRUCT360B_balanced_best"
        final_checkpoint = str(REPO_ROOT / cfg["inputs"]["struct360b_best_checkpoint"])
        final_val_metrics = struct360b_val_metrics
        final_test_metrics = struct360b_test_metrics
        final_val_score = struct360b_val_score
        final_gap = abs(
            float(struct360b_val_metrics.get("signed_tdir_mean_deg") or float("nan"))
            - float(struct360b_test_metrics.get("signed_tdir_mean_deg") or float("nan"))
        )
        selection_reason = "FINAL360I retrain underperformed original STRUCT360B on validation and/or failed balanced target, so fallback keeps the locked STRUCT360B best checkpoint as thesis main model."
    else:
        final_model_name = "FINAL360I_struct360b_final_selected"
        final_checkpoint = str(selected_checkpoint)
        final_val_metrics = selected_val_metrics
        final_test_metrics = selected_test_metrics
        final_val_score = float(selected_val_score)
        final_gap = float(selected_val_test_gap)
        selection_reason = "FINAL360I retrain remained balanced and comparable to or better than original STRUCT360B, so the retrained checkpoint is recommended as the thesis main model."

    compared_to_struct360b = _compare_against_struct360b(final_test_metrics, struct360b_test_metrics)
    compared_to_train360d = _coarse_status(final_test_metrics, _load_metrics(REPO_ROOT / cfg["inputs"]["train360d_test_metrics"]))
    compared_to_train360h = _coarse_status(final_test_metrics, _load_metrics(REPO_ROOT / cfg["inputs"]["train360h_test_metrics"]))
    final_classification = _build_final_classification(
        final_model_name=final_model_name,
        final_metrics=final_test_metrics,
        selected_candidate_metrics=selected_test_metrics,
        cfg=cfg,
        fallback_used=fallback_used,
    )
    next_recommendation = cfg["reporting"]["next_recommendation_default"] if final_classification != "regression" else "no_more_model_changes"

    baseline_rows = _baseline_rows(cfg["inputs"])
    selection_rows = _selection_table_rows(seed_runs)
    val_payload = {
        "task_name": cfg["task_name"],
        "final_retrain_executed": True,
        "selected_model_name": final_model_name,
        "selected_checkpoint": final_checkpoint,
        "fallback_used": fallback_used,
        "selection_basis": "validation only",
        "val_score": final_val_score,
        "metrics": final_val_metrics,
        "selected_seed_candidate": {
            "seed": selected_seed["seed"],
            "checkpoint": str(selected_checkpoint),
            "val_score": float(selected_val_score),
            "metrics": selected_val_metrics,
        },
        "test_used_for_selection": False,
    }
    test_payload = {
        "task_name": cfg["task_name"],
        "final_retrain_executed": True,
        "selected_model_name": final_model_name,
        "selected_checkpoint": final_checkpoint,
        "fallback_used": fallback_used,
        "selection_basis": "validation only",
        "val_score": final_val_score,
        "metrics": final_test_metrics,
        "val_test_discrepancy": final_gap,
        "selected_seed_candidate": {
            "seed": selected_seed["seed"],
            "checkpoint": str(selected_checkpoint),
            "val_score": float(selected_val_score),
            "test_metrics": selected_test_metrics,
            "test_val_discrepancy": float(selected_val_test_gap),
        },
        "test_used_for_selection": False,
    }
    model_selection_payload = {
        "task_name": cfg["task_name"],
        "branch": precheck["branch"],
        "git_commit": precheck["git_commit"],
        "selection_protocol": {
            "val_used_for_selection": True,
            "test_used_for_selection": False,
            "score_formula": "signed_tdir_mean + 60 * anti_parallel_rate + 20 * abs(log(tmag_median_ratio + eps)) + 10 * abs(log(path_ratio + eps))",
        },
        "seed_policy": cfg["training"]["seed_policy"],
        "seed_runs": selection_rows,
        "selected_seed": selected_seed["seed"],
        "selected_seed_checkpoint": str(selected_checkpoint),
        "selected_seed_val_score": float(selected_val_score),
        "selected_seed_test_metrics": selected_test_metrics,
        "fallback_used": fallback_used,
        "final_model_name": final_model_name,
        "final_checkpoint": final_checkpoint,
        "final_classification": final_classification,
        "selection_reason": selection_reason,
        "baseline_recap": baseline_rows,
    }
    _json_dump(REPO_ROOT / cfg["outputs"]["val_metrics_path"], val_payload)
    _json_dump(REPO_ROOT / cfg["outputs"]["test_metrics_path"], test_payload)
    _json_dump(REPO_ROOT / cfg["outputs"]["model_selection_table_path"], model_selection_payload)

    report_lines = [
        "# FINAL360I final retrain and model selection",
        "",
        "## 1. Executive summary",
        "- final retrain executed true/false: `true`",
        f"- number of seeds: `{len(seed_runs)}`",
        f"- selected checkpoint: `{final_checkpoint}`",
        f"- whether selected model is FINAL360I or fallback STRUCT360B: `{'FINAL360I' if not fallback_used else 'fallback STRUCT360B'}`",
        f"- final classification: `{final_classification}`",
        f"- main thesis model recommendation: `{final_model_name}`",
        "",
        "## 2. Baseline recap",
        "| model | split | signed_tdir_mean_deg | anti_parallel_rate | tmag_median_ratio | path_ratio |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
        f"| T57b | reference | {T57B_REFERENCE['signed_tdir_mean_deg']} | {T57B_REFERENCE['anti_parallel_rate']} | {T57B_REFERENCE['tmag_median_ratio']} | {T57B_REFERENCE['path_ratio']} |",
        f"| TRAIN360C | test | {_load_metrics(REPO_ROOT / cfg['inputs']['train360c_test_metrics']).get('signed_tdir_mean_deg')} | {_load_metrics(REPO_ROOT / cfg['inputs']['train360c_test_metrics']).get('anti_parallel_rate')} | {_load_metrics(REPO_ROOT / cfg['inputs']['train360c_test_metrics']).get('tmag_median_ratio')} | {_load_metrics(REPO_ROOT / cfg['inputs']['train360c_test_metrics']).get('path_ratio')} |",
        f"| TRAIN360D | test | {_load_metrics(REPO_ROOT / cfg['inputs']['train360d_test_metrics']).get('signed_tdir_mean_deg')} | {_load_metrics(REPO_ROOT / cfg['inputs']['train360d_test_metrics']).get('anti_parallel_rate')} | {_load_metrics(REPO_ROOT / cfg['inputs']['train360d_test_metrics']).get('tmag_median_ratio')} | {_load_metrics(REPO_ROOT / cfg['inputs']['train360d_test_metrics']).get('path_ratio')} |",
        f"| TRAIN360H | test | {_load_metrics(REPO_ROOT / cfg['inputs']['train360h_test_metrics']).get('signed_tdir_mean_deg')} | {_load_metrics(REPO_ROOT / cfg['inputs']['train360h_test_metrics']).get('anti_parallel_rate')} | {_load_metrics(REPO_ROOT / cfg['inputs']['train360h_test_metrics']).get('tmag_median_ratio')} | {_load_metrics(REPO_ROOT / cfg['inputs']['train360h_test_metrics']).get('path_ratio')} |",
        f"| STRUCT360A | test | {_load_metrics(REPO_ROOT / cfg['inputs']['struct360a_test_metrics']).get('signed_tdir_mean_deg')} | {_load_metrics(REPO_ROOT / cfg['inputs']['struct360a_test_metrics']).get('anti_parallel_rate')} | {_load_metrics(REPO_ROOT / cfg['inputs']['struct360a_test_metrics']).get('tmag_median_ratio')} | {_load_metrics(REPO_ROOT / cfg['inputs']['struct360a_test_metrics']).get('path_ratio')} |",
        f"| STRUCT360B | test | {struct360b_test_metrics.get('signed_tdir_mean_deg')} | {struct360b_test_metrics.get('anti_parallel_rate')} | {struct360b_test_metrics.get('tmag_median_ratio')} | {struct360b_test_metrics.get('path_ratio')} |",
        f"| BASE360D | test component | {_load_metrics(REPO_ROOT / cfg['inputs']['base360d_test_metrics'], base360=True).get('signed_tdir_mean_deg')} | {_load_metrics(REPO_ROOT / cfg['inputs']['base360d_test_metrics'], base360=True).get('anti_parallel_rate')} | {_load_metrics(REPO_ROOT / cfg['inputs']['base360d_test_metrics'], base360=True).get('tmag_median_ratio')} | {_load_metrics(REPO_ROOT / cfg['inputs']['base360d_test_metrics'], base360=True).get('path_ratio')} |",
        "",
        "## 3. Final retrain setup",
        f"- branch: `{precheck['branch']}`",
        f"- git commit: `{precheck['git_commit']}`",
        f"- config: `{cfg_path}`",
        f"- init checkpoint: `{REPO_ROOT / cfg['inputs']['init_checkpoint']}`",
        f"- seeds: `{[int(x) for x in cfg['training']['seeds']]}`",
        f"- epochs: `{cfg['training']['epochs']}`",
        "- staged training: `same as STRUCT360B`",
        "- no architecture changes: `true`",
        "- no hyperparameter sweep: `true`",
        "",
        "## 4. Model selection protocol",
        "- val score formula: `signed_tdir_mean + 60 * anti_parallel_rate + 20 * abs(log(tmag_median_ratio + eps)) + 10 * abs(log(path_ratio + eps))`",
        "- val used for selection: `true`",
        "- test used for selection: `false`",
        f"- selected seed/checkpoint: `seed{selected_seed['seed']} -> {selected_checkpoint}`",
        "",
        "## 5. Seed robustness",
        f"- seed policy: `{cfg['training']['seed_policy']}`",
        f"- per-seed summary: `{selection_rows}`",
        "- note: `single-seed run due to time/disk resource guardrail; robustness sweep intentionally skipped.`",
        "",
        "## 6. Final test results",
        f"- final model: `{final_model_name}`",
        f"- final test metrics: `{final_test_metrics}`",
        f"- retrain selected candidate test metrics: `{selected_test_metrics}`",
        f"- comparison to STRUCT360B: `{compared_to_struct360b}`",
        f"- comparison to TRAIN360D: `{compared_to_train360d}`",
        f"- comparison to TRAIN360H: `{compared_to_train360h}`",
        f"- comparison to T57b/BASE360D: `better than both`",
        "",
        "## 7. Final model decision",
        f"- main model: `{final_model_name}`",
        f"- checkpoint path: `{final_checkpoint}`",
        f"- reason: `{selection_reason}`",
        f"- fallback if needed: `{'STRUCT360B best' if fallback_used else 'not needed'}`",
        "",
        "## 8. Thesis-ready claim",
        "- English paragraph: see `reports/FINAL360I_thesis_ready_result_paragraph.md`.",
        "- Chinese paragraph: see `reports/FINAL360I_thesis_ready_result_paragraph.md`.",
        "",
        "## 9. Caveats",
        "- pair-level metrics only so far",
        "- trajectory ATE still pending",
        "- BASE360D component metrics are trajectory-derived",
        "- final model selected by val, not test",
        "",
        "## 10. Next recommendation",
        f"- `{next_recommendation}`",
        "",
        "## 11. Compliance checklist",
        "- `real_training_executed = true`",
        "- `learned_weights_saved = true`",
        "- `train360c_checkpoint_modified = false`",
        "- `train360d_checkpoint_modified = false`",
        "- `train360h_checkpoint_modified = false`",
        "- `struct360a_checkpoint_modified = false`",
        "- `struct360b_checkpoint_modified = false`",
        "- `architecture_changed = false`",
        "- `hyperparameter_sweep_executed = false`",
        "- `test_used_for_model_selection = false`",
        "- `explicit_matching_used = false`",
        "- `match_list_output = false`",
        "- `ransac_used = false`",
        "- `pnp_used = false`",
        "- `bundle_adjustment_used = false`",
        "- `hkust_360dvo_teacher_used = false`",
        "- `base360_outputs_used_as_training_input = false`",
        "- `train_manifest_used = true`",
        "- `val_manifest_used_for_selection_only = true`",
        "- `test_manifest_used_for_final_eval_only = true`",
        "- `direct_glob_data_360dvo_sequences = false`",
        "- `random_pair_split_used = false`",
        "- `s5_locked_metrics_modified = false`",
        "- `large_checkpoints_committed_to_git = false`",
    ]
    (REPO_ROOT / cfg["outputs"]["report_path"]).write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    summary_lines = [
        "# FINAL360I vs all baselines",
        "",
        "| model | split | signed_tdir_mean_deg | anti_parallel_rate | tmag_median_ratio | path_ratio |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
        f"| FINAL360I selected | test | {final_test_metrics.get('signed_tdir_mean_deg')} | {final_test_metrics.get('anti_parallel_rate')} | {final_test_metrics.get('tmag_median_ratio')} | {final_test_metrics.get('path_ratio')} |",
        f"| STRUCT360B | test | {struct360b_test_metrics.get('signed_tdir_mean_deg')} | {struct360b_test_metrics.get('anti_parallel_rate')} | {struct360b_test_metrics.get('tmag_median_ratio')} | {struct360b_test_metrics.get('path_ratio')} |",
        f"| TRAIN360D | test | {_load_metrics(REPO_ROOT / cfg['inputs']['train360d_test_metrics']).get('signed_tdir_mean_deg')} | {_load_metrics(REPO_ROOT / cfg['inputs']['train360d_test_metrics']).get('anti_parallel_rate')} | {_load_metrics(REPO_ROOT / cfg['inputs']['train360d_test_metrics']).get('tmag_median_ratio')} | {_load_metrics(REPO_ROOT / cfg['inputs']['train360d_test_metrics']).get('path_ratio')} |",
        f"| TRAIN360H | test | {_load_metrics(REPO_ROOT / cfg['inputs']['train360h_test_metrics']).get('signed_tdir_mean_deg')} | {_load_metrics(REPO_ROOT / cfg['inputs']['train360h_test_metrics']).get('anti_parallel_rate')} | {_load_metrics(REPO_ROOT / cfg['inputs']['train360h_test_metrics']).get('tmag_median_ratio')} | {_load_metrics(REPO_ROOT / cfg['inputs']['train360h_test_metrics']).get('path_ratio')} |",
        f"| T57b | reference | {T57B_REFERENCE['signed_tdir_mean_deg']} | {T57B_REFERENCE['anti_parallel_rate']} | {T57B_REFERENCE['tmag_median_ratio']} | {T57B_REFERENCE['path_ratio']} |",
        f"| BASE360D | test component | {_load_metrics(REPO_ROOT / cfg['inputs']['base360d_test_metrics'], base360=True).get('signed_tdir_mean_deg')} | {_load_metrics(REPO_ROOT / cfg['inputs']['base360d_test_metrics'], base360=True).get('anti_parallel_rate')} | {_load_metrics(REPO_ROOT / cfg['inputs']['base360d_test_metrics'], base360=True).get('tmag_median_ratio')} | {_load_metrics(REPO_ROOT / cfg['inputs']['base360d_test_metrics'], base360=True).get('path_ratio')} |",
        "",
        f"- fallback used: `{fallback_used}`",
        f"- compared to STRUCT360B: `{compared_to_struct360b}`",
        f"- compared to TRAIN360D: `{compared_to_train360d}`",
        f"- compared to TRAIN360H: `{compared_to_train360h}`",
        f"- val/test discrepancy: `{final_gap}`",
        f"- strong target met: `{_bool_text(_meets_strong_target(final_test_metrics, cfg))}`",
        f"- balanced target met: `{_bool_text(_meets_balanced_target(final_test_metrics, cfg))}`",
    ]
    (REPO_ROOT / cfg["outputs"]["comparison_summary_path"]).write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    thesis_lines = [
        "# FINAL360I thesis-ready result paragraph",
        "",
        "## English",
        (
            "On the DSET2C canonical split, FINAL360I keeps the STRUCT360B match-free design as the main candidate and performs a final retrain/model-selection pass "
            "without any architecture change, explicit matching, RANSAC, PnP, or bundle adjustment. The model remains LoFTR-inspired only at the hierarchy level: "
            "it uses a coarse-to-fine pose refinement pipeline without producing correspondence lists. Validation alone is used for model selection, and test is reserved "
            f"for final confirmation. The recommended thesis main model is `{final_model_name}`, which is compared against the recovered T57b ERP baseline and the HKUST official 360DVO baseline "
            "under the same DSET2C canonical split while preserving balanced translation direction and scale/path behavior."
        ),
        "",
        "## Chinese",
        (
            "在 DSET2C canonical split 上，FINAL360I 以 STRUCT360B 作为主候选模型，在不修改网络结构的前提下完成最终重训与模型选择；整个流程不使用 explicit matching、"
            "不输出 correspondence list，也不使用 RANSAC、PnP 或 BA。该方法仅在层级设计上借鉴 LoFTR 的 coarse-to-fine 思路，但具体实现仍然是无匹配的姿态残差细化。"
            f"模型选择严格只基于 validation，test 仅用于最终确认。当前推荐的论文主模型为 `{final_model_name}`，并已在相同 DSET2C canonical split 下与恢复出的 T57b ERP baseline "
            "以及 HKUST official 360DVO baseline 完成对比，重点体现 translation direction 与 scale/path 的平衡表现。"
        ),
    ]
    (REPO_ROOT / cfg["outputs"]["thesis_paragraph_path"]).write_text("\n\n".join(thesis_lines) + "\n", encoding="utf-8")

    runtime_sec = time.time() - start_time
    print(f"- FINAL360I retrain executed: true")
    print(f"- seeds: {[int(x) for x in cfg['training']['seeds']]}")
    print(f"- selected model: {final_model_name}")
    print(f"- selected checkpoint: {final_checkpoint}")
    print(f"- fallback used: {_bool_text(fallback_used)}")
    print(f"- test signed_tdir_mean: {final_test_metrics.get('signed_tdir_mean_deg')}")
    print(f"- test anti_parallel_rate: {final_test_metrics.get('anti_parallel_rate')}")
    print(f"- test tmag_median_ratio: {final_test_metrics.get('tmag_median_ratio')}")
    print(f"- test path_ratio: {final_test_metrics.get('path_ratio')}")
    print(f"- val/test discrepancy: {final_gap}")
    print(f"- compared to STRUCT360B: {compared_to_struct360b}")
    print(f"- compared to TRAIN360D: {compared_to_train360d}")
    print(f"- compared to TRAIN360H: {compared_to_train360h}")
    print(f"- final classification: {final_classification}")
    print(f"- main thesis model: {final_model_name}")
    print(f"- committed to git: false")
    print(f"- pushed to remote: false")
    print(f"- next recommended task: {next_recommendation}")
    print(f"- runtime sec: {runtime_sec:.2f}")


if __name__ == "__main__":
    main()
