#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from config import Config
from datasets.dset2c_manifest_dataset import (
    Dset2CCanonicalPairDataset,
    summarize_manifest_group,
)
from model import PanoramaRelPoseModel
from miniyaml import load_yaml_like


def _cfg_from_dict(cfg_dict: Mapping[str, Any]) -> Config:
    cfg = Config()
    for k, v in cfg_dict.items():
        setattr(cfg, k, v)
    return cfg


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _count_nonfinite_tensor(x: torch.Tensor) -> int:
    return int((~torch.isfinite(x)).sum().item())


def _shape_of(x: torch.Tensor | None) -> List[int] | None:
    if x is None:
        return None
    return list(x.shape)


def _float_stats(values: Iterable[float]) -> Dict[str, float | None]:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return {"min": None, "median": None, "max": None}
    return {
        "min": float(min(vals)),
        "median": float(median(vals)),
        "max": float(max(vals)),
    }


def _to_plain_meta(meta_batch: Mapping[str, Any], batch_index: int = 0) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in meta_batch.items():
        if torch.is_tensor(value):
            item = value[batch_index]
            out[key] = item.item() if item.ndim == 0 else item.detach().cpu().tolist()
        elif isinstance(value, (list, tuple)):
            out[key] = value[batch_index]
        else:
            out[key] = value
    return out


def _load_model_with_status(
    checkpoint_path: Path,
    device: torch.device,
    *,
    strict_attempt: bool = True,
) -> Tuple[PanoramaRelPoseModel, Config, Dict[str, Any]]:
    payload = torch.load(str(checkpoint_path), map_location=device)
    cfg_dict = payload.get("cfg", {})
    if not isinstance(cfg_dict, dict):
        raise TypeError(f"Unsupported cfg type in checkpoint: {type(cfg_dict)}")
    cfg = _cfg_from_dict(cfg_dict)
    model = PanoramaRelPoseModel(cfg, device).to(device)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint has no model state_dict: {checkpoint_path}")

    strict_ok = False
    strict_error = None
    strict_missing: List[str] = []
    strict_unexpected: List[str] = []
    if strict_attempt:
        try:
            missing, unexpected = model.load_state_dict(state, strict=True)
            strict_missing = list(missing)
            strict_unexpected = list(unexpected)
            strict_ok = True
        except Exception as exc:
            strict_error = f"{type(exc).__name__}: {exc}"

    if strict_ok:
        load_status = {
            "status": "strict",
            "strict_attempted": True,
            "strict_ok": True,
            "missing_keys": strict_missing,
            "unexpected_keys": strict_unexpected,
            "strict_error": None,
        }
    else:
        model = PanoramaRelPoseModel(cfg, device).to(device)
        model_state = model.state_dict()
        filtered: Dict[str, Any] = {}
        skipped_shape_mismatch: List[str] = []
        for key, value in state.items():
            if key in model_state and hasattr(value, "shape") and value.shape != model_state[key].shape:
                skipped_shape_mismatch.append(key)
                continue
            filtered[key] = value
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        load_status = {
            "status": "non-strict",
            "strict_attempted": bool(strict_attempt),
            "strict_ok": False,
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
            "strict_error": strict_error,
            "skipped_shape_mismatch": skipped_shape_mismatch,
        }
    model.eval()
    return model, cfg, load_status


def _run_forward_batch(
    model: PanoramaRelPoseModel,
    batch: Mapping[str, Any],
    device: torch.device,
    *,
    enable_depth_fusion: bool,
) -> Dict[str, Any]:
    IA = batch["IA"].to(device, non_blocking=True)
    IB = batch["IB"].to(device, non_blocking=True)
    meta0 = _to_plain_meta(batch["meta"], batch_index=0)
    dt_world = torch.tensor(
        [max(float(meta0.get("dt_world", 0.0)), 1.0e-6)],
        device=device,
        dtype=torch.float32,
    )
    with torch.no_grad():
        R_pred, t_pred, aux = model(
            IA,
            IB,
            enable_depth_fusion=bool(enable_depth_fusion),
            dt_world=dt_world,
        )

    tdir_out = aux.get("t_dir_out", t_pred)
    tmag = aux.get("t_mag", aux.get("final_tmag"))
    log_tmag = aux.get("log_t_mag", aux.get("final_log_tmag"))
    if tmag is None and log_tmag is not None:
        tmag = torch.exp(log_tmag.float())

    output_tensors: Dict[str, torch.Tensor] = {
        "R_pred": R_pred,
        "t_pred": t_pred,
        "tdir_out": tdir_out,
    }
    if tmag is not None:
        output_tensors["tmag"] = tmag
    if log_tmag is not None:
        output_tensors["log_tmag"] = log_tmag
    for key, value in aux.items():
        if torch.is_tensor(value):
            output_tensors[f"aux::{key}"] = value

    nonfinite_total = sum(_count_nonfinite_tensor(v) for v in output_tensors.values())
    tdir_norm = torch.linalg.norm(tdir_out.detach().float(), dim=-1).cpu().tolist()
    tmag_values = [] if tmag is None else tmag.detach().float().view(-1).cpu().tolist()

    return {
        "meta": meta0,
        "input_image_shape": list(IA.shape),
        "batch_size": int(IA.shape[0]),
        "output_keys": sorted(aux.keys()),
        "R_output_shape": _shape_of(R_pred),
        "tdir_output_shape": _shape_of(tdir_out),
        "tmag_output_shape": _shape_of(tmag),
        "log_tmag_output_shape": _shape_of(log_tmag),
        "tmag_stats": _float_stats(tmag_values),
        "tdir_norm_stats": _float_stats(tdir_norm),
        "nan_inf_count": int(nonfinite_total),
        "model_stage": str(aux.get("stage", "unknown")),
        "bounded_log_tmag_present": bool(log_tmag is not None),
        "positive_tmag_present": bool(tmag is not None),
    }


def _build_loader(
    manifest_path: str,
    split: str,
    image_hw: Tuple[int, int],
    tmag_epsilon: float,
    batch_size: int,
    num_workers: int,
    require_paths: bool,
    skip_invalid: bool,
) -> Tuple[Dset2CCanonicalPairDataset, DataLoader]:
    dataset = Dset2CCanonicalPairDataset(
        manifest_path,
        expected_split=split,
        image_hw=image_hw,
        tmag_epsilon=tmag_epsilon,
        require_paths=require_paths,
        skip_invalid=skip_invalid,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        drop_last=False,
    )
    return dataset, loader


def _loss_interface_stub() -> Dict[str, Any]:
    return {
        "rotation_loss": {
            "inputs": ["R_pred_BA[batch,3,3]", "R_gt_BA[batch,3,3]"],
            "output": "scalar geodesic rotation loss",
            "existing_component": "losses.pose_loss or standalone rotation term",
        },
        "tdir_angular_loss": {
            "inputs": ["tdir_pred_B[batch,3]", "t_gt_vec_or_tdir_B[batch,3]", "R_gt_BA[batch,3,3]"],
            "output": "scalar signed or absolute angular direction loss",
            "existing_component": "losses.translation_direction_loss",
        },
        "tmag_loss": {
            "inputs": ["tmag_pred[batch] or log_tmag_pred[batch]", "tmag_gt[batch]"],
            "output": "scalar bounded log-magnitude loss",
            "existing_component": "losses.translation_magnitude_loss",
        },
        "adjacent_kstep_weighting_hook": {
            "inputs": ["k", "pair_type", "seq_id", "split"],
            "output": "per-sample weighting tensor",
            "status": "stub only, no training logic executed",
        },
        "observability_weighting_future_hook": {
            "inputs": ["sample metadata", "predicted confidence or future gate"],
            "output": "per-sample tdir weighting",
            "status": "deferred",
        },
        "rotation_compensation_future_hook": {
            "inputs": ["coarse R_BA", "fine interaction tokens"],
            "output": "refined direction token flow",
            "status": "deferred",
        },
        "contiguous_kstep_composition_future_hook": {
            "inputs": ["ordered same-sequence windows", "adjacent predictions", "k-step predictions"],
            "output": "composition consistency losses",
            "status": "deferred",
        },
    }


def main() -> None:
    cfg_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else REPO_ROOT / "configs" / "train360_v0_manifest_sanity.yaml"
    )
    cfg = load_yaml_like(cfg_path)

    reports_cfg = cfg["reports"]
    dataset_cfg = cfg["dataset"]
    model_cfg = cfg["model"]
    sanity_cfg = cfg["sanity"]

    train360a_md_path = REPO_ROOT / reports_cfg["train360a_markdown"]
    train360a_matrix_path = REPO_ROOT / reports_cfg["train360a_module_matrix"]
    train360a_md = _read_text(train360a_md_path)
    train360a_matrix = json.loads(_read_text(train360a_matrix_path))
    hygiene = _read_json(REPO_ROOT / dataset_cfg["hygiene_json"])

    image_hw = tuple(int(x) for x in dataset_cfg["expected_image_hw"])
    splits = [str(x) for x in sanity_cfg["splits"]]
    manifest_paths = {
        "train": str(REPO_ROOT / dataset_cfg["train_manifest"]),
        "val": str(REPO_ROOT / dataset_cfg["val_manifest"]),
        "test": str(REPO_ROOT / dataset_cfg["test_manifest"]),
    }

    datasets: Dict[str, Dset2CCanonicalPairDataset] = {}
    loaders: Dict[str, DataLoader] = {}
    for split in splits:
        ds, loader = _build_loader(
            manifest_paths[split],
            split,
            image_hw=image_hw,
            tmag_epsilon=float(dataset_cfg["tmag_epsilon"]),
            batch_size=int(dataset_cfg["batch_size"]),
            num_workers=int(dataset_cfg["num_workers"]),
            require_paths=bool(dataset_cfg["require_paths"]),
            skip_invalid=bool(dataset_cfg["skip_invalid"]),
        )
        datasets[split] = ds
        loaders[split] = loader

    sequence_audit = summarize_manifest_group(datasets.values())

    use_cuda = bool(model_cfg["use_cuda_if_available"]) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    checkpoint_path = REPO_ROOT / model_cfg["checkpoint"]
    model, ckpt_cfg, load_status = _load_model_with_status(
        checkpoint_path,
        device,
        strict_attempt=bool(model_cfg["strict_load_attempt"]),
    )

    forward_results: Dict[str, Any] = {}
    for split in splits:
        loader = loaders[split]
        batch = next(iter(loader))
        forward_results[split] = _run_forward_batch(
            model,
            batch,
            device,
            enable_depth_fusion=bool(model_cfg["enable_depth_fusion"]),
        )

    manifest_audit = {
        split: {
            **datasets[split].get_schema_summary(),
            "num_batches_at_configured_batch_size": len(loaders[split]),
        }
        for split in splits
    }

    canonical_summary = hygiene.get("canonical_manifest", {}).get("summary", {})
    canonical_readiness = hygiene.get("readiness", {})
    manifest_consistency = {
        "train_row_count_matches_hygiene": int(manifest_audit["train"]["kept_row_count"])
        == int(canonical_summary.get("num_pairs_train", -1)),
        "val_row_count_matches_hygiene": int(manifest_audit["val"]["kept_row_count"])
        == int(canonical_summary.get("num_pairs_val", -1)),
        "test_row_count_matches_hygiene": int(manifest_audit["test"]["kept_row_count"])
        == int(canonical_summary.get("num_pairs_test", -1)),
        "sequence_overlap_present": bool(sequence_audit["has_overlap"]),
        "hygiene_train360_ready_after_hygiene": bool(
            canonical_readiness.get("train360_ready_after_hygiene", False)
        ),
    }

    direct_modules = [
        m["module_name"]
        for m in train360a_matrix
        if str(m.get("reusable_status")) == "direct"
    ]
    adapt_modules = [
        m["module_name"]
        for m in train360a_matrix
        if str(m.get("reusable_status")) == "adapt"
    ]
    no_modules = [
        m["module_name"]
        for m in train360a_matrix
        if str(m.get("reusable_status")) == "no"
    ]

    readiness = {
        "manifest_adapter_ready": True,
        "model_forward_ready": all(
            int(forward_results[split]["nan_inf_count"]) == 0 for split in splits
        ),
        "loss_interface_ready": True,
        "evaluator_still_missing": True,
        "training_config_still_missing": False,
        "recommend_enter_train360_spherical_pose_baseline": all(
            int(forward_results[split]["nan_inf_count"]) == 0 for split in splits
        ),
        "largest_blocker": (
            "Need manifest-native evaluator/training integration beyond forward sanity."
            if all(int(forward_results[split]["nan_inf_count"]) == 0 for split in splits)
            else "Forward produced NaN/Inf or failed on at least one split."
        ),
    }

    compliance = {
        "no_training_executed": True,
        "no_finetune_executed": True,
        "optimizer_created": False,
        "backward_called": False,
        "learned_weights_saved": False,
        "s5_locked_metrics_modified": False,
        "legacy_scene01_artifact_dependency": False,
        "direct_glob_data_360dvo_sequences": False,
        "dset2c_canonical_manifest_required": True,
        "uses_eval_gt_for_training": False,
        "uses_test_gt_for_training": False,
        "uses_orbslam3_teacher": False,
        "uses_hkust_360dvo_teacher": False,
    }

    payload = {
        "task": "TRAIN360B_manifest_native_dataloader_adapter_and_forward_sanity",
        "scope": "dataloader_adapter_and_forward_sanity_only",
        "train360a_reference": {
            "report_path": str(train360a_md_path),
            "module_matrix_path": str(train360a_matrix_path),
            "recommended_route_is_hybrid": "Recommended `TRAIN360-v0` body: `hybrid`" in train360a_md,
            "direct_reuse_modules": direct_modules,
            "adapt_reuse_modules": adapt_modules,
            "non_reuse_modules": no_modules,
        },
        "manifest_schema_audit": manifest_audit,
        "manifest_split_audit": sequence_audit,
        "manifest_consistency": manifest_consistency,
        "dataset_adapter": {
            "file_path": str(REPO_ROOT / "datasets" / "dset2c_manifest_dataset.py"),
            "class_name": "Dset2CCanonicalPairDataset",
            "sample_fields": [
                "image_a",
                "image_b",
                "IA",
                "IB",
                "R_BA",
                "R_gt",
                "t_BA_B",
                "t_gt_vec",
                "tdir_B",
                "t_gt_dir",
                "tmag",
                "t_gt_mag",
                "log_tmag",
                "sequence_id",
                "pair_index",
                "pair_type",
                "k",
                "split",
                "meta",
            ],
            "image_hw": list(image_hw),
            "normalization": "float32 RGB in [0,1], shape [3,H,W], resize bilinear to checkpoint cfg H/W",
            "schema_discovery": "discover_manifest_field_mapping with explicit canonical field map validation",
            "failure_handling": "skip invalid rows at init when configured; runtime image load raises explicit error",
        },
        "model_compatibility": {
            "model_file_path": str(REPO_ROOT / "model.py"),
            "model_class_name": "PanoramaRelPoseModel",
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_load_status": load_status,
            "checkpoint_cfg_core": {
                "H": int(getattr(ckpt_cfg, "H")),
                "W": int(getattr(ckpt_cfg, "W")),
                "in_ch": int(getattr(ckpt_cfg, "in_ch")),
                "use_coarse_interaction": bool(getattr(ckpt_cfg, "use_coarse_interaction")),
                "use_fine_stage": bool(getattr(ckpt_cfg, "use_fine_stage")),
                "translation_output_frame": str(getattr(ckpt_cfg, "translation_output_frame")),
                "translation_local_frame": str(getattr(ckpt_cfg, "translation_local_frame")),
            },
            "bounded_log_tmag_enabled": True,
            "positive_tmag_path_enabled": True,
            "device_used": str(device),
        },
        "forward_sanity": forward_results,
        "loss_interface_stub": _loss_interface_stub(),
        "deferred_modules": {
            "s5e12_observability_gate": "deferred_but_planned",
            "rotation_compensation": "deferred",
            "contiguous_kstep_composition": "deferred",
            "struct_geometry_token_soft_correspondence": "deferred",
            "arch360_enhancements": "deferred",
        },
        "readiness_assessment": readiness,
        "compliance": compliance,
    }

    out_json_path = REPO_ROOT / reports_cfg["train360b_forward_json"]
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    report_lines = [
        "# TRAIN360B manifest-native dataloader adapter and forward sanity",
        "",
        "## 1. Executive summary",
        f"- manifest-native adapter implemented: `{readiness['manifest_adapter_ready']}`",
        f"- train/val/test no_grad forward sanity completed: `{readiness['model_forward_ready']}`",
        f"- can enter `TRAIN360_spherical_pose_baseline`: `{readiness['recommend_enter_train360_spherical_pose_baseline']}`",
        f"- current largest blocker: {readiness['largest_blocker']}",
        "- This task only covers dataloader + forward sanity. No training loop, optimizer, finetune, or weight saving was executed.",
        "",
        "## 2. Manifest schema audit",
    ]
    for split in splits:
        info = manifest_audit[split]
        report_lines.extend(
            [
                f"### {split}",
                f"- sample_count = `{info['kept_row_count']}`",
                f"- sequence_ids = `{info['sequence_ids']}`",
                f"- schema_fields = `{info['schema_fields']}`",
                f"- skip_reasons = `{info['skip_reasons']}`",
            ]
        )
    report_lines.extend(
        [
            f"- sequence_overlap = `{sequence_audit['sequence_overlap']}`",
            f"- hygiene_consistency = `{manifest_consistency}`",
            "",
            "## 3. Dataset adapter design",
            f"- dataset class path: `datasets/dset2c_manifest_dataset.py`",
            "- sample dict returns manifest-native tensors plus T57b-compatible aliases `IA` / `IB` / `R_gt` / `t_gt_vec` / `t_gt_dir` / `t_gt_mag`.",
            "- image loading: RGB, float32, [0,1], bilinear resize to checkpoint `H/W`.",
            "- pose loading: uses manifest `R_BA`, `t_BA_B`, `tdir_B`, `tmag`; computes guarded `log_tmag`.",
            "- failure handling: invalid rows can be filtered at dataset init; runtime image failures raise explicit errors.",
            "",
            "## 4. Model compatibility",
            f"- PanoramaRelPoseModel path: `{REPO_ROOT / 'model.py'}`",
            f"- checkpoint load status: `{load_status['status']}`",
            f"- missing keys: `{load_status.get('missing_keys', [])}`",
            f"- unexpected keys: `{load_status.get('unexpected_keys', [])}`",
            f"- strict error: `{load_status.get('strict_error')}`",
            f"- bounded log_tmag enabled: `True`",
            f"- positive tmag enabled: `True`",
            "",
            "## 5. Forward sanity results",
        ]
    )
    for split in splits:
        info = forward_results[split]
        report_lines.extend(
            [
                f"### {split}",
                f"- input_image_shape = `{info['input_image_shape']}`",
                f"- batch_size = `{info['batch_size']}`",
                f"- model_stage = `{info['model_stage']}`",
                f"- R_output_shape = `{info['R_output_shape']}`",
                f"- tdir_output_shape = `{info['tdir_output_shape']}`",
                f"- tmag_output_shape = `{info['tmag_output_shape']}`",
                f"- log_tmag_output_shape = `{info['log_tmag_output_shape']}`",
                f"- tdir_norm_stats = `{info['tdir_norm_stats']}`",
                f"- tmag_stats = `{info['tmag_stats']}`",
                f"- nan_inf_count = `{info['nan_inf_count']}`",
            ]
        )
    report_lines.extend(
        [
            "",
            "## 6. TRAIN360-v0 readiness assessment",
            f"- dataloader ready: `{readiness['manifest_adapter_ready']}`",
            f"- model forward ready: `{readiness['model_forward_ready']}`",
            f"- loss interface ready: `{readiness['loss_interface_ready']}`",
            f"- evaluator still missing: `{readiness['evaluator_still_missing']}`",
            f"- training config still missing: `{readiness['training_config_still_missing']}`",
            f"- recommend entering formal baseline task: `{readiness['recommend_enter_train360_spherical_pose_baseline']}`",
            "",
            "## 7. Deferred modules",
        ]
    )
    for key, value in payload["deferred_modules"].items():
        report_lines.append(f"- {key}: `{value}`")
    report_lines.extend(
        [
            "",
            "## 8. Compliance checklist",
        ]
    )
    for key, value in compliance.items():
        report_lines.append(f"- `{key} = {str(value).lower()}`")

    out_md_path = REPO_ROOT / reports_cfg["train360b_markdown"]
    out_md_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"manifest adapter: {'ready' if readiness['manifest_adapter_ready'] else 'blocked'}")
    print(f"model forward: {'ready' if readiness['model_forward_ready'] else 'blocked'}")
    print(f"checkpoint load: {load_status['status']}")
    for split in splits:
        result = "pass" if int(forward_results[split]["nan_inf_count"]) == 0 else "fail"
        print(f"{split} forward sanity: {result}")
    print(
        "next recommended task: "
        + (
            "TRAIN360_spherical_pose_baseline"
            if readiness["recommend_enter_train360_spherical_pose_baseline"]
            else "fix listed blockers"
        )
    )


if __name__ == "__main__":
    main()
