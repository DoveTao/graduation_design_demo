#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from datasets.dset2c_manifest_dataset import Dset2CCanonicalPairDataset
from models.struct360b_match_free_coarse_to_fine import STRUCT360BMatchFreeCoarseToFineModel
from train360.core.config import Config


CONFIG_PATH = REPO_ROOT / "configs" / "final360i_struct360b_final.yaml"
TEST_MANIFEST = REPO_ROOT / "external_baselines" / "results" / "dset2c_360dvo_canonical" / "pair_manifest_test.jsonl"
CHECKPOINT_PATH = REPO_ROOT / "checkpoints" / "FINAL360I_struct360b_final" / "seed0" / "best_val.pt"
OUT_DIR = REPO_ROOT / "thesis" / "final_assets" / "figures" / "matching_geometry"
MANIFEST_PATH = REPO_ROOT / "thesis" / "final_assets" / "manifests" / "matching_geometry_eval_only_manifest.json"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_yaml_like(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _cfg_from_dict(cfg_dict: Dict[str, Any]) -> Config:
    cfg = Config()
    for key, value in cfg_dict.items():
        setattr(cfg, key, value)
    return cfg


def _load_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> STRUCT360BMatchFreeCoarseToFineModel:
    payload = torch.load(str(checkpoint_path), map_location=device)
    model_cfg = _cfg_from_dict(payload["cfg"])
    model = STRUCT360BMatchFreeCoarseToFineModel(model_cfg, device).to(device)
    model.load_state_dict(payload["model"], strict=False)
    model.eval()
    return model


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().float().cpu().numpy()


def _pick_sample_indices(dataset: Dset2CCanonicalPairDataset) -> List[int]:
    picked: List[int] = []
    targets = [("snowmobile", 1), ("ridge_to_lake", 1)]
    for sequence_id, k in targets:
        for idx, sample in enumerate(dataset.samples):
            if sample["sequence_id"] == sequence_id and int(sample["k"]) == k and sample["pair_type"] == "adjacent":
                picked.append(idx)
                break
    if not picked:
        picked = [0]
    return picked[:3]


def _heat(ax: plt.Axes, data: np.ndarray, title: str, cmap: str = "viridis") -> None:
    im = ax.imshow(data, aspect="auto", cmap=cmap)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Token B")
    ax.set_ylabel("Token A")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _render_sample(model: STRUCT360BMatchFreeCoarseToFineModel, sample: Dict[str, Any], out_prefix: Path) -> Dict[str, Any]:
    IA = sample["IA"].unsqueeze(0).to(next(model.parameters()).device)
    IB = sample["IB"].unsqueeze(0).to(next(model.parameters()).device)
    dt_world = torch.tensor([float(sample["meta"]["dt_world"])], device=IA.device, dtype=torch.float32)
    with torch.no_grad():
        _R, _t, aux = model(IA, IB, dt_world=dt_world)

    raw = _to_numpy(aux["Wf_ab_raw"][0])
    guided = _to_numpy(aux["Wf_ab"][0])
    diff = guided - raw
    allowed = _to_numpy(aux["allowed_mask"][0]).astype(np.float32)
    epi_residual = _to_numpy(aux["epi_residual"][0])
    epi_bias = _to_numpy(aux["epi_bias"][0])

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    title = (
        f"{sample['meta']['sequence_id']} | pair_index={sample['meta']['pair_index']} | "
        f"k={sample['meta']['k']} | dt={float(sample['meta']['dt_world']):.3f}"
    )
    fig.suptitle(title, fontsize=12)
    _heat(axes[0, 0], raw, "Raw Interaction $Wf_{ab}^{raw}$")
    _heat(axes[0, 1], guided, "Geometry-guided Interaction $Wf_{ab}$")
    _heat(axes[0, 2], diff, "Guided - Raw", cmap="coolwarm")
    _heat(axes[1, 0], allowed, "Allowed Mask", cmap="gray")
    _heat(axes[1, 1], epi_residual, "Epipolar Residual")
    _heat(axes[1, 2], epi_bias, "Epipolar Bias", cmap="coolwarm")

    png_path = out_prefix.with_suffix(".png")
    pdf_path = out_prefix.with_suffix(".pdf")
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)

    return {
        "file_path_png": str(png_path.relative_to(REPO_ROOT)),
        "file_path_pdf": str(pdf_path.relative_to(REPO_ROOT)),
        "sequence_id": sample["meta"]["sequence_id"],
        "pair_index": int(sample["meta"]["pair_index"]),
        "k": int(sample["meta"]["k"]),
        "source_manifest": str(TEST_MANIFEST.relative_to(REPO_ROOT)),
        "checkpoint": str(CHECKPOINT_PATH.relative_to(REPO_ROOT)),
        "default_model_behavior_changed": False,
        "debug_hooks_added": False,
        "return_debug_used": False,
        "eval_only": True,
    }


def generate_matching_geometry_assets() -> List[Dict[str, Any]]:
    cfg = _load_yaml_like(CONFIG_PATH)
    base_cfg = _load_yaml_like(REPO_ROOT / cfg["inputs"]["base_config"])
    dataset = Dset2CCanonicalPairDataset(
        str(TEST_MANIFEST),
        expected_split="test",
        image_hw=tuple(int(x) for x in base_cfg["data"]["image_hw"]),
        require_paths=bool(base_cfg["data"]["require_paths"]),
        skip_invalid=bool(base_cfg["data"]["skip_invalid"]),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model_from_checkpoint(CHECKPOINT_PATH, device)
    _ensure_dir(OUT_DIR)
    _ensure_dir(MANIFEST_PATH.parent)

    entries: List[Dict[str, Any]] = []
    for local_rank, idx in enumerate(_pick_sample_indices(dataset), start=1):
        sample = dataset[idx]
        sequence_id = sample["meta"]["sequence_id"]
        pair_index = int(sample["meta"]["pair_index"])
        prefix = OUT_DIR / f"matching_geometry_{local_rank:02d}_{sequence_id}_pair{pair_index}"
        IA = sample["IA"].unsqueeze(0).to(next(model.parameters()).device)
        IB = sample["IB"].unsqueeze(0).to(next(model.parameters()).device)
        dt_world = torch.tensor([float(sample["meta"]["dt_world"])], device=IA.device, dtype=torch.float32)
        with torch.no_grad():
            _R, _t, aux = model(IA, IB, dt_world=dt_world)
        required = {"Wf_ab_raw", "Wf_ab", "allowed_mask", "epi_residual", "epi_bias"}
        if not required.issubset(aux.keys()):
            unavailable = {
                "status": "unavailable",
                "reason": "FINAL360I retained STRUCT360B mainline does not emit the requested matching/geometry tensors without intrusive model changes.",
                "sequence_id": sample["meta"]["sequence_id"],
                "pair_index": int(sample["meta"]["pair_index"]),
                "k": int(sample["meta"]["k"]),
                "source_manifest": str(TEST_MANIFEST.relative_to(REPO_ROOT)),
                "checkpoint": str(CHECKPOINT_PATH.relative_to(REPO_ROOT)),
                "default_model_behavior_changed": False,
                "debug_hooks_added": False,
                "return_debug_used": False,
                "eval_only": True,
            }
            MANIFEST_PATH.write_text(json.dumps([unavailable], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            return []
        entries.append(_render_sample(model, sample, prefix))

    MANIFEST_PATH.write_text(json.dumps(entries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return entries


def main() -> None:
    entries = generate_matching_geometry_assets()
    print(json.dumps({"written": True, "count": len(entries), "manifest": str(MANIFEST_PATH.relative_to(REPO_ROOT))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
