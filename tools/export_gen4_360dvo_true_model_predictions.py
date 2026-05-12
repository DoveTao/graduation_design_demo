#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent

import sys

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from eval_clean_policy import _cfg_from_dict, _load_ckpt_cfg  # type: ignore
from model import PanoramaRelPoseModel
from s5e2_adjacent_dense_lib import write_json


DEFAULT_RESULTS_DIR = REPO_ROOT / "external_baselines/results/gen4_true_image_pair_inference_recovery"
DEFAULT_AUDIT = REPO_ROOT / "checkpoints/GEN4_true_image_pair_inference_recovery.json"
MANIFEST_PATH = REPO_ROOT / "external_baselines/results/gen2_360dvo_external_adapter/pair_manifest_smoke.jsonl"


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _image_tensor(path: str, h: int, w: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((w, h), resample=Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr).unsqueeze(0)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    audit = _read_json(Path(args.audit))
    results_dir = Path(args.out_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    pred_path = results_dir / "predictions.jsonl"
    summary_path = results_dir / "export_summary.json"

    if audit.get("final_classification") != "GEN4_TRUE_IMAGE_PAIR_INFERENCE_READY":
        pred_path.write_text("", encoding="utf-8")
        payload = {
            "available": False,
            "blocked": True,
            "reason": audit.get("final_classification"),
            "num_pairs_predicted": 0,
            "num_pairs_total": len(_read_jsonl(MANIFEST_PATH)),
        }
        write_json(summary_path, payload)
        return payload

    ckpt_path = REPO_ROOT / "checkpoints/T57b_no_dt_multiscale_tmag_head_400/final.pt"
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    payload = torch.load(str(ckpt_path), map_location=device)
    cfg = _cfg_from_dict(_load_ckpt_cfg(ckpt_path))
    model = PanoramaRelPoseModel(cfg, device).to(device)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state, strict=False)
    model.eval()

    manifest = _read_jsonl(MANIFEST_PATH)
    rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for row in manifest[: int(args.max_pairs)]:
            IA = _image_tensor(row["image_path_a"], int(cfg.H), int(cfg.W)).to(device)
            IB = _image_tensor(row["image_path_b"], int(cfg.H), int(cfg.W)).to(device)
            dt = torch.tensor([float(row["timestamp_b"]) - float(row["timestamp_a"])], dtype=torch.float32, device=device)
            R_pred, t_pred, aux = model(IA, IB, enable_depth_fusion=True, dt_world=dt)
            t_mag = aux["t_mag"][0].detach().cpu().item()
            t_dir = aux["t_dir"][0].detach().cpu().numpy()
            t_vec = aux.get("t_vec_out", aux.get("t_vec_local", t_pred))[0].detach().cpu().numpy()
            rows.append(
                {
                    "dataset": "360DVO",
                    "sequence": "field",
                    "pair_index": int(row["pair_index"]),
                    "pair_type": row["pair_type"],
                    "k": int(row["k"]),
                    "timestamp_a": float(row["timestamp_a"]),
                    "timestamp_b": float(row["timestamp_b"]),
                    "image_path_a": row["image_path_a"],
                    "image_path_b": row["image_path_b"],
                    "R_pred_BA": R_pred[0].detach().cpu().numpy().tolist(),
                    "tdir_pred_B": t_dir.tolist(),
                    "tmag_pred": float(t_mag),
                    "tvec_pred_B": t_vec.tolist(),
                    "uses_gt_for_prediction": False,
                    "uses_gt_scale_calibration": False,
                    "uses_orbslam3_teacher": False,
                    "source_model": "T57b_true_image_pair_inference_recovery",
                }
            )

    pred_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    summary = {
        "available": True,
        "blocked": False,
        "checkpoint_path": str(ckpt_path),
        "device": str(device),
        "num_pairs_predicted": len(rows),
        "num_pairs_total": len(manifest),
        "pair_coverage": float(len(rows) / max(len(manifest), 1)),
        "prediction_fabricated": False,
        "uses_gt_for_prediction": False,
        "uses_gt_scale_calibration": False,
    }
    write_json(summary_path, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--out-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--max-pairs", type=int, default=100)
    parser.add_argument("--force-cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
