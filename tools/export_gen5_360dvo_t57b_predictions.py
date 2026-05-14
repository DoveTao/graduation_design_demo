#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from config import Config
from model import PanoramaRelPoseModel
from s5e2_adjacent_dense_lib import angle_deg_from_rot, vector_angle_deg, write_json


def _cfg_from_dict(cfg_dict: Dict[str, Any]) -> Config:
    cfg = Config()
    for k, v in cfg_dict.items():
        setattr(cfg, k, v)
    return cfg


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _rot_to_list(R: np.ndarray) -> List[List[float]]:
    return [[float(x) for x in row] for row in np.asarray(R, dtype=np.float64).tolist()]


def _load_model(ckpt_path: Path, device: torch.device) -> Tuple[PanoramaRelPoseModel, Config, Dict[str, Any]]:
    payload = torch.load(str(ckpt_path), map_location=device)
    cfg_dict = payload.get("cfg", {})
    if not isinstance(cfg_dict, dict):
        raise TypeError(f"Unsupported cfg type in checkpoint: {type(cfg_dict)}")
    cfg = _cfg_from_dict(cfg_dict)
    model = PanoramaRelPoseModel(cfg, device).to(device)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    msg = model.load_state_dict(state, strict=False)
    model.eval()
    return model, cfg, {"missing": list(msg.missing_keys), "unexpected": list(msg.unexpected_keys)}


def _load_image(path: Path, hw: Tuple[int, int]) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    H, W = hw
    if img.size != (W, H):
        img = img.resize((W, H), resample=Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)


def _is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def _resolve_image_path(row: Dict[str, Any], key: str) -> Path:
    raw = Path(str(row[key]))
    if raw.exists() and _is_valid_image(raw):
        return raw
    seq_id = str(row.get("seq_id", ""))
    ts_key = "timestamp_a" if key.endswith("_a") else "timestamp_b"
    frame_idx = int(round(float(row.get(ts_key, 0.0)) * 10.0)) + 1
    candidate = raw.parent / f"{frame_idx:04d}{raw.suffix or '.jpg'}"
    if candidate.exists() and _is_valid_image(candidate):
        return candidate
    raise FileNotFoundError(f"GEN5 could not resolve a valid image for {key}: raw={raw} candidate={candidate}")


def _metric(R_pred: np.ndarray, t_pred: np.ndarray, R_gt: np.ndarray, t_gt: np.ndarray) -> Dict[str, Any]:
    tdir = vector_angle_deg(t_pred, t_gt, absolute=False)
    tdir_abs = vector_angle_deg(t_pred, t_gt, absolute=True)
    cosine = None if tdir is None else float(np.cos(np.deg2rad(tdir)))
    return {
        "rot_deg": angle_deg_from_rot(R_pred @ R_gt.T),
        "tdir_deg": tdir,
        "tdir_abs_deg": tdir_abs,
        "tdir_cosine": cosine,
        "anti_parallel_flag": bool(cosine is not None and cosine < 0.0),
        "tmag_ratio": float(np.linalg.norm(t_pred) / max(np.linalg.norm(t_gt), 1.0e-12)),
        "pred_step_length": float(np.linalg.norm(t_pred)),
        "gt_step_length": float(np.linalg.norm(t_gt)),
    }


def _summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    def _vals(key: str) -> np.ndarray:
        vals = [r[key] for r in rows if r.get(key) is not None]
        return np.asarray(vals, dtype=np.float64)

    def _mean(key: str) -> float | None:
        vals = _vals(key)
        return None if vals.size == 0 else float(np.mean(vals))

    def _pct(key: str, q: float) -> float | None:
        vals = _vals(key)
        return None if vals.size == 0 else float(np.percentile(vals, q))

    pred = float(sum(float(r.get("pred_step_length", 0.0)) for r in rows))
    gt = float(sum(float(r.get("gt_step_length", 0.0)) for r in rows))
    return {
        "count": int(len(rows)),
        "rot_mean_deg": _mean("rot_deg"),
        "rot_median_deg": _pct("rot_deg", 50),
        "rot_p90_deg": _pct("rot_deg", 90),
        "signed_tdir_mean_deg": _mean("tdir_deg"),
        "signed_tdir_median_deg": _pct("tdir_deg", 50),
        "signed_tdir_p90_deg": _pct("tdir_deg", 90),
        "tdir_abs_mean_deg": _mean("tdir_abs_deg"),
        "tdir_abs_median_deg": _pct("tdir_abs_deg", 50),
        "tdir_abs_p90_deg": _pct("tdir_abs_deg", 90),
        "anti_parallel_rate": _mean("anti_parallel_flag"),
        "tmag_median_ratio": _pct("tmag_ratio", 50),
        "tmag_p90_ratio": _pct("tmag_ratio", 90),
        "tmag_p95_ratio": _pct("tmag_ratio", 95),
        "path_ratio": pred / max(gt, 1.0e-12),
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    manifests = [Path(p) for p in args.manifests]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: List[Dict[str, Any]] = []
    for path in manifests:
        manifest_rows.extend(_read_jsonl(path))
    if not manifest_rows:
        raise RuntimeError("No manifest rows provided for GEN5 export.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, load_summary = _load_model(Path(args.checkpoint), device)
    hw = (int(getattr(cfg, "H", 1024)), int(getattr(cfg, "W", 2048)))

    predictions: List[Dict[str, Any]] = []
    metric_rows: List[Dict[str, Any]] = []
    skipped_pairs: List[Dict[str, Any]] = []
    with torch.no_grad():
        for idx, row in enumerate(manifest_rows):
            try:
                path_a = _resolve_image_path(row, "image_path_a")
                path_b = _resolve_image_path(row, "image_path_b")
                IA = _load_image(path_a, hw).unsqueeze(0).to(device, non_blocking=True)
                IB = _load_image(path_b, hw).unsqueeze(0).to(device, non_blocking=True)
            except Exception as exc:
                skipped_pairs.append(
                    {
                        "pair_index": int(row["pair_index"]),
                        "seq_id": row.get("seq_id"),
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            dt_val = float(row.get("timestamp_b", 0.0)) - float(row.get("timestamp_a", 0.0))
            if dt_val <= 0.0:
                dt_val = max(float(row.get("tmag", 0.0)), 1.0e-3)
            dt_world = torch.tensor([dt_val], device=device, dtype=torch.float32)
            R_pred_t, t_pred_t, aux = model(IA, IB, enable_depth_fusion=True, dt_world=dt_world)
            R_pred = R_pred_t.detach().float().cpu().numpy()[0]
            tdir_pred = aux.get("t_dir_out", t_pred_t).detach().float().cpu().numpy()[0]
            tmag_pred = float(aux["t_mag"].detach().float().view(-1).cpu().numpy()[0])
            if aux.get("t_vec_out", None) is not None:
                tvec_pred = aux["t_vec_out"].detach().float().cpu().numpy()[0]
            else:
                tvec_pred = tdir_pred * tmag_pred

            R_gt = np.asarray(row["R_BA"], dtype=np.float64)
            t_gt = np.asarray(row["t_BA_B"], dtype=np.float64)
            metrics = _metric(R_pred, np.asarray(tvec_pred, dtype=np.float64), R_gt, t_gt)

            pred_row = {
                "dataset": "360DVO",
                "source_model": "T57b_true_external_model",
                "seq_id": row["seq_id"],
                "sequence": row["seq_id"],
                "split": row["split"],
                "pair_index": int(row["pair_index"]),
                "pair_type": row["pair_type"],
                "k": int(row["k"]),
                "timestamp_a": float(row["timestamp_a"]),
                "timestamp_b": float(row["timestamp_b"]),
                "image_path_a": str(path_a),
                "image_path_b": str(path_b),
                "R_pred_BA": _rot_to_list(R_pred),
                "tdir_pred_B": [float(x) for x in np.asarray(tdir_pred, dtype=np.float64).tolist()],
                "tmag_pred": float(tmag_pred),
                "tvec_pred_B": [float(x) for x in np.asarray(tvec_pred, dtype=np.float64).tolist()],
                "uses_gt_for_prediction": False,
                "uses_gt_scale_calibration": False,
                "uses_orbslam3_teacher": False,
                "model_stage": str(aux.get("stage", "unknown")),
            }
            predictions.append(pred_row)
            metric_rows.append(
                {
                    "dataset": "360DVO",
                    "seq_id": row["seq_id"],
                    "split": row["split"],
                    "pair_index": int(row["pair_index"]),
                    "pair_type": row["pair_type"],
                    "k": int(row["k"]),
                    "timestamp_a": float(row["timestamp_a"]),
                    "timestamp_b": float(row["timestamp_b"]),
                    **metrics,
                }
            )
            if (idx + 1) % 64 == 0 or (idx + 1) == len(manifest_rows):
                print(f"[GEN5] predicted {idx + 1}/{len(manifest_rows)} pairs", flush=True)

    by_sequence = {}
    for seq_id in sorted({r["seq_id"] for r in metric_rows}):
        by_sequence[seq_id] = _summary([r for r in metric_rows if r["seq_id"] == seq_id])
    by_k = {}
    for kval in sorted({int(r["k"]) for r in metric_rows}):
        by_k[str(kval)] = _summary([r for r in metric_rows if int(r["k"]) == kval])
    by_split = {}
    for split in sorted({str(r["split"]) for r in metric_rows}):
        by_split[split] = _summary([r for r in metric_rows if str(r["split"]) == split])

    metrics_payload = {
        "source_model": "T57b_true_external_model",
        "load_summary": load_summary,
        "num_pairs": len(metric_rows),
        "component_metrics": _summary(metric_rows),
        "metrics_by_sequence": by_sequence,
        "metrics_by_k": by_k,
        "metrics_by_split": by_split,
        "metrics_rows": metric_rows,
    }

    pred_path = out_dir / "predictions.jsonl"
    pred_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in predictions), encoding="utf-8")
    write_json(out_dir / "edge_component_metrics.json", metrics_payload)
    export_summary = {
        "model_eval_available": True,
        "num_pairs_predicted": len(predictions),
        "num_pairs_total": len(manifest_rows),
        "pair_coverage": float(len(predictions) / max(len(manifest_rows), 1)),
        "predictions_path_local_only": str(pred_path),
        "load_summary": load_summary,
        "num_pairs_skipped": len(skipped_pairs),
        "uses_gt_for_prediction": False,
        "uses_gt_scale_calibration": False,
        "uses_orbslam3_teacher": False,
    }
    write_json(out_dir / "export_summary.json", export_summary)
    write_json(out_dir / "skipped_pairs.json", {"skipped_pairs": skipped_pairs})
    return export_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
