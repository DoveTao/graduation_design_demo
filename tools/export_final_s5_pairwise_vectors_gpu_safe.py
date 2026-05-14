#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent

import sys
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from eval_clean_policy import _load_ckpt_cfg, _cfg_from_dict, _build_eval_dataset  # type: ignore
from model import PanoramaRelPoseModel

DEFAULT_JSONL = "external_baselines/results/s5_pairwise_replay_s5d7/final_s5_pairwise_vectors_scene01_seq03.jsonl"
DEFAULT_NPZ = "external_baselines/results/s5_pairwise_replay_s5d7/final_s5_pairwise_vectors_scene01_seq03.npz"
DEFAULT_META = "external_baselines/results/s5_pairwise_replay_s5d7/final_s5_pairwise_export_metadata.json"


def _resolve(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else REPO_ROOT / q


def _load_model(device: torch.device):
    policy_path = REPO_ROOT / "checkpoints" / "S5_clean_tmag_calibration_policy.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    ckpt_path = REPO_ROOT / str(policy["base_checkpoint_path"])
    cfg = _cfg_from_dict(_load_ckpt_cfg(ckpt_path))
    cfg.use_fine_stage = True
    cfg.fine_rot_fuse_strength = float(policy["fine_rot_fuse_strength"])
    cfg.fine_tdir_fuse_strength = float(policy["fine_tdir_fuse_strength"])
    cfg.fine_tmag_fuse_strength = float(policy["fine_tmag_fuse_strength"])
    cfg.use_geometry_refine = bool(policy.get("use_geometry_refine", False))
    cfg.tmag_condition_on_dt = False
    cfg.max_eval_batches = 0
    cfg.odom_eval_prefer_k = 1
    cfg.odom_eval_fallback_to_min_k = False
    cfg.eval_k_list = (1, 2, 3, 5, 10, 20)

    model = PanoramaRelPoseModel(cfg, device).to(device)
    payload = torch.load(str(ckpt_path), map_location=device)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    msg = model.load_state_dict(state, strict=False)
    model.eval()
    return model, cfg, policy_path, ckpt_path, msg


def _sanitize_float(x: Any) -> float:
    return float(np.float64(x))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--timestamps", required=True)
    ap.add_argument("--groundtruth", required=True)
    ap.add_argument("--out-jsonl", default=DEFAULT_JSONL)
    ap.add_argument("--out-npz", default=DEFAULT_NPZ)
    ap.add_argument("--out-metadata", default=DEFAULT_META)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    args = ap.parse_args()

    out_jsonl = _resolve(args.out_jsonl)
    out_npz = _resolve(args.out_npz)
    out_meta = _resolve(args.out_metadata)
    for p in (out_jsonl, out_npz, out_meta):
        p.parent.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            device = torch.device("cpu")

    meta: Dict[str, Any] = {
        "experiment": "S5D7_gpu_safe_pairwise_export_hook",
        "candidate": "S5_clean_tmag_calibration_policy",
        "jsonl_path": str(out_jsonl.relative_to(REPO_ROOT)),
        "npz_path": str(out_npz.relative_to(REPO_ROOT)),
        "metadata_path": str(out_meta.relative_to(REPO_ROOT)),
        "gt_used_to_generate_predictions": False,
        "diagnostic_only": True,
        "available": False,
        "num_pairs": 0,
        "pair_selection": "unavailable",
        "coverage_vs_453": 0.0,
        "translation_frame": "unknown",
        "rotation_convention": "unknown",
        "unavailable_reason": None,
        "notes": [],
    }

    try:
        model, cfg, policy_path, ckpt_path, load_msg = _load_model(device)
        ds = _build_eval_dataset(cfg, split="test")
        manifest = ds.manifest()

        rows: List[Dict[str, Any]] = []
        with out_jsonl.open("w", encoding="utf-8") as f, torch.no_grad():
            pair_idx = 0
            for ds_idx, m in enumerate(manifest):
                if str(m.get("scene")) != args.scene or str(m.get("seq")) != args.seq:
                    continue
                try:
                    if int(m.get("k", -1)) != 1:
                        continue
                except Exception:
                    continue

                sample = ds[ds_idx]
                IA = sample["IA"].unsqueeze(0).to(device, non_blocking=True)
                IB = sample["IB"].unsqueeze(0).to(device, non_blocking=True)
                dt_world = float(m.get("dt_world", sample.get("t_gt_mag", 0.0)))
                dt_tensor = torch.tensor([dt_world], device=device, dtype=torch.float32)
                R_pred, t_pred, aux = model(IA, IB, enable_depth_fusion=True, dt_world=dt_tensor)

                t_vec = aux.get("t_vec_local", aux.get("t_vec_out", aux.get("t_vec", t_pred)))
                t_arr = t_vec.detach().float().view(-1).cpu().numpy()
                if t_arr.size >= 3:
                    t_arr = t_arr[:3]
                else:
                    t_arr = np.pad(t_arr, (0, max(0, 3 - t_arr.size)), mode="constant")

                r_arr = R_pred.detach().float().view(-1).cpu().numpy()
                if r_arr.size >= 9:
                    r_val = r_arr[:9].reshape(3, 3).tolist()
                    r_rep = "matrix"
                else:
                    r_val = r_arr.tolist()
                    r_rep = "unknown"

                rec = {
                    "pair_index": int(pair_idx),
                    "timestamp_i": _sanitize_float(m.get("tsA", m.get("ts_i", m.get("timestamp_i", 0.0)))),
                    "timestamp_j": _sanitize_float(m.get("tsB", m.get("ts_j", m.get("timestamp_j", 0.0)))),
                    "frame_i": str(m.get("i", m.get("frame_i", ""))),
                    "frame_j": str(m.get("j", m.get("frame_j", ""))),
                    "pair_type": "selected_k1",
                    "k": 1,
                    "source": "final_s5_model_prediction",
                    "candidate": "S5_clean_tmag_calibration_policy",
                    "rotation": {"representation": r_rep, "value": r_val},
                    "translation": {
                        "frame": "local",
                        "value": [float(t_arr[0]), float(t_arr[1]), float(t_arr[2])],
                        "magnitude": float(np.linalg.norm(t_arr)),
                    },
                    "confidence": None,
                    "gt_used_to_generate_prediction": False,
                    "notes": [],
                }
                f.write(json.dumps(rec, ensure_ascii=True) + "\n")
                rows.append(rec)
                pair_idx += 1

                del IA, IB, dt_tensor, R_pred, t_pred, aux, t_vec
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        arr_t = np.asarray([r["translation"]["value"] for r in rows], dtype=np.float32) if rows else np.zeros((0, 3), dtype=np.float32)
        np.savez(
            out_npz,
            available=np.array([len(rows) > 0]),
            num_pairs=np.array([len(rows)]),
            translation_local=arr_t,
        )

        meta.update(
            {
                "available": len(rows) > 0,
                "num_pairs": len(rows),
                "pair_selection": "selected_k1" if rows else "unavailable",
                "coverage_vs_453": float(len(rows) / 453.0),
                "translation_frame": "local" if rows else "unknown",
                "rotation_convention": "R_ij in model output frame" if rows else "unknown",
                "policy_path": str(policy_path.relative_to(REPO_ROOT)),
                "checkpoint_path": str(ckpt_path.relative_to(REPO_ROOT)),
                "model_load_missing": len(getattr(load_msg, "missing_keys", [])),
                "model_load_unexpected": len(getattr(load_msg, "unexpected_keys", [])),
            }
        )
        if not rows:
            meta["unavailable_reason"] = "no selected_k1 pairs found for scene/seq in eval manifest"
    except Exception as e:
        out_jsonl.write_text("", encoding="utf-8")
        np.savez(out_npz, available=np.array([False]), num_pairs=np.array([0]))
        meta["available"] = False
        meta["num_pairs"] = 0
        meta["pair_selection"] = "unavailable"
        meta["unavailable_reason"] = f"export_exception: {type(e).__name__}: {e}"
        meta["notes"].append("No fake pairwise vectors were written.")

    out_meta.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
