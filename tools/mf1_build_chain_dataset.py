#!/usr/bin/env python3
"""Build MF1a train-split k=1 contiguous chain cache from real S5 predictions."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from eval_clean_policy import _build_eval_dataset
from jrt1_build_pair_dataset import FEATURE_NAMES, _build_s5_model, _validate_config


LOCKED = {"ATE": 7.352288, "drift": 1.327343, "path_ratio": 0.932379}


def _run_guard() -> None:
    subprocess.run(["bash", "scripts/verify_final_candidate.sh"], cwd=REPO_ROOT, check=True)


def _resolve(raw: str | Path) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else REPO_ROOT / p


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _unit(v: np.ndarray) -> np.ndarray:
    return (v / max(float(np.linalg.norm(v)), 1.0e-12)).astype(np.float32)


def _pair_times(pair_id: str) -> Tuple[float, float]:
    m = re.search(r"::([0-9.]+)->([0-9.]+)::k=", pair_id)
    if not m:
        return float("nan"), float("nan")
    return float(m.group(1)), float(m.group(2))


def _select_indices(n: int, cap: int) -> List[int]:
    if cap <= 0 or n <= cap:
        return list(range(n))
    raw = np.linspace(0, n - 1, num=cap)
    out: List[int] = []
    seen = set()
    for v in raw:
        j = max(0, min(n - 1, int(round(float(v)))))
        if j not in seen:
            out.append(j)
            seen.add(j)
    return out


def _extract_rows(model, ds, device: torch.device) -> List[Dict[str, Any]]:
    manifest = ds.manifest()
    rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for ds_idx, meta in enumerate(manifest):
            if int(meta.get("k", -1)) != 1:
                continue
            if len(rows) == 0 or (len(rows) + 1) % 128 == 0:
                print(f"[mf1-dataset] extracting k=1 pair {len(rows) + 1}", flush=True)
            sample = ds[ds_idx]
            IA = sample["IA"].unsqueeze(0).to(device, non_blocking=True)
            IB = sample["IB"].unsqueeze(0).to(device, non_blocking=True)
            dt_val = float(meta.get("dt_world", float(sample["t_gt_mag"])))
            dt_world = torch.tensor([dt_val], device=device, dtype=torch.float32)
            R_pred_t, t_pred_t, aux = model(IA, IB, enable_depth_fusion=True, dt_world=dt_world)
            R_pred = R_pred_t.detach().float().cpu().numpy()[0].astype(np.float32)
            tdir_pred_t = aux.get("t_dir_out", t_pred_t) if isinstance(aux, dict) else t_pred_t
            tdir_pred = _unit(tdir_pred_t.detach().float().cpu().numpy()[0])
            tmag_pred = float(aux["t_mag"].detach().float().view(-1).cpu().numpy()[0])
            log_tmag_pred = float(math.log(max(tmag_pred, 1.0e-12)))
            R_gt = sample["R_gt"].detach().float().cpu().numpy().astype(np.float32)
            tdir_gt = _unit(sample["t_gt_dir"].detach().float().cpu().numpy())
            tmag_gt = float(sample["t_gt_mag"])
            pair_id = f"{meta.get('scene')}::{meta.get('seq')}::{meta.get('tsA')}->{meta.get('tsB')}::k=1"
            features = np.concatenate(
                [R_pred.reshape(-1), tdir_pred.reshape(-1), np.asarray([log_tmag_pred, dt_val, 1.0], dtype=np.float32)],
                axis=0,
            ).astype(np.float32)
            rows.append(
                {
                    "features": features,
                    "R_pred": R_pred,
                    "tdir_pred": tdir_pred,
                    "log_tmag_pred": log_tmag_pred,
                    "tmag_pred": tmag_pred,
                    "R_gt": R_gt,
                    "tdir_gt": tdir_gt,
                    "tmag_gt": tmag_gt,
                    "dt": dt_val,
                    "k": 1,
                    "sequence_id": f"{meta.get('scene')}::{meta.get('seq')}",
                    "pair_id": pair_id,
                    "tsA": _pair_times(pair_id)[0],
                    "tsB": _pair_times(pair_id)[1],
                }
            )
    return rows


def _build_contiguous_windows(rows: List[Dict[str, Any]], window: int, stride: int) -> List[List[int]]:
    by_seq: Dict[str, List[int]] = {}
    for i, row in enumerate(rows):
        by_seq.setdefault(row["sequence_id"], []).append(i)
    windows: List[List[int]] = []
    for _seq, idxs in sorted(by_seq.items()):
        idxs = sorted(idxs, key=lambda i: (rows[i]["tsA"], rows[i]["tsB"]))
        chain: List[int] = []
        prev_b = None
        chains: List[List[int]] = []
        for i in idxs:
            ts_a = float(rows[i]["tsA"])
            ts_b = float(rows[i]["tsB"])
            if not chain:
                chain = [i]
            elif prev_b is not None and math.isfinite(ts_a) and abs(ts_a - prev_b) <= 1.0e-4:
                chain.append(i)
            else:
                chains.append(chain)
                chain = [i]
            prev_b = ts_b
        if chain:
            chains.append(chain)
        for chain in chains:
            if len(chain) < window:
                continue
            for start in range(0, len(chain) - window + 1, max(1, stride)):
                windows.append(chain[start : start + window])
    return windows


def main() -> None:
    ap = argparse.ArgumentParser(description="Build MF1a k=1 chain smoke dataset.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--summary", required=True)
    args = ap.parse_args()

    _run_guard()
    config = _read_json(_resolve(args.config))
    _validate_config(config)
    out_path = _resolve(args.output)
    summary_path = _resolve(args.summary)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, s5_policy, load_summary = _build_s5_model(config, device)
    cfg.eval_k_list = (1,)
    train_ds = _build_eval_dataset(cfg, split="train")
    test_ds = _build_eval_dataset(cfg, split="test")
    rows = _extract_rows(model, train_ds, device)
    if not rows:
        raise RuntimeError("MF1a builder produced zero train-split k=1 S5 prediction rows.")
    chain_cfg = config["chain"]
    windows = _build_contiguous_windows(rows, int(chain_cfg["window_size"]), int(chain_cfg["stride"]))
    if not windows:
        raise RuntimeError("MF1a builder produced zero contiguous chain windows.")
    smoke = config["smoke"]
    max_train = int(smoke["max_train_chains"])
    max_val = int(smoke["max_val_chains"])
    total_cap = max_train + max_val
    use_windows = [windows[i] for i in _select_indices(len(windows), total_cap)]
    if len(use_windows) <= max_train:
        # Keep a real validation slice when the train split has fewer contiguous
        # windows than the smoke cap; record the actual counts instead of padding.
        n_val = max(1, min(max_val, len(use_windows) // 3))
        n_train = len(use_windows) - n_val
        train_chains = use_windows[:n_train]
        val_chains = use_windows[n_train:]
    else:
        train_chains = use_windows[:max_train]
        val_chains = use_windows[max_train : max_train + max_val]
    if not train_chains or not val_chains:
        raise RuntimeError(f"MF1a needs non-empty train/val chains, got {len(train_chains)}/{len(val_chains)}")
    chain_indices = np.asarray(train_chains + val_chains, dtype=np.int64)
    split = np.asarray([0] * len(train_chains) + [1] * len(val_chains), dtype=np.int64)
    metadata = {
        "experiment_name": config["experiment_name"],
        "stage": config["stage"],
        "dataset_source": "RflyPanoPanoramaPairsEvalFixedKList train split with S5 k=1 predictions",
        "source_policy": config["base_policy_path"],
        "source_policy_name": s5_policy["name"],
        "window_size": int(chain_cfg["window_size"]),
        "stride": int(chain_cfg["stride"]),
        "num_pair_rows": int(len(rows)),
        "num_available_windows": int(len(windows)),
        "num_train_chains": int(len(train_chains)),
        "num_val_chains": int(len(val_chains)),
        "num_test_chains": 0,
        "num_available_test_k1_pairs": int(len(test_ds)),
        "feature_dim": len(FEATURE_NAMES),
        "feature_fields": FEATURE_NAMES,
        "label_fields": ["R_gt_3x3", "tdir_gt_3", "tmag_gt"],
        "no_test_gt_used_for_training": True,
        "test_gt_cached": False,
        "variants": config["variants"],
        "load_missing": int(len(load_summary["missing"])),
        "load_unexpected": int(len(load_summary["unexpected"])),
    }
    np.savez_compressed(
        out_path,
        features=np.asarray([r["features"] for r in rows], dtype=np.float32),
        R_pred=np.asarray([r["R_pred"] for r in rows], dtype=np.float32),
        tdir_pred=np.asarray([r["tdir_pred"] for r in rows], dtype=np.float32),
        log_tmag_pred=np.asarray([r["log_tmag_pred"] for r in rows], dtype=np.float32),
        tmag_pred=np.asarray([r["tmag_pred"] for r in rows], dtype=np.float32),
        R_gt=np.asarray([r["R_gt"] for r in rows], dtype=np.float32),
        tdir_gt=np.asarray([r["tdir_gt"] for r in rows], dtype=np.float32),
        tmag_gt=np.asarray([r["tmag_gt"] for r in rows], dtype=np.float32),
        dt=np.asarray([r["dt"] for r in rows], dtype=np.float32),
        k=np.asarray([r["k"] for r in rows], dtype=np.int64),
        sequence_id=np.asarray([r["sequence_id"] for r in rows], dtype="U128"),
        pair_id=np.asarray([r["pair_id"] for r in rows], dtype="U256"),
        chain_indices=chain_indices,
        chain_split=split,
        metadata_json=np.asarray(json.dumps(metadata, indent=2, ensure_ascii=True)),
    )
    summary_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
