#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import quote

from miniyaml import load_yaml_like
from s5e2_adjacent_dense_lib import write_json


HF_INFO_URL = "https://huggingface.co/api/datasets/{repo_id}"
HF_RESOLVE_CACHE = "https://huggingface.co/api/resolve-cache/datasets/{repo_id}/{sha}/{path_q}"


def _curl_json(url: str) -> Any:
    proc = subprocess.run(
        ["curl", "-sS", "-L", "--http1.1", "--max-time", "120", url],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return json.loads(proc.stdout)


def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return
    subprocess.run(
        ["curl", "-sS", "-L", "--http1.1", "--max-time", "120", "-o", str(dst), url],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _load_cfg(path: Path) -> Dict[str, Any]:
    return load_yaml_like(path) or {}


def _seq_from_path(path: str) -> str | None:
    parts = path.split("/")
    if len(parts) >= 3 and parts[0] == "Sequences":
        return parts[1]
    if len(parts) == 2 and parts[0] == "GroundTruth" and parts[1].endswith(".txt"):
        return parts[1][:-4]
    return None


def _existing_sequences(local_root: Path) -> List[str]:
    seq_dir = local_root / "Sequences"
    if not seq_dir.exists():
        return []
    return sorted(p.name for p in seq_dir.iterdir() if p.is_dir())


def _pick_sequences(all_sequences: List[str], existing: List[str], target_total: int) -> List[str]:
    preferred = [
        "bridge_night",
        "field",
        "wingsuit",
        "mountains",
        "hongkong_central",
        "dragon_boat",
        "city_driving",
        "canyon_line",
        "drone_racetrack",
        "snowmobile",
        "london_bridge",
        "grove",
    ]
    picked: List[str] = []
    for seq in existing:
        if seq in all_sequences and seq not in picked:
            picked.append(seq)
    for seq in preferred:
        if seq in all_sequences and seq not in picked:
            picked.append(seq)
        if len(picked) >= target_total:
            return picked[:target_total]
    for seq in sorted(all_sequences):
        if seq not in picked:
            picked.append(seq)
        if len(picked) >= target_total:
            break
    return picked[:target_total]


def _subset_gt_and_timestamps(gt_path: Path, ts_path: Path, count: int) -> Dict[str, Any]:
    lines = [ln.strip() for ln in gt_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    subset = lines[:count]
    gt_path.write_text("\n".join(subset) + ("\n" if subset else ""), encoding="utf-8")
    ts_path.parent.mkdir(parents=True, exist_ok=True)
    ts = [f"{idx * 0.1:.1f}" for idx in range(len(subset))]
    ts_path.write_text("\n".join(ts) + ("\n" if ts else ""), encoding="utf-8")
    return {"gt_rows_kept": len(subset), "timestamp_rows_written": len(ts), "timestamp_source": "synthetic_10fps_from_dataset_readme"}


def run(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = _load_cfg(Path(args.config))
    dataset_cfg = cfg.get("dataset", {})
    target_cfg = cfg.get("target_expansion", {})
    local_root = Path(str(dataset_cfg.get("local_root", "data/360DVO")))
    local_root.mkdir(parents=True, exist_ok=True)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "dataset_repo_reachable": False,
        "local_root": str(local_root),
        "existing_sequences": _existing_sequences(local_root),
        "new_sequences_downloaded": [],
        "selected_sequences_total": [],
        "num_sequences_total": 0,
        "download_attempted": False,
        "download_complete": False,
        "total_bytes_downloaded": 0,
        "max_download_bytes_respected": True,
        "access_blocked": False,
        "blockers": [],
    }
    try:
        info = _curl_json(HF_INFO_URL.format(repo_id=str(dataset_cfg.get("source_repo"))))
        payload["dataset_repo_reachable"] = True
    except Exception as exc:
        payload["access_blocked"] = True
        payload["blockers"] = [f"ACCESS_BLOCKED:{exc}"]
        write_json(out_json, payload)
        return payload

    siblings = list(info.get("siblings", []))
    seq_to_images: Dict[str, List[str]] = {}
    seq_to_gt: Dict[str, str] = {}
    for node in siblings:
        path = str(node.get("rfilename", ""))
        seq = _seq_from_path(path)
        if seq is None:
            continue
        if path.lower().endswith(".jpg"):
            seq_to_images.setdefault(seq, []).append(path)
        elif path.startswith("GroundTruth/") and path.lower().endswith(".txt"):
            seq_to_gt[seq] = path

    valid_sequences = sorted(seq for seq in seq_to_images if seq in seq_to_gt)
    target_total = int(target_cfg.get("target_num_sequences", 10))
    selected = _pick_sequences(valid_sequences, payload["existing_sequences"], target_total)
    payload["selected_sequences_total"] = selected
    payload["download_attempted"] = True

    sha = str(info.get("sha"))
    max_bytes = int(target_cfg.get("max_download_bytes", 12884901888))
    frame_cap = min(int(target_cfg.get("max_files_per_sequence", 1200)), 110)
    total_bytes = 0
    new_sequences: List[str] = []
    try:
        for seq in selected:
            images = sorted(seq_to_images[seq])[:frame_cap]
            gt_rel = seq_to_gt[seq]
            seq_is_new = seq not in payload["existing_sequences"]
            gt_local = local_root / gt_rel
            _download(HF_RESOLVE_CACHE.format(repo_id=str(dataset_cfg.get("source_repo")), sha=sha, path_q=quote(gt_rel, safe="")), gt_local)
            seq_bytes = gt_local.stat().st_size
            for rel in images:
                dst = local_root / rel
                _download(HF_RESOLVE_CACHE.format(repo_id=str(dataset_cfg.get("source_repo")), sha=sha, path_q=quote(rel, safe="")), dst)
                seq_bytes += dst.stat().st_size
            ts_local = local_root / "Timestamps" / f"{seq}_timestamps.txt"
            _subset_gt_and_timestamps(gt_local, ts_local, len(images))
            total_bytes += seq_bytes
            if seq_is_new:
                new_sequences.append(seq)
            if total_bytes > max_bytes:
                payload["max_download_bytes_respected"] = False
                payload["blockers"].append("MAX_DOWNLOAD_BYTES_EXCEEDED")
                break
    except Exception as exc:
        payload["blockers"].append(f"DOWNLOAD_FAILED:{exc}")

    final_sequences = _existing_sequences(local_root)
    payload["new_sequences_downloaded"] = new_sequences
    payload["selected_sequences_total"] = [seq for seq in selected if seq in final_sequences]
    payload["num_sequences_total"] = len(payload["selected_sequences_total"])
    payload["download_complete"] = not bool(payload["blockers"]) and payload["num_sequences_total"] >= int(target_cfg.get("min_num_sequences", 7))
    payload["total_bytes_downloaded"] = int(total_bytes)
    if payload["num_sequences_total"] < int(target_cfg.get("min_num_sequences", 7)):
        payload["blockers"].append("INSUFFICIENT_SEQUENCES")
    write_json(out_json, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-json", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
