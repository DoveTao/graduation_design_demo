#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from miniyaml import load_yaml_like
from s5e2_adjacent_dense_lib import write_json


def _load_cfg(path: Path) -> Dict[str, Any]:
    return load_yaml_like(path) or {}


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _local_image_count(seq_dir: Path) -> int:
    if not seq_dir.exists():
        return 0
    return len(list(seq_dir.glob("*.jpg")))


def _local_sequence_stats(local_root: Path, seq: str) -> Dict[str, int]:
    return {
        "image_count": _local_image_count(local_root / "Sequences" / seq),
        "pose_count": _line_count(local_root / "GroundTruth" / f"{seq}.txt"),
        "timestamp_count": _line_count(local_root / "Timestamps" / f"{seq}_timestamps.txt"),
    }


def _candidate_score(
    seq: str,
    rec: Dict[str, Any],
    existing_target: Sequence[str],
    local_root: Path,
    frame_cap: int,
) -> Tuple[int, int, int, int, str]:
    stats = _local_sequence_stats(local_root, seq)
    fully_downloadable = 1 if int(rec.get("image_files", 0)) <= frame_cap else 0
    is_current_existing = 1 if seq in existing_target else 0
    has_local_payload = 1 if max(stats.values()) > 0 else 0
    has_timestamps = 1 if rec.get("has_timestamps") else 0
    return (
        fully_downloadable,
        is_current_existing,
        has_local_payload,
        has_timestamps,
        int(rec.get("image_files", 0)),
    )


def _select_sequences(
    inventory: Dict[str, Dict[str, Any]],
    existing_target: Sequence[str],
    local_root: Path,
    target_total: int,
    frame_cap: int,
) -> List[str]:
    eligible = [
        seq for seq, rec in inventory.items()
        if rec.get("has_images") and rec.get("has_groundtruth")
    ]
    ranked = sorted(
        eligible,
        key=lambda seq: (_candidate_score(seq, inventory[seq], existing_target, local_root, frame_cap), seq),
        reverse=True,
    )
    return ranked[:target_total]


def _write_synthetic_timestamps(ts_path: Path, count: int) -> None:
    ts_path.parent.mkdir(parents=True, exist_ok=True)
    values = [f"{idx * 0.1:.1f}" for idx in range(count)]
    ts_path.write_text("\n".join(values) + ("\n" if values else ""), encoding="utf-8")


def _collect_sequence_paths(repo_id: str, seq: str) -> List[str]:
    return [
        f"Sequences/{seq}/*",
        f"GroundTruth/{seq}.txt",
        f"Timestamps/{seq}*",
    ]


def _snapshot_download_selected(
    repo_id: str,
    repo_type: str,
    local_root: Path,
    selected: Sequence[str],
) -> tuple[bool, List[str], str]:
    allow_patterns: List[str] = []
    for seq in selected:
        allow_patterns.extend(_collect_sequence_paths(repo_id=repo_id, seq=seq))
    try:
        from huggingface_hub import snapshot_download  # type: ignore

        snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            local_dir=str(local_root),
            local_dir_use_symlinks=False,
            allow_patterns=allow_patterns,
            resume_download=True,
        )
        return True, allow_patterns, ""
    except Exception as exc:
        return False, allow_patterns, f"{type(exc).__name__}:{exc}"


def run(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = _load_cfg(Path(args.config))
    remote = _read_json(Path(args.remote_inventory))
    repo_id = str(cfg.get("dataset", {}).get("source_repo", "chris1004336379/360DVO"))
    repo_type = str(cfg.get("dataset", {}).get("repo_type", "dataset"))
    local_root = Path(str(cfg.get("dataset", {}).get("local_root", "data/360DVO")))
    local_root.mkdir(parents=True, exist_ok=True)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    download_cfg = cfg.get("download", {})
    existing_target = list(cfg.get("current_dset2", {}).get("existing_sequences", []))
    target_total = int(download_cfg.get("target_total_sequences", 12))
    min_total = int(download_cfg.get("min_total_sequences", 10))
    frame_cap = int(download_cfg.get("max_files_per_sequence", 1200))
    max_bytes = int(download_cfg.get("max_download_bytes", 12884901888))
    inventory = remote.get("sequences", {})

    selected = _select_sequences(
        inventory=inventory,
        existing_target=existing_target,
        local_root=local_root,
        target_total=target_total,
        frame_cap=frame_cap,
    )
    oversized = [
        seq for seq in selected
        if int(inventory.get(seq, {}).get("image_files", 0)) > frame_cap
    ]
    fully_downloadable = [seq for seq in selected if seq not in oversized]
    existing_local_with_images = [
        seq for seq in selected
        if _local_sequence_stats(local_root, seq)["image_count"] > 0
    ]
    pre_existing_images = {seq: _local_sequence_stats(local_root, seq)["image_count"] for seq in selected}

    payload: Dict[str, Any] = {
        "repo_id": repo_id,
        "local_root": str(local_root),
        "existing_sequences": existing_target,
        "new_sequences_selected": [seq for seq in selected if seq not in existing_target],
        "new_sequences_downloaded": [],
        "selected_sequences_total": selected,
        "num_sequences_total": len(selected),
        "download_attempted": True,
        "download_complete": False,
        "total_bytes_downloaded": 0,
        "max_download_bytes_respected": True,
        "access_blocked": False,
        "dependency_blocked": False,
        "blockers": list(remote.get("blockers", [])),
        "oversized_sequences_skipped_for_full_sync": oversized,
        "snapshot_allow_patterns_used": [],
        "repaired_existing_sequences": [],
        "synthetic_timestamps_written": [],
        "remote_timestamps_missing_for_all_selected": all(
            not inventory.get(seq, {}).get("has_timestamps", False) for seq in selected
        ) if selected else False,
    }

    if not selected:
        payload["blockers"].append("NO_ELIGIBLE_REMOTE_SEQUENCES")
        write_json(out_json, payload)
        return payload

    ok, allow_patterns, err = _snapshot_download_selected(
        repo_id=repo_id,
        repo_type=repo_type,
        local_root=local_root,
        selected=fully_downloadable,
    )
    payload["snapshot_allow_patterns_used"] = allow_patterns
    if not ok:
        payload["dependency_blocked"] = "ModuleNotFoundError" in err
        payload["access_blocked"] = not payload["dependency_blocked"]
        payload["blockers"].append(f"SNAPSHOT_DOWNLOAD_FAILED:{err}")
        write_json(out_json, payload)
        return payload

    for seq in fully_downloadable:
        gt_path = local_root / "GroundTruth" / f"{seq}.txt"
        ts_path = local_root / "Timestamps" / f"{seq}_timestamps.txt"
        pose_count = _line_count(gt_path)
        ts_count = _line_count(ts_path)
        if pose_count > 0 and ts_count != pose_count:
            _write_synthetic_timestamps(ts_path, pose_count)
            payload["synthetic_timestamps_written"].append(seq)

    bytes_downloaded = 0
    selected_with_data: List[str] = []
    repaired_existing: List[str] = []
    new_downloaded: List[str] = []
    for seq in selected:
        stats = _local_sequence_stats(local_root, seq)
        image_dir = local_root / "Sequences" / seq
        gt_path = local_root / "GroundTruth" / f"{seq}.txt"
        ts_path = local_root / "Timestamps" / f"{seq}_timestamps.txt"
        if stats["image_count"] > 0 and stats["pose_count"] > 0 and stats["timestamp_count"] > 0:
            selected_with_data.append(seq)
        if seq in fully_downloadable:
            bytes_downloaded += sum(
                path.stat().st_size
                for path in list(image_dir.glob("*.jpg")) + [p for p in (gt_path, ts_path) if p.exists()]
            )
        if seq in existing_target and stats["image_count"] > pre_existing_images.get(seq, 0):
            repaired_existing.append(seq)
        if seq not in existing_target and stats["image_count"] > 0:
            new_downloaded.append(seq)

    payload["selected_sequences_total"] = selected_with_data
    payload["num_sequences_total"] = len(selected_with_data)
    payload["new_sequences_downloaded"] = sorted(new_downloaded)
    payload["repaired_existing_sequences"] = sorted(repaired_existing)
    payload["total_bytes_downloaded"] = int(bytes_downloaded)
    payload["max_download_bytes_respected"] = bytes_downloaded <= max_bytes
    if not payload["max_download_bytes_respected"]:
        payload["blockers"].append("MAX_DOWNLOAD_BYTES_EXCEEDED")
    if len(selected_with_data) < min_total:
        payload["blockers"].append("INSUFFICIENT_SEQUENCES")
    payload["download_complete"] = (
        len(selected_with_data) >= min_total
        and payload["max_download_bytes_respected"]
        and not payload["access_blocked"]
        and not payload["dependency_blocked"]
    )
    write_json(out_json, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--remote-inventory", required=True)
    parser.add_argument("--out-json", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
