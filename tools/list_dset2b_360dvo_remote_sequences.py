#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from miniyaml import load_yaml_like
from s5e2_adjacent_dense_lib import write_json


HF_INFO_URL = "https://huggingface.co/api/datasets/{repo_id}"


def _load_cfg(path: Path) -> Dict[str, Any]:
    return load_yaml_like(path) or {}


def _curl_json(url: str) -> Any:
    proc = subprocess.run(
        ["curl", "-sS", "-L", "--http1.1", "--max-time", "120", url],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return json.loads(proc.stdout)


def _empty_seq_record() -> Dict[str, Any]:
    return {
        "has_images": False,
        "has_groundtruth": False,
        "has_timestamps": False,
        "image_files": 0,
        "groundtruth_files": 0,
        "timestamp_files": 0,
    }


def _collect_files_via_hf_api(repo_id: str, repo_type: str) -> tuple[List[str], bool, List[str], str]:
    blockers: List[str] = []
    try:
        from huggingface_hub import HfApi  # type: ignore

        api = HfApi()
        files = [str(path) for path in api.list_repo_files(repo_id=repo_id, repo_type=repo_type)]
        return files, False, blockers, "huggingface_hub"
    except Exception as exc:
        blockers.append(f"DEPENDENCY_MISSING:{type(exc).__name__}")
        try:
            info = _curl_json(HF_INFO_URL.format(repo_id=repo_id))
            files = [str(node.get("rfilename", "")) for node in info.get("siblings", []) if node.get("rfilename")]
            return files, True, blockers, "http_fallback"
        except Exception as sub_exc:
            blockers.append(f"REMOTE_INVENTORY_FAILED:{type(sub_exc).__name__}:{sub_exc}")
            return [], True, blockers, "unavailable"


def _update_seq_record(seqs: Dict[str, Dict[str, Any]], path: str) -> None:
    parts = path.split("/")
    if len(parts) >= 3 and parts[0] == "Sequences":
        seq = parts[1]
        rec = seqs.setdefault(seq, _empty_seq_record())
        rec["has_images"] = True
        if path.lower().endswith(".jpg"):
            rec["image_files"] += 1
        return
    if len(parts) == 2 and parts[0] == "GroundTruth" and parts[1].lower().endswith(".txt"):
        seq = Path(parts[1]).stem
        rec = seqs.setdefault(seq, _empty_seq_record())
        rec["has_groundtruth"] = True
        rec["groundtruth_files"] += 1
        return
    if len(parts) == 2 and parts[0] == "Timestamps":
        name = parts[1]
        seq = name.replace("_timestamps.txt", "").replace(".txt", "")
        rec = seqs.setdefault(seq, _empty_seq_record())
        rec["has_timestamps"] = True
        rec["timestamp_files"] += 1


def run(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = _load_cfg(Path(args.config))
    repo_id = str(cfg.get("dataset", {}).get("source_repo", "chris1004336379/360DVO"))
    repo_type = str(cfg.get("dataset", {}).get("repo_type", "dataset"))
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    files, dependency_missing, blockers, method = _collect_files_via_hf_api(repo_id=repo_id, repo_type=repo_type)
    seqs: Dict[str, Dict[str, Any]] = {}
    for path in files:
        _update_seq_record(seqs, path)

    payload: Dict[str, Any] = {
        "repo_id": repo_id,
        "repo_type": repo_type,
        "dependency_missing": dependency_missing,
        "inventory_method": method,
        "num_sequences": len(seqs),
        "sequences": dict(sorted(seqs.items())),
        "blockers": blockers,
    }
    write_json(out_json, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-json", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
