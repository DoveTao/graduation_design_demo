#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import quote

from s5e2_adjacent_dense_lib import write_json


REPO_DEFAULT = "chris1004336379/360DVO"
RESULT_DEFAULT = "external_baselines/results/gen2_360dvo_external_adapter/gen2b_download_summary.json"
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


def _pip_show() -> bool:
    proc = subprocess.run(
        ["/home/dovetao/miniconda3/envs/pytorch/bin/python", "-m", "pip", "show", "huggingface_hub"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return proc.returncode == 0


def _download_file(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return
    subprocess.run(
        ["curl", "-sS", "-L", "--http1.1", "--max-time", "120", "-o", str(out_path), url],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _sequence_name(path: str) -> str | None:
    parts = path.split("/")
    if len(parts) >= 3 and parts[0] == "Sequences":
        return parts[1]
    if len(parts) == 2 and parts[0] == "GroundTruth" and parts[1].endswith(".txt"):
        return parts[1][:-4]
    return None


def _build_download_plan(siblings: List[Dict[str, Any]], prefer_sequence: str, max_files: int, max_bytes: int) -> Tuple[str | None, List[Dict[str, Any]], List[str]]:
    del max_bytes  # The mini-sequence plan is bounded by file count and sequence cap.
    images_by_seq: Dict[str, List[str]] = {}
    gt_by_seq: Dict[str, str] = {}
    for node in siblings:
        path = str(node.get("rfilename", ""))
        seq = _sequence_name(path)
        if path.lower().endswith(".jpg") and seq:
            images_by_seq.setdefault(seq, []).append(path)
        elif path.startswith("GroundTruth/") and path.lower().endswith(".txt") and seq:
            gt_by_seq[seq] = path
    candidate_sequences = sorted(
        [seq for seq in images_by_seq if seq in gt_by_seq],
        key=lambda seq: (len(images_by_seq[seq]), seq),
    )
    blockers: List[str] = []
    if not candidate_sequences:
        return None, [], ["NO_SEQUENCE_WITH_IMAGES_AND_GT"]
    if prefer_sequence != "auto":
        if prefer_sequence not in candidate_sequences:
            return None, [], [f"PREFERRED_SEQUENCE_NOT_FOUND:{prefer_sequence}"]
        selected = prefer_sequence
    else:
        selected = candidate_sequences[0]

    selected_images = sorted(images_by_seq[selected])
    planned: List[Dict[str, Any]] = []
    # GT file first.
    planned.append({"path": gt_by_seq[selected]})
    max_image_files = max(2, min(max_files - 2, 101))
    for image_path in selected_images[:max_image_files]:
        if len(planned) >= max_files:
            break
        planned.append({"path": image_path})
    if len(planned) <= 1:
        blockers.append("MAX_BYTES_TOO_SMALL_FOR_MINISEQ")
    return selected, planned, blockers


def _subset_gt_and_write_timestamps(gt_path: Path, timestamps_path: Path, selected_image_names: List[str]) -> Dict[str, Any]:
    lines = [ln.strip() for ln in gt_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    subset = lines[: len(selected_image_names)]
    gt_path.write_text("\n".join(subset) + ("\n" if subset else ""), encoding="utf-8")
    timestamps = [f"{idx * 0.1:.1f}" for idx in range(len(subset))]
    timestamps_path.parent.mkdir(parents=True, exist_ok=True)
    timestamps_path.write_text("\n".join(timestamps) + ("\n" if timestamps else ""), encoding="utf-8")
    return {
        "gt_rows_kept": len(subset),
        "timestamp_rows_written": len(timestamps),
        "timestamp_source": "synthetic_10fps_from_dataset_readme",
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    local_dir = Path(args.local_dir)
    dependency_available = _pip_show()
    dependency_missing_but_fallback_used = not dependency_available
    payload: Dict[str, Any] = {
        "repo_id": args.repo_id,
        "dataset_repo_reachable": False,
        "local_dir": str(local_dir),
        "selected_sequence": None,
        "download_attempted": False,
        "download_complete": False,
        "files_downloaded": [],
        "total_bytes_downloaded": 0,
        "access_blocked": False,
        "dependency_blocked": False,
        "dependency_missing_but_fallback_used": dependency_missing_but_fallback_used,
        "blockers": [],
    }
    try:
        info = _curl_json(HF_INFO_URL.format(repo_id=args.repo_id))
        payload["dataset_repo_reachable"] = True
    except subprocess.CalledProcessError as exc:
        payload["access_blocked"] = True
        payload["blockers"] = ["ACCESS_BLOCKED", exc.stderr.strip() or exc.stdout.strip() or str(exc)]
        write_json(out_path, payload)
        return payload
    except Exception as exc:
        payload["blockers"] = [f"DOWNLOAD_SUMMARY_ERROR:{exc}"]
        write_json(out_path, payload)
        return payload

    sha = str(info.get("sha") or "")
    siblings = list(info.get("siblings", []))
    selected, plan, plan_blockers = _build_download_plan(siblings, args.prefer_sequence, int(args.max_files), int(args.max_bytes))
    payload["selected_sequence"] = selected
    payload["planned_files"] = [str(node["path"]) for node in plan]
    payload["planned_total_bytes"] = None
    if plan_blockers:
        payload["blockers"] = plan_blockers
        write_json(out_path, payload)
        return payload

    if args.dry_run:
        write_json(out_path, payload)
        return payload

    payload["download_attempted"] = True
    try:
        local_dir.mkdir(parents=True, exist_ok=True)
        for node in plan:
            rel_path = str(node["path"])
            url = HF_RESOLVE_CACHE.format(repo_id=args.repo_id, sha=sha, path_q=quote(rel_path, safe=""))
            dst = local_dir / rel_path
            _download_file(url, dst)
            payload["files_downloaded"].append(str(dst))
            payload["total_bytes_downloaded"] += int(dst.stat().st_size)

        if selected is not None:
            gt_path = local_dir / "GroundTruth" / f"{selected}.txt"
            image_dir = local_dir / "Sequences" / selected
            timestamps_path = local_dir / "Timestamps" / f"{selected}_timestamps.txt"
            selected_images = sorted([p.name for p in image_dir.glob("*.jpg")])
            subset_info = _subset_gt_and_write_timestamps(gt_path, timestamps_path, selected_images)
            payload["files_downloaded"].append(str(timestamps_path))
            payload["selected_sequence_image_count"] = len(selected_images)
            payload["selected_sequence_timestamp_file"] = str(timestamps_path)
            payload["subset_info"] = subset_info
        payload["download_complete"] = True
    except subprocess.CalledProcessError as exc:
        payload["blockers"] = ["DOWNLOAD_FAILED", exc.stderr.strip() or exc.stdout.strip() or str(exc)]
    except Exception as exc:
        payload["blockers"] = [f"DOWNLOAD_FAILED:{exc}"]

    write_json(out_path, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=REPO_DEFAULT)
    parser.add_argument("--repo-type", default="dataset")
    parser.add_argument("--local-dir", default="data/360DVO")
    parser.add_argument("--max-files", type=int, default=500)
    parser.add_argument("--max-bytes", type=int, default=5368709120)
    parser.add_argument("--prefer-sequence", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-json", default=RESULT_DEFAULT)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
