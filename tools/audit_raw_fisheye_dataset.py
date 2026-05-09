#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EXPORTED_BASELINE_DIR = REPO_ROOT / "external_baselines" / "dataset" / "scene01_seq03"
DEFAULT_OUT_JSON = REPO_ROOT / "checkpoints/ORB1a_fisheye_dataset_audit.json"
DEFAULT_OUT_REPORT = REPO_ROOT / "reports/orbslam3_fisheye_dataset_audit.md"
DEFAULT_PANORAMA_ROOT = REPO_ROOT / "data" / "PanoramaView"
ALLOWED_CLASSIFICATIONS = {
    "RAW_FISHEYE_READY",
    "RAW_FISHEYE_MISSING",
    "CALIBRATION_PARSE_FAILED",
    "TIMESTAMP_ALIGNMENT_ISSUE",
    "IMAGE_SIZE_MISMATCH",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
TIMESTAMP_RE = re.compile(r"(?<!\d)(\d{10}(?:\.\d+)?)(?!\d)")


def resolve_path(raw: str | None, default: Path | None = None) -> Optional[Path]:
    if raw is None:
        return default
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path)


def discover_raw_root(scene: str, seq: str) -> Tuple[Optional[Path], List[str]]:
    candidates = [
        REPO_ROOT / "data" / "FisheyeView",
        Path.home() / "Downloads" / "FisheyeView",
    ]
    notes: List[str] = []
    for root in candidates:
        if (root / scene / seq).is_dir():
            notes.append(f"Selected discovered raw root: {root}")
            return root, notes
        if root.exists():
            notes.append(f"Candidate exists but missing {scene}/{seq}: {root}")

    search_roots = [Path.home(), REPO_ROOT]
    found: List[Path] = []
    for base in search_roots:
        if not base.exists():
            continue
        try:
            for path in base.rglob("*"):
                if len(path.parts) - len(base.parts) > 8:
                    continue
                if path.is_dir() and path.name.lower() == "fisheyeview":
                    found.append(path)
        except OSError as exc:
            notes.append(f"Discovery skipped part of {base}: {exc}")
    for root in sorted(set(found)):
        if (root / scene / seq).is_dir():
            notes.append(f"Selected discovered raw root from recursive scan: {root}")
            return root, notes
    if found:
        notes.append("Found FisheyeView candidates, but none contained requested scene/seq.")
    else:
        notes.append("No raw fisheye root discovered.")
    return None, notes


def numeric_values(text: str) -> List[float]:
    return [float(x) for x in re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text)]


def parse_cam_infos_line(line: str) -> Dict[str, Any]:
    values = numeric_values(line)
    if len(values) != 18:
        raise ValueError(f"expected 18 numeric values, found {len(values)}")
    return {
        "mapping_coeffs": values[0:4],
        "image_size": [int(round(values[4])), int(round(values[5]))],
        "distortion_center": values[6:8],
        "stretch_matrix": values[8:12],
        "rpy": values[12:15],
        "translation": values[15:18],
    }


def parse_cam_infos(path: Path) -> Tuple[bool, List[Dict[str, Any]], str]:
    if not path.is_file():
        return False, [], "cam_infos.txt not found"
    rows: List[Dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            rows.append(parse_cam_infos_line(line))
        except ValueError as exc:
            return False, rows, str(exc)
    if len(rows) != 4:
        return False, rows, f"expected 4 calibration rows, found {len(rows)}"
    return True, rows, "ok"


def resolve_cam_infos_path(raw_root: Optional[Path], scene: str, seq: str) -> Tuple[Path, List[str]]:
    notes: List[str] = []
    candidates: List[Path] = []
    if raw_root is not None:
        candidates.append(raw_root / scene / seq / "cam_infos.txt")
        if raw_root.name.lower() == "fisheyeview":
            candidates.append(raw_root.parent / "PanoramaView" / scene / seq / "cam_infos.txt")
    candidates.append(DEFAULT_PANORAMA_ROOT / scene / seq / "cam_infos.txt")

    seen = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if path.is_file():
            if raw_root is not None and path != raw_root / scene / seq / "cam_infos.txt":
                notes.append(f"Using panorama-side cam_infos.txt: {path}")
            else:
                notes.append(f"Using raw-side cam_infos.txt: {path}")
            return path, notes
    fallback = candidates[0] if candidates else Path("__missing__")
    notes.append("cam_infos.txt was not found in raw or panorama-side scene/seq folders.")
    return fallback, notes


def list_images(path: Path) -> List[Path]:
    if not path.exists():
        return []
    if path.is_file():
        return [path] if path.suffix.lower() in IMAGE_SUFFIXES else []
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def discover_camera_streams(scene_seq_dir: Path) -> Dict[str, Dict[str, Any]]:
    streams: Dict[str, Dict[str, Any]] = {}
    if not scene_seq_dir.is_dir():
        return streams

    dir_patterns = [
        re.compile(r"^(?:cam|camera|image|images|img)[_-]?([0-3])$", re.IGNORECASE),
        re.compile(r"^([0-3])$"),
    ]
    for child in sorted(scene_seq_dir.iterdir()):
        if not child.is_dir():
            continue
        for pattern in dir_patterns:
            match = pattern.match(child.name)
            if match:
                cam = f"cam{match.group(1)}"
                images = list_images(child)
                if images:
                    streams[cam] = {
                        "discovery_pattern": "directory",
                        "path": str(child),
                        "images": images,
                    }
                break

    flat: Dict[str, List[Path]] = {}
    flat_patterns = [
        re.compile(r"^(?:img|cam|camera|image)[_-]?([0-3])[_-].*", re.IGNORECASE),
        re.compile(r"^([0-3])[_-].*"),
    ]
    for image in list_images(scene_seq_dir):
        for pattern in flat_patterns:
            match = pattern.match(image.name)
            if match:
                flat.setdefault(f"cam{match.group(1)}", []).append(image)
                break
    for cam, images in sorted(flat.items()):
        if cam not in streams or len(images) > len(streams[cam]["images"]):
            streams[cam] = {
                "discovery_pattern": "flat_filename_prefix",
                "path": str(scene_seq_dir),
                "images": sorted(images),
            }
    return streams


def read_image_size(path: Path) -> Optional[List[int]]:
    try:
        from PIL import Image  # type: ignore

        with Image.open(path) as image:
            width, height = image.size
            return [int(width), int(height)]
    except Exception:
        pass

    try:
        with path.open("rb") as handle:
            header = handle.read(24)
            if len(header) >= 24 and header.startswith(b"\x89PNG\r\n\x1a\n"):
                width, height = struct.unpack(">II", header[16:24])
                return [int(width), int(height)]
            handle.seek(0)
            if handle.read(2) != b"\xff\xd8":
                return None
            while True:
                marker_start = handle.read(1)
                if marker_start != b"\xff":
                    return None
                marker = handle.read(1)
                while marker == b"\xff":
                    marker = handle.read(1)
                if marker in {b"\xc0", b"\xc1", b"\xc2", b"\xc3", b"\xc5", b"\xc6", b"\xc7", b"\xc9", b"\xca", b"\xcb", b"\xcd", b"\xce", b"\xcf"}:
                    handle.read(3)
                    height, width = struct.unpack(">HH", handle.read(4))
                    return [int(width), int(height)]
                size_raw = handle.read(2)
                if len(size_raw) != 2:
                    return None
                segment_size = struct.unpack(">H", size_raw)[0]
                handle.seek(segment_size - 2, 1)
    except Exception:
        return None


def parse_timestamp_from_name(path: Path) -> Optional[float]:
    match = TIMESTAMP_RE.search(path.stem)
    return float(match.group(1)) if match else None


def list_timestamped_files(path: Path, prefix: str, suffix: str) -> List[Path]:
    if not path.is_dir():
        return []
    return sorted(p for p in path.iterdir() if p.is_file() and p.name.startswith(prefix) and p.suffix.lower() == suffix)


def read_timestamps(path: Path) -> List[float]:
    if not path.is_file():
        return []
    out: List[float] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            out.append(float(line.split()[0]))
        except (ValueError, IndexError):
            continue
    return out


def count_tum_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    count = 0
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and len(line.split()) >= 8:
            count += 1
    return count


def compare_timestamps(raw_ts: Sequence[float], exported_ts: Sequence[float], tolerance: float = 1.0e-6) -> str:
    if not raw_ts:
        return "raw timestamps not parseable"
    if not exported_ts:
        return "exported timestamps missing"
    if abs(len(raw_ts) - len(exported_ts)) > max(3, int(0.05 * max(len(raw_ts), len(exported_ts)))):
        return "count mismatch"
    paired = zip(sorted(raw_ts), sorted(exported_ts))
    max_delta = max((abs(a - b) for a, b in paired), default=float("inf"))
    if max_delta <= tolerance:
        return "aligned_exact"
    if max_delta <= 1.0e-3:
        return "aligned_with_small_tolerance"
    return f"timestamp mismatch: max_delta={max_delta:.6f}s"


def exported_baseline_summary(root: Path) -> Dict[str, Any]:
    gt = root / "groundtruth_tum.txt"
    timestamps = root / "timestamps.txt"
    image_list = root / "image_list.txt"
    metadata = root / "metadata.json"
    camera_yaml = root / "camera.yaml"
    return {
        "path": str(root),
        "groundtruth_tum_exists": gt.is_file(),
        "timestamps_exists": timestamps.is_file(),
        "image_list_exists": image_list.is_file(),
        "metadata_exists": metadata.is_file(),
        "camera_yaml_exists": camera_yaml.is_file(),
        "num_gt_poses": count_tum_rows(gt),
        "num_exported_timestamps": len(read_timestamps(timestamps)),
    }


def build_report(payload: Dict[str, Any]) -> str:
    streams = payload["camera_streams"]
    stream_lines = []
    for cam, info in streams.items():
        stream_lines.append(
            f"- {cam}: {info['num_images']} images, path `{info['path']}`, pattern `{info['discovery_pattern']}`"
        )
    if not stream_lines:
        stream_lines.append("- No camera image streams discovered.")

    size_lines = []
    for cam, info in streams.items():
        size_lines.append(
            f"- {cam}: observed={info['image_size_observed']}, calibration={info['image_size_calibration']}, match={info['image_size_match']}"
        )
    if not size_lines:
        size_lines.append("- No image size validation was possible.")

    notes = "\n".join(f"- {note}" for note in payload["notes"]) or "- None."
    candidate = payload["orbslam3_candidate_input"]
    candidate_notes = "\n".join(f"- {note}" for note in candidate["notes"]) or "- None."
    exported = payload["exported_baseline_dataset"]
    align = payload["timestamp_alignment"]
    labels = payload["panorama_side_dataset"]

    return f"""# ORB1a Raw Fisheye Dataset Audit

## Executive summary

- Experiment: `{payload['experiment']}`
- Final classification: `{payload['classification']}`
- Candidate ORB-SLAM3 monocular fisheye input: `{candidate['candidate_camera']}`
- Ready for next stage: `{candidate['ready_for_next_stage']}`

This audit does not run ORB-SLAM3 and does not produce baseline metrics.

## Dataset root discovered / provided

- Raw root: `{payload['raw_root']}`
- Scene: `{payload['scene']}`
- Sequence: `{payload['seq']}`

## scene01/seq03 availability

- scene/seq exists: `{payload['scene_seq_exists']}`

## Camera stream inventory

{chr(10).join(stream_lines)}

## cam_infos.txt parse result

- cam_infos.txt exists: `{payload['cam_infos_exists']}`
- parse ok: `{payload['cam_infos_parse_ok']}`
- cameras in calibration: `{payload['num_cameras_in_calibration']}`
- cam_infos source: `{payload['cam_infos_path']}`
- parse status: `{payload['cam_infos_parse_status']}`

## Image size validation

{chr(10).join(size_lines)}

## Timestamp / GT alignment result

- raw timestamps parseable: `{align['raw_timestamps_parseable']}`
- raw timestamps: `{align['num_raw_timestamps']}`
- exported timestamps: `{align['num_exported_timestamps']}`
- alignment status: `{align['alignment_status']}`
- groundtruth poses: `{exported['num_gt_poses']}`

## Panorama-side labels / cam info

- panorama scene/seq path: `{labels['path']}`
- panorama images: `{labels['num_panorama_images']}`
- labels: `{labels['num_labels']}`
- labels align with exported timestamps: `{labels['labels_align_with_exported']}`
- panorama images align with exported timestamps: `{labels['panorama_align_with_exported']}`

## Exported baseline dataset

- groundtruth_tum.txt: `{exported['groundtruth_tum_exists']}`
- timestamps.txt: `{exported['timestamps_exists']}`
- image_list.txt: `{exported['image_list_exists']}`
- metadata.json: `{exported['metadata_exists']}`
- camera.yaml: `{exported['camera_yaml_exists']}`

## ORB-SLAM3 next-stage feasibility

- candidate camera: `{candidate['candidate_camera']}`
- ready for next stage: `{candidate['ready_for_next_stage']}`

{candidate_notes}

## Notes

{notes}

## Final classification

`{payload['classification']}`
"""


def audit(args: argparse.Namespace) -> Dict[str, Any]:
    notes: List[str] = []
    raw_root = resolve_path(args.raw_root)
    if raw_root is None:
        raw_root, discovery_notes = discover_raw_root(args.scene, args.seq)
        notes.extend(discovery_notes)
    else:
        notes.append(f"Using provided raw root: {raw_root}")

    scene_seq_dir = raw_root / args.scene / args.seq if raw_root is not None else None
    scene_seq_exists = bool(scene_seq_dir and scene_seq_dir.is_dir())
    cam_infos_path, cam_info_notes = resolve_cam_infos_path(raw_root, args.scene, args.seq)
    notes.extend(cam_info_notes)
    cam_infos_ok, calibration, cam_infos_status = parse_cam_infos(cam_infos_path)
    streams_raw = discover_camera_streams(scene_seq_dir) if scene_seq_dir is not None else {}
    panorama_seq_dir = DEFAULT_PANORAMA_ROOT / args.scene / args.seq

    camera_streams: Dict[str, Any] = {}
    raw_timestamps: List[float] = []
    any_size_mismatch = False
    for cam in [f"cam{i}" for i in range(4)]:
        raw = streams_raw.get(cam)
        images = raw["images"] if raw else []
        first_image = images[0] if images else None
        observed_size = read_image_size(first_image) if first_image else None
        calib_index = int(cam[-1])
        calibration_size = calibration[calib_index]["image_size"] if cam_infos_ok and calib_index < len(calibration) else None
        size_match = bool(observed_size == calibration_size) if observed_size and calibration_size else False
        if observed_size and calibration_size and not size_match:
            any_size_mismatch = True
        timestamps = [ts for ts in (parse_timestamp_from_name(p) for p in images) if ts is not None]
        if cam == "cam0":
            raw_timestamps = timestamps
        camera_streams[cam] = {
            "path": raw["path"] if raw else None,
            "discovery_pattern": raw["discovery_pattern"] if raw else None,
            "num_images": len(images),
            "first_image": str(first_image) if first_image else None,
            "image_size_observed": observed_size,
            "image_size_calibration": calibration_size,
            "image_size_match": size_match,
            "num_parseable_timestamps": len(timestamps),
        }

    exported_dir = resolve_path(args.exported_baseline_dir, DEFAULT_EXPORTED_BASELINE_DIR)
    assert exported_dir is not None
    exported = exported_baseline_summary(exported_dir)
    exported_timestamps = read_timestamps(exported_dir / "timestamps.txt")
    alignment_status = compare_timestamps(raw_timestamps, exported_timestamps)
    alignment_ok = alignment_status in {"aligned_exact", "aligned_with_small_tolerance"}
    panorama_images = list_timestamped_files(panorama_seq_dir, "panorama_", ".jpg")
    label_files = list_timestamped_files(panorama_seq_dir, "label_", ".txt")
    panorama_timestamps = [ts for ts in (parse_timestamp_from_name(p) for p in panorama_images) if ts is not None]
    label_timestamps = [ts for ts in (parse_timestamp_from_name(p) for p in label_files) if ts is not None]
    panorama_alignment = compare_timestamps(panorama_timestamps, exported_timestamps)
    label_alignment = compare_timestamps(label_timestamps, exported_timestamps)

    if raw_root is None or not scene_seq_exists or not any(info["num_images"] for info in camera_streams.values()):
        classification = "RAW_FISHEYE_MISSING"
    elif not cam_infos_ok:
        classification = "CALIBRATION_PARSE_FAILED"
    elif any_size_mismatch:
        classification = "IMAGE_SIZE_MISMATCH"
    elif not alignment_ok:
        classification = "TIMESTAMP_ALIGNMENT_ISSUE"
    else:
        classification = "RAW_FISHEYE_READY"

    candidate_notes: List[str] = []
    cam0 = camera_streams["cam0"]
    if cam0["num_images"] == 0:
        candidate_notes.append("cam0 stream was not discovered.")
    if not cam_infos_ok:
        candidate_notes.append("cam_infos.txt is missing or not parseable; no ORB-SLAM3 fisheye YAML should be generated yet.")
    if cam0["image_size_observed"] and cam0["image_size_calibration"] and not cam0["image_size_match"]:
        candidate_notes.append("cam0 observed image size does not match calibration.")
    if not alignment_ok:
        candidate_notes.append(f"cam0 raw timestamps are not aligned with exported timestamps: {alignment_status}.")
    if cam0["num_images"] > 0 and cam_infos_ok and cam0["image_size_match"] and alignment_ok:
        candidate_notes.append("cam0 is a plausible monocular fisheye input for the next stage.")

    payload: Dict[str, Any] = {
        "experiment": "ORB1a_raw_fisheye_dataset_audit",
        "classification": classification,
        "allowed_classifications": sorted(ALLOWED_CLASSIFICATIONS),
        "raw_root": str(raw_root) if raw_root is not None else None,
        "scene": args.scene,
        "seq": args.seq,
        "scene_seq_exists": scene_seq_exists,
        "cam_infos_exists": cam_infos_path.is_file(),
        "cam_infos_path": str(cam_infos_path) if cam_infos_path.is_file() else None,
        "cam_infos_parse_ok": cam_infos_ok,
        "cam_infos_parse_status": cam_infos_status,
        "num_cameras_in_calibration": len(calibration),
        "camera_streams": camera_streams,
        "exported_baseline_dataset": exported,
        "panorama_side_dataset": {
            "path": str(panorama_seq_dir),
            "exists": panorama_seq_dir.is_dir(),
            "num_panorama_images": len(panorama_images),
            "num_labels": len(label_files),
            "num_panorama_timestamps": len(panorama_timestamps),
            "num_label_timestamps": len(label_timestamps),
            "panorama_alignment_status": panorama_alignment,
            "label_alignment_status": label_alignment,
            "panorama_align_with_exported": panorama_alignment in {"aligned_exact", "aligned_with_small_tolerance"},
            "labels_align_with_exported": label_alignment in {"aligned_exact", "aligned_with_small_tolerance"},
        },
        "timestamp_alignment": {
            "raw_timestamps_parseable": bool(raw_timestamps),
            "num_raw_timestamps": len(raw_timestamps),
            "num_exported_timestamps": len(exported_timestamps),
            "alignment_status": alignment_status,
        },
        "orbslam3_candidate_input": {
            "candidate_camera": "cam0",
            "ready_for_next_stage": classification == "RAW_FISHEYE_READY",
            "notes": candidate_notes,
        },
        "notes": notes,
    }
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Audit raw fisheye dataset availability and alignment.")
    parser.add_argument("--raw-root", default=None)
    parser.add_argument("--scene", default="scene01")
    parser.add_argument("--seq", default="seq03")
    parser.add_argument("--exported-baseline-dir", default=str(DEFAULT_EXPORTED_BASELINE_DIR.relative_to(REPO_ROOT)))
    parser.add_argument("--out-json", default=str(DEFAULT_OUT_JSON.relative_to(REPO_ROOT)))
    parser.add_argument("--out-report", default=str(DEFAULT_OUT_REPORT.relative_to(REPO_ROOT)))
    args = parser.parse_args(argv)

    payload = audit(args)
    out_json = resolve_path(args.out_json, DEFAULT_OUT_JSON)
    out_report = resolve_path(args.out_report, DEFAULT_OUT_REPORT)
    assert out_json is not None and out_report is not None
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_report.write_text(build_report(payload), encoding="utf-8")
    print(json.dumps({"classification": payload["classification"], "raw_root": payload["raw_root"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
