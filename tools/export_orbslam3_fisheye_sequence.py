#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ORB1A = REPO_ROOT / "checkpoints/ORB1a_fisheye_dataset_audit.json"
DEFAULT_ORB1B = REPO_ROOT / "checkpoints/ORB1b_calibration_conversion_audit.json"
DEFAULT_ORB0B = REPO_ROOT / "checkpoints/ORB0b_orbslam3_build_remediation.json"
DEFAULT_RAW_ROOT = REPO_ROOT / "data/FisheyeView"
DEFAULT_TIMESTAMPS = REPO_ROOT / "external_baselines/dataset/scene01_seq03/timestamps.txt"
DEFAULT_SETTINGS = REPO_ROOT / "external_baselines/config/orbslam3_fisheye_cam0_scene01_seq03.yaml"
DEFAULT_OUT_DIR = REPO_ROOT / "external_baselines/results/orbslam3_fisheye_cam0"
TIMESTAMP_RE = re.compile(r"(?<!\d)(\d{10}(?:\.\d+)?)(?!\d)")


def resolve_path(raw: str | None, default: Path) -> Path:
    if raw is None:
        return default
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path)


def read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_timestamp(path: Path) -> str | None:
    match = TIMESTAMP_RE.search(path.stem)
    return match.group(1) if match else None


def read_timestamps(path: Path) -> List[str]:
    out: List[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            out.append(line.split()[0])
    return out


def check_preconditions(orb1a: Path, orb1b: Path, orb0b: Path) -> Dict[str, object]:
    a = read_json(orb1a)
    b = read_json(orb1b)
    z = read_json(orb0b)
    passed = (
        a.get("classification") == "RAW_FISHEYE_READY"
        and b.get("final_classification") in {"ORB1B_READY_FITTED_KB8", "ORB1B_READY_DIRECT_KB8"}
        and z.get("final_classification") == "ORB0_READY_AFTER_REMEDIATION"
        and z.get("ready_for_orb1c") is True
    )
    return {
        "orb1a_classification": a.get("classification"),
        "orb1b_classification": b.get("final_classification"),
        "orb0b_classification": z.get("final_classification"),
        "orb0b_ready_for_orb1c": z.get("ready_for_orb1c"),
        "passed": passed,
    }


def export_sequence(raw_root: Path, scene: str, seq: str, timestamps_path: Path, out_dir: Path) -> Dict[str, object]:
    seq_dir = raw_root / scene / seq
    images = sorted(seq_dir.glob("img_0_*.jpg"), key=lambda p: parse_timestamp(p) or "")
    ts_to_image = {parse_timestamp(path): path for path in images if parse_timestamp(path) is not None}
    timestamps = read_timestamps(timestamps_path)
    ordered_images: List[Path] = []
    missing: List[str] = []
    for timestamp in timestamps:
        image = ts_to_image.get(timestamp)
        if image is None:
            missing.append(timestamp)
        else:
            ordered_images.append(image)
    if missing or len(ordered_images) != len(timestamps):
        raise RuntimeError(f"cam0 image/timestamp alignment failed: missing={len(missing)} count_images={len(ordered_images)} count_timestamps={len(timestamps)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    images_out = out_dir / "scene01_seq03_images.txt"
    timestamps_out = out_dir / "scene01_seq03_timestamps.txt"
    images_out.write_text("\n".join(str(p) for p in ordered_images) + "\n", encoding="utf-8")
    timestamps_out.write_text("\n".join(timestamps) + "\n", encoding="utf-8")
    return {
        "images_path": str(images_out),
        "timestamps_path": str(timestamps_out),
        "num_images": len(ordered_images),
        "num_timestamps": len(timestamps),
    }


def ensure_runtime_settings(settings_yaml: Path, out_dir: Path) -> Path:
    text = settings_yaml.read_text(encoding="utf-8")
    preamble = ""
    marker = "%YAML:1.0"
    if marker in text and not text.lstrip().startswith(marker):
        before, after = text.split(marker, 1)
        preamble = before.strip()
        text = marker + after
    text = re.sub(r"(?m)^Camera\.fps:\s*20\.0\s*$", "Camera.fps: 20", text)
    additions = []
    if "ORBextractor.nFeatures" not in text:
        additions.append(
            """
# Runtime ORB extractor parameters added by ORB1c without modifying the ORB1b source YAML.
ORBextractor.nFeatures: 1500
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 8
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7
"""
        )
    if "Viewer.KeyFrameSize" not in text:
        additions.append(
            """
# Runtime viewer parameters required by ORB-SLAM3 settings parser. Viewer is disabled in ORB1c.
Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1.0
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -3.5
Viewer.ViewpointF: 500.0
"""
        )
    runtime_yaml = out_dir / "scene01_seq03_orbslam3_runtime.yaml"
    if preamble:
        text = text.rstrip() + "\n\n# ORB1b source YAML warning moved below the YAML directive for OpenCV FileStorage compatibility.\n" + preamble + "\n"
    runtime_yaml.write_text(text.rstrip() + "\n" + "\n".join(additions).rstrip() + "\n", encoding="utf-8")
    return runtime_yaml


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export cam0 raw fisheye sequence for ORB-SLAM3.")
    parser.add_argument("--orb1a", default=str(DEFAULT_ORB1A.relative_to(REPO_ROOT)))
    parser.add_argument("--orb1b", default=str(DEFAULT_ORB1B.relative_to(REPO_ROOT)))
    parser.add_argument("--orb0b", default=str(DEFAULT_ORB0B.relative_to(REPO_ROOT)))
    parser.add_argument("--raw-root", default=str(DEFAULT_RAW_ROOT))
    parser.add_argument("--scene", default="scene01")
    parser.add_argument("--seq", default="seq03")
    parser.add_argument("--timestamps", default=str(DEFAULT_TIMESTAMPS.relative_to(REPO_ROOT)))
    parser.add_argument("--settings-yaml", default=str(DEFAULT_SETTINGS.relative_to(REPO_ROOT)))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR.relative_to(REPO_ROOT)))
    args = parser.parse_args(argv)

    out_dir = resolve_path(args.out_dir, DEFAULT_OUT_DIR)
    try:
        prechecks = check_preconditions(
            resolve_path(args.orb1a, DEFAULT_ORB1A),
            resolve_path(args.orb1b, DEFAULT_ORB1B),
            resolve_path(args.orb0b, DEFAULT_ORB0B),
        )
        if not prechecks["passed"]:
            print(json.dumps({"status": "ORB1C_BLOCKED_BY_PRECHECK", "prechecks": prechecks}, sort_keys=True))
            return 2
        exported = export_sequence(
            resolve_path(args.raw_root, DEFAULT_RAW_ROOT),
            args.scene,
            args.seq,
            resolve_path(args.timestamps, DEFAULT_TIMESTAMPS),
            out_dir,
        )
        runtime_yaml = ensure_runtime_settings(resolve_path(args.settings_yaml, DEFAULT_SETTINGS), out_dir)
        payload = {"status": "ok", "prechecks": prechecks, "exported": exported, "runtime_yaml": str(runtime_yaml)}
        print(json.dumps(payload, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "ORB1C_SEQUENCE_EXPORT_FAILED", "error": str(exc)}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
