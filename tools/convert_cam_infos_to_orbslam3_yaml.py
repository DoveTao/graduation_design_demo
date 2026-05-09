#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_AUDIT = REPO_ROOT / "checkpoints/ORB1a_fisheye_dataset_audit.json"
DEFAULT_CAM_INFOS = REPO_ROOT / "data" / "PanoramaView" / "scene01" / "seq03" / "cam_infos.txt"
DEFAULT_RAW_ROOT = REPO_ROOT / "data" / "FisheyeView"
DEFAULT_OUT_JSON = REPO_ROOT / "checkpoints/ORB1b_calibration_conversion_audit.json"
DEFAULT_OUT_REPORT = REPO_ROOT / "reports/orbslam3_fisheye_calibration_conversion_report.md"
DEFAULT_OUT_YAML = REPO_ROOT / "external_baselines/config/orbslam3_fisheye_cam0_scene01_seq03.yaml"
ALLOWED_FINAL_CLASSIFICATIONS = {
    "ORB1B_READY_FITTED_KB8",
    "ORB1B_READY_DIRECT_KB8",
    "ORB1B_CONVERSION_UNSUPPORTED",
    "ORB1B_BLOCKED_BY_DATASET_AUDIT",
    "ORB1B_ERROR",
}
CONVERSION_CLASSIFICATIONS = {
    "DIRECT_KB8_CONVERSION_SUPPORTED",
    "FITTED_KB8_APPROXIMATION_SUPPORTED",
    "CONVERSION_UNSUPPORTED",
}
THRESHOLDS = {"mean_max": 1.0, "p90_max": 2.0, "max_max": 5.0}
YAML_WARNING = """# This file is generated from dataset polynomial fisheye calibration.
# It is a fitted/converted approximation for ORB-SLAM3 compatibility.
# It must not be treated as original factory KannalaBrandt8 calibration."""
DIRECT_COPY_GUARD = "Never copy mapping_coeffs directly into k1/k2/k3/k4."


def resolve_path(raw: str | None, default: Path) -> Path:
    if raw is None:
        return default
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path)


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


def parse_cam_infos(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        rows.append(parse_cam_infos_line(line))
    if len(rows) != 4:
        raise ValueError(f"expected 4 camera rows in cam_infos.txt, found {len(rows)}")
    return rows


def polynomial_z(mapping_coeffs: Sequence[float], rho: np.ndarray) -> np.ndarray:
    z = np.zeros_like(rho, dtype=np.float64)
    power = np.ones_like(rho, dtype=np.float64)
    for coeff in mapping_coeffs:
        z += float(coeff) * power
        power *= rho
    return z


def sample_pixels(width: int, height: int, margin: int = 12, steps: int = 49) -> np.ndarray:
    xs = np.linspace(margin, width - 1 - margin, steps, dtype=np.float64)
    ys = np.linspace(margin, height - 1 - margin, steps, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(xs, ys)
    return np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)


def fit_kb8_approximation(calib: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Any]]:
    width, height = calib["image_size"]
    cx, cy = [float(v) for v in calib["distortion_center"]]
    stretch = np.asarray(calib["stretch_matrix"], dtype=np.float64).reshape(2, 2)
    inv_stretch = np.linalg.inv(stretch)
    pixels = sample_pixels(width, height)
    centered = pixels - np.asarray([cx, cy], dtype=np.float64)
    canonical = centered @ inv_stretch.T
    rho = np.linalg.norm(canonical, axis=1)
    valid = rho > 1.0e-9
    canonical = canonical[valid]
    pixels = pixels[valid]
    rho = rho[valid]
    z = polynomial_z(calib["mapping_coeffs"], rho)
    theta = np.arctan2(rho, z)
    valid = np.isfinite(theta) & (theta > 1.0e-9) & (theta < math.pi)
    canonical = canonical[valid]
    pixels = pixels[valid]
    rho = rho[valid]
    theta = theta[valid]

    x = np.stack([np.ones_like(theta), theta**2, theta**4, theta**6, theta**8], axis=1)
    y = rho / theta
    coeffs, residuals, rank, singular_values = np.linalg.lstsq(x, y, rcond=None)
    focal = float(coeffs[0])
    if not math.isfinite(focal) or abs(focal) < 1.0e-9:
        raise ValueError("KB8 fit produced an invalid focal scale")
    k1, k2, k3, k4 = [float(v / focal) for v in coeffs[1:]]

    rho_pred = theta * (x @ coeffs)
    directions = canonical / rho[:, None]
    canonical_pred = directions * rho_pred[:, None]
    pixels_pred = canonical_pred @ stretch.T + np.asarray([cx, cy], dtype=np.float64)
    errors = np.linalg.norm(pixels_pred - pixels, axis=1)
    stats = {
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "p90": float(np.percentile(errors, 90)),
        "max": float(np.max(errors)),
    }
    params = {
        "fx": focal,
        "fy": focal,
        "cx": cx,
        "cy": cy,
        "k1": k1,
        "k2": k2,
        "k3": k3,
        "k4": k4,
    }
    diagnostics = {
        "sample_count": int(len(errors)),
        "fit_rank": int(rank),
        "singular_values": [float(v) for v in singular_values],
        "residual_sum": float(residuals[0]) if len(residuals) else 0.0,
        "assumed_polynomial_convention": "z = sum(mapping_coeffs[i] * rho**i); theta = atan2(rho, z)",
        "direct_copy_guard": DIRECT_COPY_GUARD,
    }
    return params, stats, diagnostics


def error_within_thresholds(stats: Dict[str, float]) -> bool:
    return (
        stats["mean"] <= THRESHOLDS["mean_max"]
        and stats["p90"] <= THRESHOLDS["p90_max"]
        and stats["max"] <= THRESHOLDS["max_max"]
    )


def write_yaml(path: Path, params: Dict[str, float], calib: Dict[str, Any], args: argparse.Namespace) -> None:
    width, height = calib["image_size"]
    text = f"""{YAML_WARNING}

%YAML:1.0

File.version: "1.0"

Camera.type: "KannalaBrandt8"

Camera1.fx: {params['fx']:.12g}
Camera1.fy: {params['fy']:.12g}
Camera1.cx: {params['cx']:.12g}
Camera1.cy: {params['cy']:.12g}

Camera1.k1: {params['k1']:.12g}
Camera1.k2: {params['k2']:.12g}
Camera1.k3: {params['k3']:.12g}
Camera1.k4: {params['k4']:.12g}

Camera.width: {width}
Camera.height: {height}
Camera.fps: 20.0
Camera.RGB: 1

ORB1b.audit: "fitted KB8 approximation; do not treat as factory calibration"
ORB1b.source_cam_infos: "{resolve_path(args.cam_infos, DEFAULT_CAM_INFOS)}"
ORB1b.raw_root: "{resolve_path(args.raw_root, DEFAULT_RAW_ROOT)}"
ORB1b.scene: "{args.scene}"
ORB1b.seq: "{args.seq}"
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_payload(args: argparse.Namespace) -> Tuple[Dict[str, Any], Optional[Dict[str, float]], Optional[Dict[str, Any]]]:
    input_audit = resolve_path(args.input_audit, DEFAULT_INPUT_AUDIT)
    cam_infos = resolve_path(args.cam_infos, DEFAULT_CAM_INFOS)
    raw_root = resolve_path(args.raw_root, DEFAULT_RAW_ROOT)
    out_yaml = resolve_path(args.out_yaml, DEFAULT_OUT_YAML)
    notes: List[str] = [
        "Source calibration is dataset polynomial fisheye, not native ORB-SLAM3 KannalaBrandt8.",
        DIRECT_COPY_GUARD,
    ]

    if not input_audit.is_file():
        return {
            "experiment": "ORB1b_calibration_conversion_feasibility_audit",
            "input_audit": str(input_audit.relative_to(REPO_ROOT)) if input_audit.is_relative_to(REPO_ROOT) else str(input_audit),
            "camera_index": args.camera_index,
            "source_calibration": {"cam_infos_path": str(cam_infos), "model": "dataset_polynomial_fisheye"},
            "conversion": {
                "target_model": "ORB_SLAM3_KannalaBrandt8",
                "method": "unsupported",
                "classification": "CONVERSION_UNSUPPORTED",
                "reprojection_error_px": None,
                "thresholds": THRESHOLDS,
                "ready_for_orbslam3_yaml": False,
            },
            "output_yaml": {"path": str(out_yaml), "written": False, "status": "not_written"},
            "final_classification": "ORB1B_BLOCKED_BY_DATASET_AUDIT",
            "notes": notes + ["ORB1a checkpoint is missing."],
        }, None, None

    audit = json.loads(input_audit.read_text(encoding="utf-8"))
    if audit.get("classification") != "RAW_FISHEYE_READY":
        return {
            "experiment": "ORB1b_calibration_conversion_feasibility_audit",
            "input_audit": str(input_audit.relative_to(REPO_ROOT)),
            "camera_index": args.camera_index,
            "source_calibration": {"cam_infos_path": str(cam_infos), "model": "dataset_polynomial_fisheye"},
            "conversion": {
                "target_model": "ORB_SLAM3_KannalaBrandt8",
                "method": "unsupported",
                "classification": "CONVERSION_UNSUPPORTED",
                "reprojection_error_px": None,
                "thresholds": THRESHOLDS,
                "ready_for_orbslam3_yaml": False,
            },
            "output_yaml": {"path": str(out_yaml), "written": False, "status": "not_written"},
            "final_classification": "ORB1B_BLOCKED_BY_DATASET_AUDIT",
            "notes": notes + [f"ORB1a classification was {audit.get('classification')!r}, not RAW_FISHEYE_READY."],
        }, None, None

    rows = parse_cam_infos(cam_infos)
    if args.camera_index < 0 or args.camera_index >= len(rows):
        raise ValueError(f"camera-index {args.camera_index} is out of range for {len(rows)} calibration rows")
    calib = rows[args.camera_index]
    params, stats, diagnostics = fit_kb8_approximation(calib)
    ready = error_within_thresholds(stats)
    conversion_classification = "FITTED_KB8_APPROXIMATION_SUPPORTED" if ready else "CONVERSION_UNSUPPORTED"
    final_classification = "ORB1B_READY_FITTED_KB8" if ready else "ORB1B_CONVERSION_UNSUPPORTED"
    output_status = "official_candidate" if ready else "not_written"

    payload = {
        "experiment": "ORB1b_calibration_conversion_feasibility_audit",
        "input_audit": str(input_audit.relative_to(REPO_ROOT)),
        "camera_index": args.camera_index,
        "source_calibration": {
            "cam_infos_path": str(cam_infos),
            "model": "dataset_polynomial_fisheye",
            "image_size": calib["image_size"],
            "distortion_center": calib["distortion_center"],
            "mapping_coeffs": calib["mapping_coeffs"],
            "stretch_matrix": calib["stretch_matrix"],
            "rpy": calib["rpy"],
            "translation": calib["translation"],
        },
        "conversion": {
            "target_model": "ORB_SLAM3_KannalaBrandt8",
            "method": "fitted_approximation" if ready else "unsupported",
            "classification": conversion_classification,
            "reprojection_error_px": stats,
            "thresholds": THRESHOLDS,
            "ready_for_orbslam3_yaml": ready,
            "fitted_parameters": params if ready else None,
            "diagnostics": diagnostics,
        },
        "output_yaml": {"path": str(out_yaml), "written": ready, "status": output_status},
        "final_classification": final_classification,
        "notes": notes
        + [
            "Direct KB8 conversion is not claimed; this audit uses sampled polynomial projection fitting.",
            "The generated YAML is only an ORB-SLAM3 compatibility candidate, not factory KannalaBrandt8 calibration.",
        ],
    }
    return payload, params if ready else None, calib if ready else None


def build_report(payload: Dict[str, Any]) -> str:
    source = payload["source_calibration"]
    conversion = payload["conversion"]
    errors = conversion.get("reprojection_error_px") or {}
    yaml_info = payload["output_yaml"]
    notes = "\n".join(f"- {note}" for note in payload["notes"])
    if errors:
        error_table = f"""| metric | px |
| --- | ---: |
| mean | {errors['mean']:.6f} |
| median | {errors['median']:.6f} |
| p90 | {errors['p90']:.6f} |
| max | {errors['max']:.6f} |"""
    else:
        error_table = "No reprojection error was computed because conversion was blocked before fitting."

    return f"""# ORB1b Calibration Conversion Feasibility Audit

## Executive summary

- Experiment: `{payload['experiment']}`
- Final classification: `{payload['final_classification']}`
- Camera index: `{payload['camera_index']}`
- Ready for ORB-SLAM3 YAML: `{conversion['ready_for_orbslam3_yaml']}`

This audit does not run ORB-SLAM3 and does not produce baseline metrics.

## Source calibration model

- cam_infos path: `{source.get('cam_infos_path')}`
- model: `{source.get('model')}`
- image size: `{source.get('image_size')}`
- distortion center: `{source.get('distortion_center')}`
- mapping coeffs: `{source.get('mapping_coeffs')}`
- stretch matrix: `{source.get('stretch_matrix')}`

The dataset provides a polynomial fisheye projection model. This is not the same thing as native ORB-SLAM3 KannalaBrandt8 calibration.

## Why direct parameter copying is invalid

`mapping_coeffs` describe the dataset polynomial projection. They must not be copied directly into ORB-SLAM3 `k1/k2/k3/k4`; doing so would fabricate a calibration model.

## Conversion / fitting method

- target model: `{conversion['target_model']}`
- method: `{conversion['method']}`
- conversion classification: `{conversion['classification']}`
- thresholds: `{conversion['thresholds']}`

The fitted path samples the polynomial model over the image domain, derives angle/radius pairs under the documented audit assumption, and fits the KB8 radial polynomial by least squares.

## Reprojection error table

{error_table}

## YAML generation status

- path: `{yaml_info['path']}`
- written: `{yaml_info['written']}`
- status: `{yaml_info['status']}`

## Final classification

`{payload['final_classification']}`

## Next-stage recommendation

If the classification is `ORB1B_READY_FITTED_KB8`, cam0 may proceed to ORB1c as a fitted KB8 compatibility candidate. The YAML must remain labeled as converted/fitted calibration. If unsupported, use a derived undistorted pinhole route instead of claiming a direct raw fisheye baseline.

## Notes

{notes}
"""


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Audit cam_infos polynomial fisheye conversion feasibility for ORB-SLAM3.")
    parser.add_argument("--input-audit", default=str(DEFAULT_INPUT_AUDIT.relative_to(REPO_ROOT)))
    parser.add_argument("--cam-infos", default=str(DEFAULT_CAM_INFOS))
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--raw-root", default=str(DEFAULT_RAW_ROOT))
    parser.add_argument("--scene", default="scene01")
    parser.add_argument("--seq", default="seq03")
    parser.add_argument("--out-json", default=str(DEFAULT_OUT_JSON.relative_to(REPO_ROOT)))
    parser.add_argument("--out-report", default=str(DEFAULT_OUT_REPORT.relative_to(REPO_ROOT)))
    parser.add_argument("--out-yaml", default=str(DEFAULT_OUT_YAML.relative_to(REPO_ROOT)))
    args = parser.parse_args(argv)

    out_json = resolve_path(args.out_json, DEFAULT_OUT_JSON)
    out_report = resolve_path(args.out_report, DEFAULT_OUT_REPORT)
    try:
        payload, params, calib = build_payload(args)
        if payload["output_yaml"]["written"] and params is not None and calib is not None:
            write_yaml(resolve_path(args.out_yaml, DEFAULT_OUT_YAML), params, calib, args)
    except Exception as exc:
        payload = {
            "experiment": "ORB1b_calibration_conversion_feasibility_audit",
            "input_audit": str(resolve_path(args.input_audit, DEFAULT_INPUT_AUDIT)),
            "camera_index": args.camera_index,
            "source_calibration": {
                "cam_infos_path": str(resolve_path(args.cam_infos, DEFAULT_CAM_INFOS)),
                "model": "dataset_polynomial_fisheye",
            },
            "conversion": {
                "target_model": "ORB_SLAM3_KannalaBrandt8",
                "method": "unsupported",
                "classification": "CONVERSION_UNSUPPORTED",
                "reprojection_error_px": None,
                "thresholds": THRESHOLDS,
                "ready_for_orbslam3_yaml": False,
            },
            "output_yaml": {"path": str(resolve_path(args.out_yaml, DEFAULT_OUT_YAML)), "written": False, "status": "not_written"},
            "final_classification": "ORB1B_ERROR",
            "notes": [DIRECT_COPY_GUARD, f"Error: {exc}"],
        }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_report.write_text(build_report(payload), encoding="utf-8")
    print(json.dumps({"final_classification": payload["final_classification"], "yaml_written": payload["output_yaml"]["written"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
