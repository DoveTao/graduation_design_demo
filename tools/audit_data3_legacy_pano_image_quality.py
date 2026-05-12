#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image

from miniyaml import load_yaml_like
from s5e2_adjacent_dense_lib import scan_frames, write_json


ALLOWED_CLASSIFICATIONS = [
    "DATA3_LEGACY_IMAGE_QUALITY_RISK_CONFIRMED",
    "DATA3_LEGACY_IMAGE_QUALITY_RISK_SUSPECTED",
    "DATA3_LEGACY_IMAGE_QUALITY_NOT_MAJOR",
    "DATA3_LEGACY_IMAGES_NOT_AVAILABLE",
    "DATA3_INSUFFICIENT_ARTIFACTS",
    "DATA3_ERROR",
]


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _image_metrics(path: str) -> Dict[str, float]:
    img = Image.open(path).convert("RGB").resize((512, 256), resample=Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    gray = arr.mean(axis=2)
    gx = np.diff(gray, axis=1, append=gray[:, -1:])
    gy = np.diff(gray, axis=0, append=gray[-1:, :])
    grad = np.sqrt(gx * gx + gy * gy)
    lap = (
        -4.0 * gray
        + np.roll(gray, 1, axis=0)
        + np.roll(gray, -1, axis=0)
        + np.roll(gray, 1, axis=1)
        + np.roll(gray, -1, axis=1)
    )
    midx = gray.shape[1] // 2
    midy = gray.shape[0] // 2
    seam_v = float(np.mean(np.abs(gray[:, midx - 1] - gray[:, midx])))
    seam_h = float(np.mean(np.abs(gray[midy - 1, :] - gray[midy, :])))
    left_mean = float(np.mean(gray[:, :midx]))
    right_mean = float(np.mean(gray[:, midx:]))
    bright = gray > 0.80
    low_tex = grad < 0.02
    blur_score = float(np.var(lap))
    feature_density = float(np.mean(grad > 0.05))
    low_texture_fraction = float(np.mean(low_tex))
    exposure_disc = abs(left_mean - right_mean)
    bright_low_tex = float(np.mean(bright & low_tex))
    quality_score = float(
        1.0
        - 0.35 * low_texture_fraction
        - 0.15 * min(1.0, seam_v * 5.0)
        - 0.10 * min(1.0, seam_h * 5.0)
        - 0.10 * min(1.0, exposure_disc * 4.0)
        - 0.15 * bright_low_tex
        + 0.15 * min(1.0, feature_density * 3.0)
        + 0.10 * min(1.0, blur_score / 0.02)
    )
    return {
        "low_texture_fraction": low_texture_fraction,
        "blur_score_laplacian": blur_score,
        "seam_score_vertical": seam_v,
        "seam_score_horizontal": seam_h,
        "exposure_discontinuity": exposure_disc,
        "gradient_feature_density": feature_density,
        "bright_low_texture_area_proxy": bright_low_tex,
        "image_quality_score": quality_score,
    }


def _corr(xs: List[float], ys: List[float]) -> float | None:
    if len(xs) < 3 or len(ys) < 3:
        return None
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if float(np.std(x)) < 1.0e-12 or float(np.std(y)) < 1.0e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def run(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = load_yaml_like(Path("configs/dset1_360dvo_main_dataset_migration.yaml")) or {}
    legacy_root = Path(str(cfg.get("legacy_dataset", {}).get("root", "data/PanoramaView")))
    sample_limit = int(cfg.get("legacy_dataset", {}).get("quality_audit", {}).get("sample_limit_per_sequence", 200))
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    sequences = [("scene01", "seq01"), ("scene01", "seq02"), ("scene01", "seq03")]
    legacy_sequences: Dict[str, Any] = {}
    if not legacy_root.exists():
        payload = {
            "experiment": "DATA3_legacy_pano_image_quality_audit",
            "legacy_sequences": {},
            "quality_metrics": {},
            "error_correlation": {},
            "recommendation": {
                "downgrade_legacy_dataset_to_diagnostic": False,
                "use_360dvo_as_main_dataset": False,
            },
            "s5_locked_metrics_policy_unchanged": True,
            "final_classification": "DATA3_LEGACY_IMAGES_NOT_AVAILABLE",
        }
        write_json(out_json, payload)
        Path(args.out_report).write_text("# DATA3 legacy 图像不可用\n", encoding="utf-8")
        return payload

    for scene, seq in sequences:
        frames = scan_frames(Path("data"), scene=scene, seq=seq).get((scene, seq), [])
        sample = frames[: min(sample_limit, len(frames))]
        metrics = [_image_metrics(f.image_path) for f in sample]
        pair_changes = []
        for i in range(max(0, len(sample) - 1)):
            a = np.asarray(Image.open(sample[i].image_path).convert("RGB").resize((256, 128)), dtype=np.float32) / 255.0
            b = np.asarray(Image.open(sample[i + 1].image_path).convert("RGB").resize((256, 128)), dtype=np.float32) / 255.0
            pair_changes.append(float(np.mean(np.abs(a - b))))
        legacy_sequences[f"{scene}/{seq}"] = {
            "num_frames": len(frames),
            "sampled_frames": len(sample),
            "mean_metrics": {
                k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]
            } if metrics else {},
            "pair_photometric_change_mean": float(np.mean(pair_changes)) if pair_changes else None,
        }

    s5e15 = _read_json(Path(args.s5e15_results) / "edge_component_metrics.json")
    struct1b = _read_json(Path(args.struct1b_results) / "edge_component_metrics.json")
    seq03_frames = scan_frames(Path("data"), scene="scene01", seq="seq03").get(("scene01", "seq03"), [])
    edge_quality: List[float] = []
    for i in range(max(0, len(seq03_frames) - 1)):
        ma = _image_metrics(seq03_frames[i].image_path)
        mb = _image_metrics(seq03_frames[i + 1].image_path)
        edge_quality.append(float(0.5 * (ma["image_quality_score"] + mb["image_quality_score"])))
    s5_rows = s5e15.get("metrics_rows", [])
    s1b_rows = struct1b.get("metrics_rows", [])
    n = min(len(edge_quality), len(s5_rows), len(s1b_rows))
    q = edge_quality[:n]
    error_correlation = {
        "quality_vs_s5e15_signed_tdir": _corr(q, [float(r["tdir_deg"]) for r in s5_rows[:n]]),
        "quality_vs_s5e15_tmag_ratio": _corr(q, [float(r["tmag_ratio"]) for r in s5_rows[:n]]),
        "quality_vs_struct1b_signed_tdir": _corr(q, [float(r["tdir_deg"]) for r in s1b_rows[:n]]),
        "quality_vs_struct1b_tmag_ratio": _corr(q, [float(r["tmag_ratio"]) for r in s1b_rows[:n]]),
        "quality_vs_struct1b_path_contribution": _corr(q, [float(r["pred_step_length"]) for r in s1b_rows[:n]]),
    }

    seq03_quality = legacy_sequences["scene01/seq03"]["mean_metrics"]
    risk_confirmed = bool(
        seq03_quality.get("low_texture_fraction", 0.0) > 0.45
        or seq03_quality.get("bright_low_texture_area_proxy", 0.0) > 0.20
        or (error_correlation["quality_vs_struct1b_signed_tdir"] is not None and error_correlation["quality_vs_struct1b_signed_tdir"] < -0.15)
    )
    if risk_confirmed:
        final = "DATA3_LEGACY_IMAGE_QUALITY_RISK_CONFIRMED"
    elif n >= 50:
        final = "DATA3_LEGACY_IMAGE_QUALITY_RISK_SUSPECTED"
    else:
        final = "DATA3_INSUFFICIENT_ARTIFACTS"

    payload = {
        "experiment": "DATA3_legacy_pano_image_quality_audit",
        "legacy_sequences": legacy_sequences,
        "quality_metrics": {
            "metric_definitions": [
                "low_texture_fraction",
                "blur_score_laplacian",
                "seam_score_vertical",
                "seam_score_horizontal",
                "exposure_discontinuity",
                "gradient_feature_density",
                "bright_low_texture_area_proxy",
                "pair_photometric_change",
                "image_quality_score",
            ]
        },
        "error_correlation": error_correlation,
        "recommendation": {
            "downgrade_legacy_dataset_to_diagnostic": final in {"DATA3_LEGACY_IMAGE_QUALITY_RISK_CONFIRMED", "DATA3_LEGACY_IMAGE_QUALITY_RISK_SUSPECTED"},
            "use_360dvo_as_main_dataset": final in {"DATA3_LEGACY_IMAGE_QUALITY_RISK_CONFIRMED", "DATA3_LEGACY_IMAGE_QUALITY_RISK_SUSPECTED"},
        },
        "s5_locked_metrics_policy_unchanged": True,
        "final_classification": final,
    }
    write_json(out_json, payload)
    report = [
        "# DATA3 legacy 全景图像质量审计",
        "",
        "## 1. 审计动机",
        "DATA2 已显示旧数据存在明显 motion shift 风险，因此需要进一步确认 legacy 图像质量是否也是 tdir / tmag / path 退化的重要来源。",
        "",
        "## 2. 图像质量指标定义",
        json.dumps(payload["quality_metrics"], ensure_ascii=False, indent=2),
        "",
        "## 3. 每个旧序列质量统计",
        json.dumps(legacy_sequences, ensure_ascii=False, indent=2),
        "",
        "## 4. 低质量图像与 tdir/tmag/path error 的关系",
        json.dumps(error_correlation, ensure_ascii=False, indent=2),
        "",
        "## 5. 是否确认 legacy 数据质量风险",
        f"- final_classification = {final}",
        "",
        "## 6. 对训练失败的解释",
        "如果 seq03 同时表现出高 low-texture / bright-low-texture 区域与较差方向/尺度误差相关性，则 legacy 数据质量很可能是 DATA2/STRUCT1B 失败的重要背景因素。",
        "",
        "## 7. 建议",
        json.dumps(payload["recommendation"], ensure_ascii=False, indent=2),
    ]
    Path(args.out_report).write_text("\n".join(report) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--data2-checkpoint", required=True)
    parser.add_argument("--s5e15-results", required=True)
    parser.add_argument("--struct1b-results", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-report", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
