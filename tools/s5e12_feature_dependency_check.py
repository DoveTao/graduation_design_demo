#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def _load_simple_yaml(path: Path) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    if not path.exists():
        return data
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip()
    return data


def run(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = _load_simple_yaml(Path(args.config))
    camera_path = Path(cfg.get("camera_intrinsics_path", ""))
    cv2_available = False
    orb_available = False
    sift_available = False
    matcher_available = False
    essential_matrix_available = False
    recover_pose_available = False
    optical_flow_available = False
    cv2_version = None
    cv2_cuda_devices: Any = "unavailable"
    cv2_error = None

    try:
        import cv2  # type: ignore

        cv2_available = True
        cv2_version = getattr(cv2, "__version__", None)
        orb_available = hasattr(cv2, "ORB_create")
        sift_available = hasattr(cv2, "SIFT_create")
        matcher_available = hasattr(cv2, "BFMatcher")
        essential_matrix_available = hasattr(cv2, "findEssentialMat")
        recover_pose_available = hasattr(cv2, "recoverPose")
        optical_flow_available = hasattr(cv2, "calcOpticalFlowPyrLK")
        try:
            cv2_cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
        except Exception as exc:  # pragma: no cover
            cv2_cuda_devices = f"unavailable: {exc}"
    except Exception as exc:  # pragma: no cover
        cv2_error = str(exc)

    kornia_available = False
    kornia_error = None
    try:
        import kornia  # type: ignore  # noqa: F401

        kornia_available = True
    except Exception as exc:  # pragma: no cover
        kornia_error = str(exc)

    camera_intrinsics_available = False
    camera_intrinsics_notes = []
    if camera_path.exists():
        meta = _load_simple_yaml(camera_path)
        txt = camera_path.read_text(encoding="utf-8", errors="ignore")
        lowered = txt.lower()
        explicit_false = str(meta.get("pinhole_intrinsics_available", "")).lower() == "false"
        fx = str(meta.get("fx", "")).lower()
        fy = str(meta.get("fy", "")).lower()
        cx = str(meta.get("cx", "")).lower()
        cy = str(meta.get("cy", "")).lower()
        numeric_like = all(v not in {"", "null", "none"} for v in [fx, fy, cx, cy])
        tokens = ["camera_matrix", "projection_parameters", "distortion_parameters"]
        camera_intrinsics_available = (not explicit_false) and (numeric_like or any(tok in lowered for tok in tokens))
        if camera_intrinsics_available:
            camera_intrinsics_notes.append("检测到相机内参字段，且未发现显式的 intrinsics unavailable 标记。")
        else:
            camera_intrinsics_notes.append("camera.yaml 存在，但当前数据是 panorama / non-pinhole 描述，不能当作可直接用于 essential geometry 的标准 intrinsics。")
    else:
        camera_intrinsics_notes.append("未找到 camera intrinsics 文件。")

    can_extract_real_correspondence_features = bool(
        cv2_available
        and orb_available
        and matcher_available
        and optical_flow_available
    )

    payload = {
        "experiment": "S5E12_real_correspondence_feature_gate",
        "config": str(Path(args.config)),
        "cv2_available": cv2_available,
        "cv2_version": cv2_version,
        "cv2_cuda_devices": cv2_cuda_devices,
        "cv2_error": cv2_error,
        "orb_available": orb_available,
        "sift_available": sift_available,
        "matcher_available": matcher_available,
        "essential_matrix_available": essential_matrix_available,
        "recover_pose_available": recover_pose_available,
        "kornia_available": kornia_available,
        "kornia_error": kornia_error,
        "optical_flow_available": optical_flow_available,
        "camera_intrinsics_available": camera_intrinsics_available,
        "camera_intrinsics_path": str(camera_path) if camera_path else None,
        "camera_intrinsics_notes": camera_intrinsics_notes,
        "can_extract_real_correspondence_features": can_extract_real_correspondence_features,
        "dependency_classification": (
            "S5E12_DEPENDENCY_READY"
            if can_extract_real_correspondence_features
            else "S5E12_DEPENDENCY_BLOCKED"
        ),
        "notes": [
            "PyTorch 继续负责 CUDA 训练，OpenCV 当前只要求支持 CPU 侧 keypoint / match / essential geometry 特征提取。",
            "即使 cv2 cuda devices 为 0，也不影响 S5E12 real correspondence feature gate。",
        ],
    }
    out_path = Path(cfg.get("output_json", args.out_json or "external_baselines/results/s5e12_feature_gate/dependency_check.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--out-json", default="")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
