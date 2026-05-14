#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config import Config
from dataset_pano_only import _parse_label_13, _scan_seq_frames


DEFAULT_POLICY = REPO_ROOT / "checkpoints" / "S5_clean_tmag_calibration_policy.json"
DEFAULT_MANIFEST = REPO_ROOT / "checkpoints" / "final_clean_candidate_manifest.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "external_baselines" / "dataset" / "scene01_seq03"


def _resolve_path(raw: str | Path | None, default: Path) -> Path:
    if raw is None:
        return default
    path = Path(str(raw))
    return path if path.is_absolute() else (REPO_ROOT / path)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_cfg_from_policy(policy_path: Path) -> Tuple[Config, Dict[str, Any]]:
    policy = _read_json(policy_path)
    ckpt_path = _resolve_path(policy["base_checkpoint_path"], REPO_ROOT / policy["base_checkpoint_path"])
    payload = torch.load(str(ckpt_path), map_location="cpu")
    cfg_dict = payload.get("cfg", {})
    if not isinstance(cfg_dict, dict):
        raise TypeError(f"Unsupported cfg payload type: {type(cfg_dict)}")
    cfg = Config()
    for key, value in cfg_dict.items():
        setattr(cfg, key, value)
    return cfg, policy


def _rotmat_to_quat_xyzw(R: np.ndarray) -> Tuple[float, float, float, float]:
    R = np.asarray(R, dtype=np.float64)
    trace = float(np.trace(R))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(R)))
        if idx == 0:
            s = math.sqrt(max(1.0 + R[0, 0] - R[1, 1] - R[2, 2], 1.0e-12)) * 2.0
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif idx == 1:
            s = math.sqrt(max(1.0 + R[1, 1] - R[0, 0] - R[2, 2], 1.0e-12)) * 2.0
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = math.sqrt(max(1.0 + R[2, 2] - R[0, 0] - R[1, 1], 1.0e-12)) * 2.0
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
    q = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    q /= max(np.linalg.norm(q), 1.0e-12)
    return float(q[0]), float(q[1]), float(q[2]), float(q[3])


def _repo_rel_or_abs(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except Exception:
        return str(path.resolve())


def _choose_test_sequence(cfg: Config) -> Tuple[str, str, Dict[str, Any]]:
    seq_frames = _scan_seq_frames(str(cfg.data_root), scenes=None, seqs=None)
    from dataset_pano_only import _partition_seq_keys

    test_keys, split_summary = _partition_seq_keys(
        list(seq_frames.keys()),
        "test",
        str(cfg.split_by),
        float(cfg.train_ratio),
        int(cfg.split_seed),
    )
    if not test_keys:
        raise RuntimeError("No test sequence found from locked historical split.")
    if len(test_keys) == 1:
        return str(test_keys[0][0]), str(test_keys[0][1]), split_summary
    preferred = ("scene01", "seq03")
    if preferred in test_keys:
        return preferred[0], preferred[1], split_summary
    first = test_keys[0]
    return str(first[0]), str(first[1]), split_summary


def export_external_baseline_dataset(
    *,
    policy_path: Path = DEFAULT_POLICY,
    manifest_path: Path = DEFAULT_MANIFEST,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Dict[str, Any]:
    cfg, policy = _load_cfg_from_policy(policy_path)
    manifest = _read_json(manifest_path)
    scene, seq, split_summary = _choose_test_sequence(cfg)
    seq_frames = _scan_seq_frames(str(cfg.data_root), scenes=[scene], seqs=[seq])
    frames = seq_frames.get((scene, seq), [])
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    image_source_available = len(frames) > 0
    image_lines: List[str] = []
    ts_lines: List[str] = []
    gt_lines: List[str] = []

    for fr in frames:
        pano_path = Path(fr.pano_path)
        ts_val = float(fr.ts_val)
        R_wb, t_w = _parse_label_13(fr.label_path)
        qx, qy, qz, qw = _rotmat_to_quat_xyzw(R_wb)
        image_lines.append(f"{fr.ts_str} {_repo_rel_or_abs(pano_path)}")
        ts_lines.append(fr.ts_str)
        gt_lines.append(
            f"{ts_val:.6f} {float(t_w[0]):.9f} {float(t_w[1]):.9f} {float(t_w[2]):.9f} "
            f"{qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}"
        )

    (images_dir / "README.md").write_text(
        "\n".join(
            [
                "# Images",
                "",
                "This export keeps the original panorama files in place to avoid committing heavy duplicates.",
                "Use `../image_list.txt` to resolve the source image paths for external baselines.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (output_dir / "image_list.txt").write_text("\n".join(image_lines) + ("\n" if image_lines else ""), encoding="utf-8")
    (output_dir / "timestamps.txt").write_text("\n".join(ts_lines) + ("\n" if ts_lines else ""), encoding="utf-8")
    (output_dir / "groundtruth_tum.txt").write_text("\n".join(gt_lines) + ("\n" if gt_lines else ""), encoding="utf-8")

    camera_yaml = "\n".join(
        [
            "camera_model: equirectangular_panorama",
            f"image_width: {int(cfg.W)}",
            f"image_height: {int(cfg.H)}",
            "pinhole_intrinsics_available: false",
            "fx: null",
            "fy: null",
            "cx: null",
            "cy: null",
            "note: This dataset uses panorama images rather than a standard pinhole camera model.",
            "",
        ]
    )
    (output_dir / "camera.yaml").write_text(camera_yaml, encoding="utf-8")

    metadata = {
        "name": "EXT1_external_algorithm_baseline_comparison_dataset_export",
        "final_candidate": manifest["final_candidate_name"],
        "locked_metrics": dict(manifest["final_metrics"]),
        "source_policy": _repo_rel_or_abs(policy_path),
        "source_manifest": _repo_rel_or_abs(manifest_path),
        "data_root": _repo_rel_or_abs(_resolve_path(str(cfg.data_root), REPO_ROOT / str(cfg.data_root))),
        "sequence": {"scene": scene, "seq": seq},
        "historical_split": {
            "split_by": str(cfg.split_by),
            "train_ratio": float(cfg.train_ratio),
            "split_seed": int(cfg.split_seed),
            "test_sequence_keys": [list(x) for x in split_summary.get("selected_seq_keys", [])],
        },
        "input_modality": "monocular panorama / equirectangular RGB",
        "image_source_available": bool(image_source_available),
        "images_exported": False,
        "image_list_exported": True,
        "num_frames": int(len(frames)),
        "timestamps_path": "timestamps.txt",
        "camera_path": "camera.yaml",
        "groundtruth_path": "groundtruth_tum.txt",
        "evaluation_metadata": {
            "same_sequence_required_for_numeric_comparison": True,
            "do_not_compare_published_cross_dataset_numbers": True,
            "alignment_modes": ["none", "se3", "sim3"],
            "path_ratio_is_scale_sensitive": True,
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    readme_lines = [
        f"# External Baseline Dataset Export: {scene}/{seq}",
        "",
        "This export is intended for protocol-compatible external SLAM/VO baseline evaluation.",
        "It does not change the final S5 candidate, the locked metrics, the historical split, or the eval convention.",
        "",
        "## Contents",
        "- `image_list.txt`: source panorama image paths for each timestamp.",
        "- `timestamps.txt`: ordered timestamps for the historical final test sequence.",
        "- `camera.yaml`: equirectangular camera metadata and caveat that pinhole intrinsics are unavailable.",
        "- `groundtruth_tum.txt`: ground-truth trajectory in TUM pose format.",
        "- `metadata.json`: evaluation metadata and protocol caveats.",
        "",
        "## Notes",
        "- Images are not copied to avoid heavy repository artifacts.",
        "- Numeric comparison is valid only after an external baseline is run on this exact sequence under this protocol.",
        "- Published results from different datasets must not be compared directly against S5.",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")

    summary = {
        "output_dir": _repo_rel_or_abs(output_dir),
        "sequence": f"{scene}/{seq}",
        "num_frames": int(len(frames)),
        "image_source_available": bool(image_source_available),
        "image_list_path": _repo_rel_or_abs(output_dir / "image_list.txt"),
        "timestamps_path": _repo_rel_or_abs(output_dir / "timestamps.txt"),
        "camera_path": _repo_rel_or_abs(output_dir / "camera.yaml"),
        "groundtruth_path": _repo_rel_or_abs(output_dir / "groundtruth_tum.txt"),
        "metadata_path": _repo_rel_or_abs(output_dir / "metadata.json"),
    }
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export the historical final test sequence into a lightweight external-baseline dataset bundle.")
    p.add_argument("--policy", default=str(DEFAULT_POLICY), help="Path to locked S5 policy json.")
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Path to final candidate manifest json.")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output dataset directory.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    summary = export_external_baseline_dataset(
        policy_path=_resolve_path(args.policy, DEFAULT_POLICY),
        manifest_path=_resolve_path(args.manifest, DEFAULT_MANIFEST),
        output_dir=_resolve_path(args.output_dir, DEFAULT_OUTPUT_DIR),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
