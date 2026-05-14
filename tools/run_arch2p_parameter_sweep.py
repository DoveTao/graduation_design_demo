#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from s5e2_adjacent_dense_lib import write_json


PYBIN = "/home/dovetao/miniconda3/envs/pytorch/bin/python"
ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs/arch2_mainline_rotation_compensated_softcorr_geometry.yaml"
RESULTS_DIR = ROOT / "external_baselines/results/arch2p_parameter_sweep"
WORK_CKPT_DIR = ROOT / "checkpoints/_arch2p_runs"


def _parse_cfg(path: Path) -> Dict[str, Any]:
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    lines = []
    for raw in raw_lines:
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        lines.append(raw.rstrip("\n"))

    def parse_scalar(text: str) -> Any:
        text = text.strip()
        if text.lower() == "true":
            return True
        if text.lower() == "false":
            return False
        try:
            if "." in text:
                return float(text)
            return int(text)
        except Exception:
            return text.strip('"')

    def parse_block(start: int, indent: int) -> tuple[Any, int]:
        if lines[start].lstrip().startswith("- "):
            out_list = []
            i = start
            while i < len(lines):
                line = lines[i]
                cur_indent = len(line) - len(line.lstrip(" "))
                if cur_indent != indent or not line.lstrip().startswith("- "):
                    break
                rest = line.lstrip()[2:].strip()
                if rest:
                    out_list.append(parse_scalar(rest))
                    i += 1
                else:
                    val, i = parse_block(i + 1, indent + 2)
                    out_list.append(val)
            return out_list, i
        out_dict: Dict[str, Any] = {}
        i = start
        while i < len(lines):
            line = lines[i]
            cur_indent = len(line) - len(line.lstrip(" "))
            if cur_indent != indent or line.lstrip().startswith("- "):
                break
            key, value = line.strip().split(":", 1)
            value = value.strip()
            if value:
                out_dict[key] = parse_scalar(value)
                i += 1
            else:
                if i + 1 >= len(lines):
                    out_dict[key] = {}
                    i += 1
                    continue
                next_indent = len(lines[i + 1]) - len(lines[i + 1].lstrip(" "))
                val, i = parse_block(i + 1, next_indent)
                out_dict[key] = val
        return out_dict, i

    parsed, _ = parse_block(0, 0)
    return parsed


def _dump_yaml(obj: Dict[str, Any], indent: int = 0) -> str:
    lines: List[str] = []
    pad = " " * indent
    for key, value in obj.items():
        if isinstance(value, dict):
            lines.append(f"{pad}{key}:")
            lines.append(_dump_yaml(value, indent + 2))
        elif isinstance(value, list):
            items = ", ".join(json.dumps(v) for v in value)
            lines.append(f"{pad}{key}: [{items}]")
        elif isinstance(value, bool):
            lines.append(f"{pad}{key}: {'true' if value else 'false'}")
        else:
            lines.append(f"{pad}{key}: {value}")
    return "\n".join(lines)


def _set_nested(cfg: Dict[str, Any], path: List[str], value: Any) -> None:
    cur = cfg
    for key in path[:-1]:
        cur = cur.setdefault(key, {})
    cur[path[-1]] = value


def _build_run_specs(sweep_cfg: Dict[str, Any], preset: str, max_runs: int | None) -> List[Dict[str, Any]]:
    sw = sweep_cfg["sweep"]
    specs: List[Dict[str, Any]] = []
    baseline = {
        "delta_alpha": 0.05,
        "delta_clip_norm": 0.02,
        "high_confidence_quantile": 0.25,
        "softcorr_temperature": 0.05,
        "w_tdir": 1.0,
        "w_path": 0.2,
        "w_delta_l2": 3.0,
        "train_steps": 120,
        "tag": "baseline_arch2",
    }
    specs.append(baseline)
    for v in sw["delta_alpha"]:
        if v != baseline["delta_alpha"]:
            specs.append({**baseline, "delta_alpha": v, "tag": f"alpha_{v}"})
    for v in sw["delta_clip_norm"]:
        if v != baseline["delta_clip_norm"]:
            specs.append({**baseline, "delta_clip_norm": v, "tag": f"clip_{v}"})
    for v in sw["high_confidence_quantile"]:
        if v != baseline["high_confidence_quantile"]:
            specs.append({**baseline, "high_confidence_quantile": v, "tag": f"hq_{v}"})
    for v in sw["softcorr_temperature"]:
        if v != baseline["softcorr_temperature"]:
            specs.append({**baseline, "softcorr_temperature": v, "tag": f"temp_{v}"})
    pressure = [
        {**baseline, "w_tdir": 2.0, "w_path": 0.2, "w_delta_l2": 1.0, "train_steps": 300, "tag": "tdir_up"},
        {**baseline, "w_tdir": 5.0, "w_path": 0.5, "w_delta_l2": 1.0, "train_steps": 300, "tag": "aggressive_dir"},
        {**baseline, "w_tdir": 2.0, "w_path": 0.5, "w_delta_l2": 5.0, "train_steps": 800, "tag": "balanced_long"},
        {**baseline, "delta_alpha": 0.02, "delta_clip_norm": 0.01, "w_tdir": 2.0, "w_path": 0.5, "w_delta_l2": 1.0, "train_steps": 300, "tag": "mid_safe_push"},
        {**baseline, "delta_alpha": 0.01, "delta_clip_norm": 0.005, "high_confidence_quantile": 0.50, "softcorr_temperature": 0.03, "w_tdir": 5.0, "w_path": 1.0, "w_delta_l2": 1.0, "train_steps": 800, "tag": "high_conf_focus"},
    ]
    specs.extend(pressure)
    if preset == "quick":
        limit = 12
    elif preset == "focused":
        limit = 24
    else:
        limit = 48
        # add more combinatorial exploration for full
        extras: List[Dict[str, Any]] = []
        for alpha in sw["delta_alpha"]:
            for temp in sw["softcorr_temperature"]:
                extras.append({**baseline, "delta_alpha": alpha, "softcorr_temperature": temp, "w_tdir": 2.0, "w_path": 0.5, "w_delta_l2": 5.0, "train_steps": 300, "tag": f"full_a{alpha}_t{temp}"})
        specs.extend(extras)
    if max_runs is not None:
        limit = min(limit, int(max_runs))
    dedup: List[Dict[str, Any]] = []
    seen = set()
    for spec in specs:
        key = tuple(sorted((k, json.dumps(v, sort_keys=True)) for k, v in spec.items() if k != "tag"))
        if key in seen:
            continue
        seen.add(key)
        dedup.append(spec)
    return dedup[:limit]


def _make_temp_config(base_cfg: Dict[str, Any], spec: Dict[str, Any], out_path: Path) -> None:
    cfg = json.loads(json.dumps(base_cfg))
    _set_nested(cfg, ["tdir_head", "final_tdir", "max_alpha"], spec["delta_alpha"])
    _set_nested(cfg, ["tdir_head", "delta_clip_norm"], spec["delta_clip_norm"])
    _set_nested(cfg, ["softcorr_geometry", "temperature"], spec["softcorr_temperature"])
    _set_nested(cfg, ["observability_weighting", "high_threshold_quantile"], max(0.0, min(1.0, 1.0 - float(spec["high_confidence_quantile"]))))
    _set_nested(cfg, ["losses", "w_tdir_signed"], spec["w_tdir"])
    _set_nested(cfg, ["losses", "w_path_window"], spec["w_path"])
    _set_nested(cfg, ["losses", "w_delta_tdir_l2"], spec["w_delta_l2"])
    _set_nested(cfg, ["training", "max_steps"], spec["train_steps"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_dump_yaml(cfg) + "\n", encoding="utf-8")


def _run(cmd: List[str]) -> None:
    subprocess.run(cmd, cwd=ROOT, check=True)


def _mean_high_conf_tdir(prov_path: Path) -> float | None:
    if not prov_path.exists():
        return None
    vals = []
    for line in prov_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if float(row.get("gate_value", 0.0)) >= 0.5:
            mp = row.get("metric_preview", {})
            if mp.get("tdir_deg") is not None:
                vals.append(float(mp["tdir_deg"]))
    return None if not vals else float(sum(vals) / len(vals))


def _cleanup_artifacts(run_dir: Path, candidate_dir: Path) -> None:
    if candidate_dir.exists():
        shutil.rmtree(candidate_dir, ignore_errors=True)
    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    sweep_cfg = _parse_cfg(Path(args.config))
    base_cfg = _parse_cfg(BASE_CONFIG)
    specs = _build_run_specs(sweep_cfg, args.preset, args.max_runs)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    WORK_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        payload = {"preset": args.preset, "num_runs": len(specs), "runs": specs}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return payload

    executed: List[Dict[str, Any]] = []
    for idx, spec in enumerate(specs):
        run_id = f"{idx:03d}_{spec['tag']}"
        summary_path = RESULTS_DIR / f"run_{run_id}_summary.json"
        if args.resume and summary_path.exists():
            executed.append(json.loads(summary_path.read_text(encoding="utf-8")))
            continue
        candidate_dir = WORK_CKPT_DIR / f"run_{run_id}"
        run_dir = RESULTS_DIR / f"run_{run_id}"
        temp_cfg = candidate_dir / "arch2p_temp_config.yaml"
        if candidate_dir.exists():
            shutil.rmtree(candidate_dir)
        if run_dir.exists():
            shutil.rmtree(run_dir)
        candidate_dir.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        _make_temp_config(base_cfg, spec, temp_cfg)
        _run([PYBIN, "tools/train_arch2_mainline_softcorr_geometry.py", "--config", str(temp_cfg), "--candidate-dir", str(candidate_dir)])
        _run([
            PYBIN, "tools/export_arch2_adjacent_dense_predictions.py",
            "--scene", "scene01",
            "--seq", "seq03",
            "--candidate", str(candidate_dir),
            "--s5e15-checkpoint", "checkpoints/S5E15_scale_deunderfit_antiparallel_candidate.json",
            "--timestamps", "external_baselines/dataset/scene01_seq03/timestamps.txt",
            "--groundtruth", "external_baselines/dataset/scene01_seq03/groundtruth_tum.txt",
            "--out-tum", str(run_dir / "scene01_seq03_arch2p_tum.txt"),
            "--out-provenance", str(run_dir / "edge_provenance.jsonl"),
            "--out-metrics", str(run_dir / "edge_component_metrics.json"),
        ])
        _run([
            PYBIN, "tools/audit_arch2_pose_convention_softcorr_noop.py",
            "--candidate", str(candidate_dir),
            "--results", str(run_dir),
            "--out-json", str(run_dir / "integrity_audit.json"),
        ])
        _run([
            PYBIN, "tools/evaluate_arch2_traceable_dense.py",
            "--trajectory", str(run_dir / "scene01_seq03_arch2p_tum.txt"),
            "--metrics", str(run_dir / "edge_component_metrics.json"),
            "--integrity-audit", str(run_dir / "integrity_audit.json"),
            "--groundtruth", "external_baselines/dataset/scene01_seq03/groundtruth_tum.txt",
            "--out-json", str(candidate_dir / "run_checkpoint.json"),
            "--out-report", str(candidate_dir / "run_audit_report.md"),
        ])
        ckpt = json.loads((candidate_dir / "run_checkpoint.json").read_text(encoding="utf-8"))
        train = json.loads((candidate_dir / "training_status.json").read_text(encoding="utf-8"))
        audit = json.loads((run_dir / "integrity_audit.json").read_text(encoding="utf-8"))
        comp = ckpt.get("component_metrics", {})
        ext = ckpt.get("external_eval", {})
        high_subset = _mean_high_conf_tdir(run_dir / "edge_provenance.jsonl")
        delta_mean = ckpt.get("delta_tdir_control", {}).get("delta_tdir_norm_mean")
        delta_p90 = ckpt.get("delta_tdir_control", {}).get("delta_tdir_norm_p90")
        mod_frac = None
        nonzero = audit.get("no_op_guard", {}).get("delta_tdir_nonzero_count")
        if nonzero is not None:
            mod_frac = float(nonzero) / 453.0
        summary = {
            "run_id": run_id,
            "parameters": spec,
            "real_training_executed": bool(train.get("real_training_executed")),
            "optimizer_step_count": train.get("optimizer_step_count"),
            "delta_tdir_norm_mean": delta_mean,
            "delta_tdir_norm_p90": delta_p90,
            "modified_edge_fraction": mod_frac,
            "tdir_mean_deg": comp.get("signed_tdir_mean_deg", comp.get("tdir_mean_deg")),
            "anti_parallel_rate": comp.get("anti_parallel_rate"),
            "tmag_median_ratio": comp.get("tmag_median_ratio"),
            "path_ratio": comp.get("path_ratio"),
            "sim3_ate": ext.get("sim3", {}).get("ate"),
            "high_confidence_subset_tdir": high_subset,
            "no_op_risk": bool(audit.get("no_op_guard", {}).get("no_op_risk")),
            "fallback_risk": bool(audit.get("no_op_guard", {}).get("fallback_to_s5e15_only")),
            "uses_eval_gt_for_training": False,
            "uses_orbslam3_teacher": False,
            "improvement_vs_s5e15": ckpt.get("improvement_vs_s5e15", {}),
        }
        write_json(summary_path, summary)
        _cleanup_artifacts(run_dir, candidate_dir)
        executed.append(summary)
    return {"preset": args.preset, "num_runs": len(executed), "runs": executed}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--preset", choices=["quick", "focused", "full"], default="quick")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-runs", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
