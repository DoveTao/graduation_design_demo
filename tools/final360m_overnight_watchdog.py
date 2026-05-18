#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parent.parent


def _run(cmd: List[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=False)


def _read_tail(path: Path, lines: int = 120) -> str:
    if not path.exists():
        return ""
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return ""
    return "\n".join(content[-lines:])


def _json_dump(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_md(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _is_process_alive(pid: int) -> Dict[str, Any]:
    proc = _run(["ps", "-p", str(pid), "-o", "pid=,etime=,cmd="])
    text = proc.stdout.strip()
    alive = bool(text)
    elapsed = None
    if alive:
        parts = text.split(maxsplit=2)
        if len(parts) >= 2:
            elapsed = parts[1]
    return {"alive": alive, "ps_line": text, "elapsed": elapsed}


def _disk_free_bytes() -> int:
    return int(shutil.disk_usage(REPO_ROOT).free)


def _format_gib(num_bytes: int) -> str:
    return f"{num_bytes / (1024 ** 3):.2f} GiB"


def _checkpoint_listing(ckpt_dir: Path) -> List[str]:
    if not ckpt_dir.exists():
        return []
    return sorted(p.name for p in ckpt_dir.iterdir())


def _candidate_count(ckpt_dir: Path) -> int:
    return len(list(ckpt_dir.glob("candidate_epoch_*.pt")))


def _parse_train_log(train_log: Path) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "exists": train_log.exists(),
        "latest_epoch": None,
        "latest_update": None,
        "latest_train_loss_total": None,
        "latest_loss_rot": None,
        "latest_loss_tdir": None,
        "latest_loss_tmag": None,
        "latest_grad_norm": None,
        "nonfinite_count_recent50": 0,
        "train_nan_inf_count": None,
        "mini_val_score": None,
        "full_val_score": None,
        "recent_loss_finite": None,
        "recent_grad_finite": None,
        "nonfinite_detected": False,
        "recent_rows": 0,
    }
    if not train_log.exists():
        return payload
    rows = []
    for line in train_log.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    recent = rows[-50:]
    payload["recent_rows"] = len(recent)
    if not recent:
        return payload
    last = recent[-1]
    payload["latest_epoch"] = last.get("epoch")
    payload["latest_train_loss_total"] = last.get("train_loss_total")
    payload["latest_loss_rot"] = last.get("train_loss_rot")
    payload["latest_loss_tdir"] = last.get("train_loss_tdir")
    payload["latest_loss_tmag"] = last.get("train_loss_tmag")
    payload["latest_grad_norm"] = last.get("grad_norm")
    payload["train_nan_inf_count"] = last.get("train_nan_inf_count")
    payload["mini_val_score"] = last.get("mini_val_score")
    payload["full_val_score"] = last.get("full_val_score")

    loss_finite = True
    grad_finite: Optional[bool] = True
    saw_grad = False
    nonfinite_count = 0
    for row in recent:
        for key in ["train_loss_total", "train_loss_rot", "train_loss_tdir", "train_loss_tmag"]:
            value = row.get(key)
            if isinstance(value, float) and not math.isfinite(value):
                nonfinite_count += 1
                loss_finite = False
        grad = row.get("grad_norm")
        if grad is not None:
            saw_grad = True
            if isinstance(grad, float) and not math.isfinite(grad):
                nonfinite_count += 1
                grad_finite = False
        if (row.get("train_nan_inf_count") or 0) > 0:
            nonfinite_count += 1
        mv = row.get("mini_val_score")
        if isinstance(mv, float) and not math.isfinite(mv):
            nonfinite_count += 1
        fv = row.get("full_val_score")
        if isinstance(fv, float) and not math.isfinite(fv):
            nonfinite_count += 1
    payload["recent_loss_finite"] = loss_finite
    payload["recent_grad_finite"] = grad_finite if saw_grad else None
    payload["nonfinite_count_recent50"] = nonfinite_count
    payload["nonfinite_detected"] = (
        (payload["recent_loss_finite"] is False)
        or (payload["recent_grad_finite"] is False)
        or ((payload["train_nan_inf_count"] or 0) > 0)
    )
    return payload


def _artifact_status(ckpt_dir: Path) -> Dict[str, Any]:
    return {
        "last_pt": (ckpt_dir / "last.pt").exists(),
        "best_full_val_pt": (ckpt_dir / "best_full_val.pt").exists(),
        "candidate_epoch_pts": sorted(p.name for p in ckpt_dir.glob("candidate_epoch_*.pt")),
        "metrics_val_exists": (REPO_ROOT / "reports" / "FINAL360M_metrics_val.json").exists(),
        "metrics_test_exists": (REPO_ROOT / "reports" / "FINAL360M_metrics_test.json").exists(),
        "model_selection_exists": (REPO_ROOT / "reports" / "FINAL360M_model_selection_table.json").exists(),
        "guarded_metrics_val_exists": (REPO_ROOT / "reports" / "FINAL360M_fulltrain_guarded_metrics_val.json").exists(),
        "guarded_metrics_test_exists": (REPO_ROOT / "reports" / "FINAL360M_fulltrain_guarded_metrics_test.json").exists(),
        "guarded_model_selection_exists": (REPO_ROOT / "reports" / "FINAL360M_fulltrain_guarded_model_selection_table.json").exists(),
    }


def _stop_training(pid: int) -> str:
    alive = _is_process_alive(pid)["alive"]
    if not alive:
        return "already_dead"
    os.kill(pid, signal.SIGTERM)
    time.sleep(10)
    alive = _is_process_alive(pid)["alive"]
    if alive:
        os.kill(pid, signal.SIGKILL)
        time.sleep(2)
        return "KILL"
    return "TERM"


def _status_md(payload: Dict[str, Any]) -> List[str]:
    return [
        "# FINAL360M overnight watchdog status",
        "",
        f"- timestamp: `{payload['timestamp']}`",
        f"- watchdog started: `true`",
        f"- watchdog pid: `{payload['watchdog_pid']}`",
        f"- monitored training pid: `{payload['monitored_training_pid']}`",
        f"- monitoring interval sec: `{payload['monitoring_interval_sec']}`",
        f"- process alive: `{str(payload['process_alive']).lower()}`",
        f"- elapsed_time: `{payload.get('elapsed_time')}`",
        f"- latest epoch: `{payload.get('latest_epoch')}`",
        f"- latest update: `{payload.get('latest_update')}`",
        f"- recent loss finite: `{payload.get('recent_loss_finite')}`",
        f"- recent grad finite: `{payload.get('recent_grad_finite')}`",
        f"- train_nan_inf_count: `{payload.get('train_nan_inf_count')}`",
        f"- disk free: `{payload.get('disk_free_human')}`",
        f"- candidate checkpoint count: `{payload.get('candidate_checkpoint_count')}`",
        f"- best_full_val_exists: `{str(payload.get('best_full_val_exists')).lower()}`",
        f"- metrics_val_exists: `{str(payload.get('metrics_val_exists')).lower()}`",
        f"- metrics_test_exists: `{str(payload.get('metrics_test_exists')).lower()}`",
        f"- recommended_action: `{payload.get('recommended_action')}`",
    ]


def _critical_report(path_stem: str, payload: Dict[str, Any], title: str) -> None:
    json_path = REPO_ROOT / "reports" / f"{path_stem}.json"
    md_path = REPO_ROOT / "reports" / f"{path_stem}.md"
    _json_dump(json_path, payload)
    lines = [
        f"# {title}",
        "",
        f"- timestamp: `{payload['timestamp']}`",
        f"- monitored training pid: `{payload['monitored_training_pid']}`",
        f"- process alive before stop: `{str(payload['process_alive_before_stop']).lower()}`",
        f"- critical stop executed: `{str(payload['critical_stop_executed']).lower()}`",
        f"- stop reason: `{payload['stop_reason']}`",
        f"- stop method: `{payload['stop_method']}`",
        f"- disk free: `{payload['disk_free_human']}`",
        f"- latest epoch: `{payload.get('latest_epoch')}`",
        f"- recent loss finite: `{payload.get('recent_loss_finite')}`",
        f"- recent grad finite: `{payload.get('recent_grad_finite')}`",
        f"- train_nan_inf_count: `{payload.get('train_nan_inf_count')}`",
    ]
    _write_md(md_path, lines)


def _final_report(payload: Dict[str, Any]) -> None:
    _json_dump(REPO_ROOT / "reports" / "FINAL360M_overnight_watchdog_final.json", payload)
    lines = [
        "# FINAL360M overnight watchdog final",
        "",
        f"- watchdog started: `{str(payload['watchdog_started']).lower()}`",
        f"- watchdog pid: `{payload['watchdog_pid']}`",
        f"- monitored training pid: `{payload['monitored_training_pid']}`",
        f"- monitoring interval sec: `{payload['monitoring_interval_sec']}`",
        f"- process alive at start: `{str(payload['process_alive_at_start']).lower()}`",
        f"- process alive at end: `{str(payload['process_alive_at_end']).lower()}`",
        f"- latest epoch: `{payload.get('latest_epoch')}`",
        f"- latest update: `{payload.get('latest_update')}`",
        f"- recent loss finite: `{payload.get('recent_loss_finite')}`",
        f"- recent grad finite: `{payload.get('recent_grad_finite')}`",
        f"- nonfinite detected: `{str(payload.get('nonfinite_detected')).lower()}`",
        f"- disk free min: `{payload.get('disk_free_min_human')}`",
        f"- low disk warning: `{str(payload.get('low_disk_warning')).lower()}`",
        f"- critical stop executed: `{str(payload.get('critical_stop_executed')).lower()}`",
        f"- stop reason: `{payload.get('stop_reason')}`",
        f"- best_full_val.pt exists: `{str(payload.get('best_full_val_exists')).lower()}`",
        f"- metrics_val generated: `{str(payload.get('metrics_val_generated')).lower()}`",
        f"- metrics_test generated: `{str(payload.get('metrics_test_generated')).lower()}`",
        f"- reports written: `true`",
        f"- metrics modified: `false`",
        f"- checkpoints deleted: `false`",
        f"- recommended next action: `{payload.get('recommended_next_action')}`",
    ]
    _write_md(REPO_ROOT / "reports" / "FINAL360M_overnight_watchdog_final.md", lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--interval-sec", type=int, default=900)
    parser.add_argument("--max-hours", type=float, default=8.0)
    args = parser.parse_args()

    status_json = REPO_ROOT / "reports" / "FINAL360M_overnight_watchdog_status.json"
    status_md = REPO_ROOT / "reports" / "FINAL360M_overnight_watchdog_status.md"

    start_time = time.time()
    process_alive_at_start = _is_process_alive(args.pid)["alive"]
    disk_free_min = _disk_free_bytes()
    low_disk_warning = False
    critical_stop_executed = False
    stop_reason = None
    stop_method = None
    final_payload: Optional[Dict[str, Any]] = None

    while True:
        proc = _is_process_alive(args.pid)
        tail = _read_tail(args.log, lines=120)
        disk_free = _disk_free_bytes()
        disk_free_min = min(disk_free_min, disk_free)
        train_log = args.ckpt_dir / "train_log.jsonl"
        train_state = _parse_train_log(train_log)
        artifacts = _artifact_status(args.ckpt_dir)
        candidate_count = len(artifacts["candidate_epoch_pts"])

        recommended_action = "continue_waiting"
        if disk_free < 5 * 1024 ** 3:
            low_disk_warning = True
            recommended_action = "monitor_disk"
        if train_state["nonfinite_detected"]:
            recommended_action = "stop_due_to_nan"
        elif not proc["alive"] and artifacts["best_full_val_pt"]:
            recommended_action = "final_evaluate"
        elif not proc["alive"]:
            recommended_action = "resume_required"

        status_payload = {
            "timestamp": _now_ts(),
            "watchdog_started": True,
            "watchdog_pid": os.getpid(),
            "monitored_training_pid": args.pid,
            "monitoring_interval_sec": args.interval_sec,
            "process_alive": proc["alive"],
            "elapsed_time": proc.get("elapsed"),
            "latest_epoch": train_state["latest_epoch"],
            "latest_update": train_state["latest_update"],
            "recent_loss_finite": train_state["recent_loss_finite"],
            "recent_grad_finite": train_state["recent_grad_finite"],
            "train_nan_inf_count": train_state["train_nan_inf_count"],
            "disk_free_bytes": disk_free,
            "disk_free_human": _format_gib(disk_free),
            "candidate_checkpoint_count": candidate_count,
            "best_full_val_exists": artifacts["best_full_val_pt"],
            "metrics_val_exists": artifacts["guarded_metrics_val_exists"] or artifacts["metrics_val_exists"],
            "metrics_test_exists": artifacts["guarded_metrics_test_exists"] or artifacts["metrics_test_exists"],
            "recommended_action": recommended_action,
            "ps_line": proc["ps_line"],
            "tail_excerpt": tail,
            "checkpoint_dir_contents": _checkpoint_listing(args.ckpt_dir),
            "full_val_score": train_state["full_val_score"],
            "mini_val_score": train_state["mini_val_score"],
            "nonfinite_detected": train_state["nonfinite_detected"],
            "low_disk_warning": low_disk_warning,
        }
        _json_dump(status_json, status_payload)
        _write_md(status_md, _status_md(status_payload))

        if train_state["nonfinite_detected"]:
            stop_reason = "nonfinite_detected"
            stop_method = _stop_training(args.pid)
            critical_stop_executed = stop_method != "already_dead"
            critical_payload = dict(status_payload)
            critical_payload.update(
                {
                    "process_alive_before_stop": proc["alive"],
                    "critical_stop_executed": critical_stop_executed,
                    "stop_reason": stop_reason,
                    "stop_method": stop_method,
                }
            )
            _critical_report("FINAL360M_watchdog_CRITICAL_nonfinite", critical_payload, "FINAL360M watchdog critical nonfinite")
            final_payload = critical_payload
            break

        if disk_free < 2 * 1024 ** 3:
            stop_reason = "critical_low_disk"
            stop_method = _stop_training(args.pid)
            critical_stop_executed = stop_method != "already_dead"
            critical_payload = dict(status_payload)
            critical_payload.update(
                {
                    "process_alive_before_stop": proc["alive"],
                    "critical_stop_executed": critical_stop_executed,
                    "stop_reason": stop_reason,
                    "stop_method": stop_method,
                }
            )
            _critical_report("FINAL360M_watchdog_CRITICAL_low_disk", critical_payload, "FINAL360M watchdog critical low disk")
            final_payload = critical_payload
            break

        if not proc["alive"]:
            final_payload = dict(status_payload)
            break

        if time.time() - start_time >= args.max_hours * 3600.0:
            final_payload = dict(status_payload)
            break

        time.sleep(args.interval_sec)

    if final_payload is None:
        final_payload = {}
    final_summary = {
        "watchdog_started": True,
        "watchdog_pid": os.getpid(),
        "monitored_training_pid": args.pid,
        "monitoring_interval_sec": args.interval_sec,
        "process_alive_at_start": process_alive_at_start,
        "process_alive_at_end": bool(_is_process_alive(args.pid)["alive"]),
        "latest_epoch": final_payload.get("latest_epoch"),
        "latest_update": final_payload.get("latest_update"),
        "recent_loss_finite": final_payload.get("recent_loss_finite"),
        "recent_grad_finite": final_payload.get("recent_grad_finite"),
        "nonfinite_detected": bool(final_payload.get("nonfinite_detected", False)),
        "disk_free_min_bytes": disk_free_min,
        "disk_free_min_human": _format_gib(disk_free_min),
        "low_disk_warning": low_disk_warning,
        "critical_stop_executed": critical_stop_executed,
        "stop_reason": stop_reason,
        "stop_method": stop_method,
        "best_full_val_exists": bool(_artifact_status(args.ckpt_dir)["best_full_val_pt"]),
        "metrics_val_generated": bool(_artifact_status(args.ckpt_dir)["guarded_metrics_val_exists"] or _artifact_status(args.ckpt_dir)["metrics_val_exists"]),
        "metrics_test_generated": bool(_artifact_status(args.ckpt_dir)["guarded_metrics_test_exists"] or _artifact_status(args.ckpt_dir)["metrics_test_exists"]),
        "recommended_next_action": final_payload.get("recommended_action"),
    }
    _final_report(final_summary)


if __name__ == "__main__":
    main()
