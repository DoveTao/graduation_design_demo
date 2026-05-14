#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
S6_NAME = "s6_final_clean_candidate_lockdown_audit.py"
WATCH_TERMS = (S6_NAME, "verify_final_candidate", "project_health_check")
REPORT_PATH = "reports/s5d11_validation_concurrency_audit.md"
CHECKPOINT_PATH = "checkpoints/S5D11_validation_concurrency_audit.json"


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _run(cmd: List[str]) -> Dict[str, Any]:
    try:
        cp = subprocess.run(cmd, text=True, capture_output=True, check=False)
        return {
            "available": cp.returncode == 0,
            "returncode": cp.returncode,
            "stdout": cp.stdout,
            "stderr": cp.stderr,
        }
    except FileNotFoundError as exc:
        return {"available": False, "returncode": None, "stdout": "", "stderr": str(exc)}


def _list_processes() -> List[Dict[str, Any]]:
    ps = _run(["ps", "-eo", "pid=,ppid=,stat=,command="])
    rows: List[Dict[str, Any]] = []
    for line in ps.get("stdout", "").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        pid, ppid, stat, cmd = parts
        if "audit_s5d11_validation_concurrency.py" in cmd:
            continue
        if any(term in cmd for term in WATCH_TERMS):
            rows.append({
                "pid": int(pid),
                "ppid": int(ppid),
                "stat": stat,
                "command": cmd,
                "is_s6_eval_only": (S6_NAME in cmd and "--eval-only" in cmd),
            })
    return rows


def _gpu_process_memory() -> Dict[int, Dict[str, Any]]:
    query = _run([
        "nvidia-smi",
        "--query-compute-apps=pid,used_memory,process_name",
        "--format=csv,noheader,nounits",
    ])
    out: Dict[int, Dict[str, Any]] = {}
    for line in query.get("stdout", "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
            used = int(parts[1])
        except ValueError:
            continue
        out[pid] = {"used_memory_mib": used, "process_name": ",".join(parts[2:]).strip()}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", default=CHECKPOINT_PATH)
    ap.add_argument("--out-report", default=REPORT_PATH)
    args = ap.parse_args()

    out_json = _resolve(args.out_json)
    out_report = _resolve(args.out_report)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / "logs").mkdir(exist_ok=True)

    smi = _run(["nvidia-smi"])
    smi_text = smi.get("stdout", "") + (("\n[stderr]\n" + smi.get("stderr", "")) if smi.get("stderr") else "")
    (REPO_ROOT / "logs/s5d11_nvidia_smi_before.log").write_text(smi_text, encoding="utf-8")

    gpu_by_pid = _gpu_process_memory()
    processes = _list_processes()
    for proc in processes:
        proc["gpu_memory_mib"] = gpu_by_pid.get(proc["pid"], {}).get("used_memory_mib")
        proc["gpu_process_name"] = gpu_by_pid.get(proc["pid"], {}).get("process_name")

    s6_eval = [p for p in processes if p["is_s6_eval_only"]]
    concurrent_s6 = len(s6_eval) > 1
    total_related_gpu = sum(int(p.get("gpu_memory_mib") or 0) for p in processes)
    risk = "CUDA_OOM_RISK_CONCURRENT_EVAL" if concurrent_s6 else "NO_CONCURRENT_S6_EVAL_ONLY_DETECTED"

    payload: Dict[str, Any] = {
        "experiment": "S5D11_validation_concurrency_audit",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checks": {
            "s6_eval_only_process_count": len(s6_eval),
            "concurrent_s6_processes_found": concurrent_s6,
            "concurrency_risk": risk,
            "related_gpu_memory_mib_total": total_related_gpu,
            "nvidia_smi_available": bool(smi.get("available")),
            "nvidia_smi_log_path": "logs/s5d11_nvidia_smi_before.log",
        },
        "related_processes": processes,
        "gpu_process_memory_by_pid": gpu_by_pid,
        "final_status": risk,
    }
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# S5D11 Validation Concurrency Audit",
        "",
        "## Summary",
        f"- concurrent s6 eval-only processes found: `{concurrent_s6}`",
        f"- s6 eval-only process count: `{len(s6_eval)}`",
        f"- concurrency risk: `{risk}`",
        f"- related GPU memory total: `{total_related_gpu} MiB`",
        f"- nvidia-smi log: `logs/s5d11_nvidia_smi_before.log`",
        "",
        "## Related Processes",
    ]
    if processes:
        for p in processes:
            lines.append(
                f"- pid={p['pid']} gpu_memory_mib={p.get('gpu_memory_mib')} "
                f"is_s6_eval_only={p['is_s6_eval_only']} cmd=`{p['command']}`"
            )
    else:
        lines.append("- none")
    out_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
