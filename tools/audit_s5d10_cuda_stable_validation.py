#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parent.parent


def _resolve(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else REPO_ROOT / q


def _pass(path: Path) -> bool:
    if not path.exists():
        return False
    s = path.read_text(encoding='utf-8', errors='ignore').lower()
    if 'traceback' in s or 'failed' in s or 'error' in s:
        return False
    return ('pass' in s) or ('ok' in s) or ('status: pass' in s)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--s5d9-s6-log', default='logs/s5d9_s6_lockdown_eval_only.log')
    ap.add_argument('--out-json', required=True)
    ap.add_argument('--out-report', required=True)
    args = ap.parse_args()

    s5d9 = _resolve(args.s5d9_s6_log)
    out_json = _resolve(args.out_json)
    out_report = _resolve(args.out_report)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)

    s = s5d9.read_text(encoding='utf-8', errors='ignore') if s5d9.exists() else ''
    prev_oom = ('outofmemoryerror' in s.lower() or 'cuda out of memory' in s.lower())

    # best-effort nvidia-smi snapshot
    smi = ''
    try:
        smi = subprocess.check_output(['bash', '-lc', 'nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader'], cwd=REPO_ROOT, text=True)
    except Exception:
        smi = 'unavailable'

    verify_log = REPO_ROOT / 'logs/s5d10_verify_final_candidate.log'
    health_log = REPO_ROOT / 'logs/s5d10_project_health_check.log'
    s6_log = REPO_ROOT / 'logs/s5d10_s6_eval_only.log'
    unit_log = REPO_ROOT / 'logs/s5d10_unittest.log'

    verify = _pass(verify_log)
    health = _pass(health_log)
    s6 = _pass(s6_log)
    us = unit_log.read_text(encoding='utf-8', errors='ignore') if unit_log.exists() else ''
    unittest = ('OK' in us and 'FAILED' not in us)
    m = re.search(r'Ran\s+(\d+)\s+tests', us)
    tcount = int(m.group(1)) if m else None

    cur_oom = ('outofmemoryerror' in (s6_log.read_text(encoding='utf-8', errors='ignore').lower()) if s6_log.exists() else False)
    if verify and health:
        status = 'resolved'
    elif cur_oom:
        status = 'still_present'
    else:
        status = 'intermittent'

    payload: Dict[str, object] = {
        'previous_s5d9_oom': prev_oom,
        's5d9_log': str(s5d9.relative_to(REPO_ROOT)) if s5d9.exists() else str(s5d9),
        'nvidia_smi_snapshot': smi.strip().splitlines() if smi else [],
        'validation': {
            'verify_final_candidate': {'passed': verify, 'log_path': 'logs/s5d10_verify_final_candidate.log'},
            'project_health_check': {'passed': health, 'log_path': 'logs/s5d10_project_health_check.log'},
            's6_eval_only': {'passed': s6, 'log_path': 'logs/s5d10_s6_eval_only.log'},
            'unittest': {'passed': unittest, 'test_count': tcount, 'log_path': 'logs/s5d10_unittest.log'},
            'validation_clean': bool(verify and health and s6 and unittest),
            'cuda_oom_status': status,
        },
    }
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    out_report.write_text('# S5D10 CUDA stable validation audit\n\n' + json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
