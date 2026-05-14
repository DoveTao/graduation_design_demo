#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def run(args: argparse.Namespace) -> Dict[str, Any]:
    s5e15 = _read_jsonl(Path(args.s5e15_results) / "edge_provenance.jsonl")
    s5e19c = _read_jsonl(Path(args.s5e19c_results) / "edge_provenance.jsonl")
    learned_path = Path(args.candidate) / "s5e19c_model.pt"
    delta_norms = [float(r.get("delta_tdir_norm", 0.0)) for r in s5e19c]
    sign_scores = [float(r.get("sign_score", 0.0)) for r in s5e19c]
    final_equals_s5e15 = 0
    for a, b in zip(s5e15, s5e19c):
        if b.get("final_tdir") == a.get("translation_direction"):
            final_equals_s5e15 += 1
    eval_text = Path("tools/evaluate_s5e19c_traceable_dense.py").read_text(encoding="utf-8")
    hardcoded_reuse = any(
        token in eval_text
        for token in [
            '"component_metrics": S5E15_REF',
            '"external_eval": S5E15_REF',
            "return S5E15_REF",
            "= S5E15_REF.copy()",
        ]
    )
    payload = {
        "delta_tdir_zero_count": sum(1 for x in delta_norms if x <= 1.0e-12),
        "delta_tdir_norm_mean": sum(delta_norms) / len(delta_norms) if delta_norms else None,
        "sign_score_unique_count": len({round(x, 12) for x in sign_scores}),
        "final_equals_s5e15_direction_count": final_equals_s5e15,
        "fallback_to_s5e15_direction": any(bool(r.get("uses_s5e15_direction_fallback", False)) for r in s5e19c),
        "learned_weights_loaded": learned_path.exists(),
        "export_used_learned_weights": learned_path.exists(),
        "evaluator_hardcoded_s5e15_metrics": hardcoded_reuse,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--s5e15-results", required=True)
    p.add_argument("--s5e19c-results", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--out-json", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
