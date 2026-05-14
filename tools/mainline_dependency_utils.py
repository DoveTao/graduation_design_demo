from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


def path_status(path: Path) -> Dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "is_dir": path.is_dir(),
    }


def optional_read_json(path: Path, default: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    if not path.exists():
        if default is not None:
            return dict(default)
        return {"status": "missing_after_cleanup", "path": str(path)}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "read_error", "path": str(path), "error": str(exc)}
    if isinstance(payload, dict):
        return payload
    return {"status": "non_mapping_json", "path": str(path), "payload_type": type(payload).__name__}


def optional_read_text(path: Path, default: str = "") -> str:
    if not path.exists():
        return default
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return default


def summarize_optional_artifact(path: Path, label: str) -> Dict[str, Any]:
    status = "present" if path.exists() else "missing_after_cleanup"
    return {
        "label": label,
        "path": str(path),
        "status": status,
    }
