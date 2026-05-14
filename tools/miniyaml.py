from __future__ import annotations

from pathlib import Path
from typing import Any, List, Tuple


def _parse_scalar(text: str) -> Any:
    s = text.strip()
    if s in {"true", "True"}:
        return True
    if s in {"false", "False"}:
        return False
    if s in {"null", "None", "~"}:
        return None
    if s.startswith("[") and s.endswith("]"):
        body = s[1:-1].strip()
        if not body:
            return []
        return [_parse_scalar(part.strip()) for part in body.split(",")]
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    try:
        if "." in s:
            return float(s)
        return int(s)
    except Exception:
        return s


def _next_nonempty(lines: List[str], idx: int) -> Tuple[int, str]:
    j = idx + 1
    while j < len(lines):
        raw = lines[j]
        if raw.strip() and not raw.lstrip().startswith("#"):
            return j, raw
        j += 1
    return len(lines), ""


def _parse_block(lines: List[str], start: int, indent: int) -> Tuple[Any, int]:
    result: Any = {}
    i = start
    while i < len(lines):
        raw = lines[i]
        if not raw.strip() or raw.lstrip().startswith("#"):
            i += 1
            continue
        current_indent = len(raw) - len(raw.lstrip(" "))
        if current_indent < indent:
            break
        if current_indent > indent:
            raise ValueError(f"Unexpected indentation near line: {raw}")
        stripped = raw.strip()
        if stripped.startswith("- "):
            items: List[Any] = []
            while i < len(lines):
                raw = lines[i]
                if not raw.strip() or raw.lstrip().startswith("#"):
                    i += 1
                    continue
                current_indent = len(raw) - len(raw.lstrip(" "))
                if current_indent < indent:
                    break
                if current_indent != indent or not raw.strip().startswith("- "):
                    break
                item_text = raw.strip()[2:].strip()
                if not item_text:
                    j, next_raw = _next_nonempty(lines, i)
                    if j >= len(lines):
                        items.append(None)
                        i = j
                        continue
                    next_indent = len(next_raw) - len(next_raw.lstrip(" "))
                    value, ni = _parse_block(lines, i + 1, next_indent)
                    items.append(value)
                    i = ni
                    continue
                if ":" in item_text and not item_text.startswith(("http://", "https://")):
                    key, rest = item_text.split(":", 1)
                    key = key.strip()
                    rest = rest.strip()
                    if rest:
                        items.append({key: _parse_scalar(rest)})
                        i += 1
                    else:
                        j, next_raw = _next_nonempty(lines, i)
                        next_indent = len(next_raw) - len(next_raw.lstrip(" ")) if j < len(lines) else indent + 2
                        value, ni = _parse_block(lines, i + 1, next_indent)
                        items.append({key: value})
                        i = ni
                else:
                    items.append(_parse_scalar(item_text))
                    i += 1
            return items, i
        if ":" not in stripped:
            raise ValueError(f"Expected key:value near line: {raw}")
        key, rest = stripped.split(":", 1)
        key = key.strip()
        rest = rest.strip()
        if rest:
            result[key] = _parse_scalar(rest)
            i += 1
            continue
        j, next_raw = _next_nonempty(lines, i)
        if j >= len(lines):
            result[key] = {}
            i = j
            continue
        next_indent = len(next_raw) - len(next_raw.lstrip(" "))
        if next_indent <= indent:
            result[key] = {}
            i += 1
            continue
        value, ni = _parse_block(lines, i + 1, next_indent)
        result[key] = value
        i = ni
    return result, i


def load_yaml_like(path: Path) -> Any:
    lines = path.read_text(encoding="utf-8").splitlines()
    parsed, _ = _parse_block(lines, 0, 0)
    return parsed
