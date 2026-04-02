"""Helpers for saving JSON and image outputs with consistent paths."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist and return it."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(payload: dict[str, Any], output_path: str | Path) -> Path:
    """Save a dictionary as pretty-printed JSON."""
    out = Path(output_path)
    ensure_dir(out.parent)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out
