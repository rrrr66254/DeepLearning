"""Centralized image loading helpers."""

from __future__ import annotations

from pathlib import Path
from PIL import Image


def resolve_image_path(image_path: str | Path) -> Path:
    """Resolve and validate an image path in an OS-agnostic way."""
    path = Path(image_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    if not path.is_file():
        raise ValueError(f"Image path is not a file: {path}")
    return path


def load_rgb_image(image_path: str | Path) -> Image.Image:
    """Load an image and normalize to RGB for model compatibility."""
    path = resolve_image_path(image_path)
    with Image.open(path) as img:
        return img.convert("RGB")
