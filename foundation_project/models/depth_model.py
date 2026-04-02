"""Reusable Depth Anything V2 module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from transformers import pipeline

from utils.device import pick_device
from utils.image_io import load_rgb_image, resolve_image_path
from utils.save_utils import ensure_dir


DEFAULT_DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


def _normalize_depth(depth_array: np.ndarray) -> np.ndarray:
    min_v = float(depth_array.min())
    max_v = float(depth_array.max())
    if max_v - min_v < 1e-8:
        return np.zeros_like(depth_array, dtype=np.uint8)
    normalized = (depth_array - min_v) / (max_v - min_v)
    return (normalized * 255).astype(np.uint8)


def run_depth(
    image_path: str | Path,
    model_id: str = DEFAULT_DEPTH_MODEL,
    save_depth_to: str | Path | None = None,
) -> dict:
    """Estimate depth for an input image and optionally save a depth map image."""
    device = pick_device()
    image_file = resolve_image_path(image_path)
    image = load_rgb_image(image_file)

    estimator = pipeline(
        task="depth-estimation",
        model=model_id,
        device=0 if device.type == "cuda" else -1,
    )

    output = estimator(image)
    predicted_depth = np.array(output["predicted_depth"])  # HxW float depth
    depth_image = Image.fromarray(_normalize_depth(predicted_depth), mode="L")

    depth_output = None
    if save_depth_to is not None:
        out_path = Path(save_depth_to)
        ensure_dir(out_path.parent)
        depth_image.save(out_path)
        depth_output = str(out_path.resolve())

    return {
        "model": model_id,
        "task": "depth_estimation",
        "image": str(image_file),
        "depth_map": depth_output,
        "shape": list(predicted_depth.shape),
        "device": str(device),
    }
