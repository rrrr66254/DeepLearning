"""Device selection helper for cross-platform inference."""

from __future__ import annotations

import torch


def pick_device() -> torch.device:
    """Return the best available compute device.

    Priority:
    1. CUDA (typical on Linux/Windows with NVIDIA GPU)
    2. MPS (Apple Silicon on macOS)
    3. CPU fallback
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    # MPS exists only in newer PyTorch builds on macOS.
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def preferred_dtype(device: torch.device) -> torch.dtype:
    """Pick a safe default dtype based on the target device."""
    if device.type == "cuda":
        return torch.float16
    if device.type == "mps":
        # bfloat16 can be unstable on some macOS/PyTorch combinations.
        return torch.float16
    return torch.float32
