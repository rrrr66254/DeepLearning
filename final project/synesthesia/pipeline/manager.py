"""Tiny helpers for moving models on/off the GPU between pipeline stages."""
from __future__ import annotations

import gc

import torch


def free(*objs) -> None:
    """Delete references and reclaim VRAM.

    Pass any model/processor objects; the caller should also drop its own
    references (e.g. set the variable to None) right after.
    """
    for obj in objs:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def vram_summary() -> str:
    if not torch.cuda.is_available():
        return "cpu (no CUDA)"
    used = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return f"VRAM allocated={used:.2f} GiB reserved={reserved:.2f} GiB / {total:.1f} GiB"
