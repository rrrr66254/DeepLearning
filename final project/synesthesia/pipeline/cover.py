"""Stage 3 - Stable Diffusion: paint an album cover for the song."""
from __future__ import annotations

import os

import config
from pipeline.manager import free


def make_cover(cover_prompt: str, out_path: str, steps: int | None = None) -> str:
    from diffusers import AutoPipelineForText2Image

    steps = steps or config.SD_STEPS
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    prompt = (cover_prompt or "abstract album cover art, vivid colors") + \
        ", album cover, highly detailed, artistic"

    try:
        pipe = AutoPipelineForText2Image.from_pretrained(
            config.SD_MODEL, torch_dtype=config.DTYPE, variant="fp16"
        )
    except Exception:
        pipe = AutoPipelineForText2Image.from_pretrained(
            config.SD_MODEL, torch_dtype=config.DTYPE
        )
    pipe = pipe.to(config.DEVICE)

    image = pipe(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=config.SD_GUIDANCE,
        height=config.SD_SIZE,
        width=config.SD_SIZE,
    ).images[0]
    image.save(out_path)

    if config.FREE_AFTER_STAGE:
        pipe = pipe.to("cpu")
        free(pipe)

    return out_path
