"""Reusable Florence-2 object detection/grounding module."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import ImageDraw
from transformers import AutoModelForCausalLM, AutoProcessor

from utils.device import pick_device, preferred_dtype
from utils.image_io import load_rgb_image, resolve_image_path
from utils.save_utils import ensure_dir


DEFAULT_FLORENCE_MODEL = "microsoft/Florence-2-base-ft"


def run_florence(
    image_path: str | Path,
    model_id: str = DEFAULT_FLORENCE_MODEL,
    task_prompt: str = "<OD>",
    save_annotated_to: str | Path | None = None,
) -> dict:
    """Run Florence-2 object detection task and optionally save an annotated image."""
    device = pick_device()
    dtype = preferred_dtype(device)
    image_file = resolve_image_path(image_path)
    image = load_rgb_image(image_file)

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    inputs = processor(text=task_prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.inference_mode():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs.get("pixel_values"),
            max_new_tokens=512,
            num_beams=3,
        )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=image.size,
    )

    objects = parsed.get(task_prompt, {})
    result = {
        "model": model_id,
        "task": "object_detection",
        "image": str(image_file),
        "task_prompt": task_prompt,
        "objects": objects,
        "device": str(device),
        "output_image": None,
    }

    if save_annotated_to is not None:
        out_path = Path(save_annotated_to)
        ensure_dir(out_path.parent)
        draw = ImageDraw.Draw(image)

        bboxes = objects.get("bboxes", []) if isinstance(objects, dict) else []
        labels = objects.get("labels", []) if isinstance(objects, dict) else []

        for bbox, label in zip(bboxes, labels):
            x1, y1, x2, y2 = bbox
            draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
            draw.text((x1, max(0, y1 - 15)), str(label), fill="red")

        image.save(out_path)
        result["output_image"] = str(out_path.resolve())

    return result
