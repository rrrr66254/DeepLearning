"""Reusable Qwen visual question answering module."""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from utils.device import pick_device, preferred_dtype
from utils.image_io import load_rgb_image, resolve_image_path


DEFAULT_QWEN_MODEL = "Qwen/Qwen3-VL-2B-Instruct"


def run_qwen(image_path: str | Path, question: str, model_id: str = DEFAULT_QWEN_MODEL) -> dict:
    """Run visual question answering with a Qwen VL model.

    This function is intentionally reusable so midterm and final scripts
    can call the same internal logic.
    """
    device = pick_device()
    dtype = preferred_dtype(device)
    image_file = resolve_image_path(image_path)
    image = load_rgb_image(image_file)

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=128)

    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return {
        "model": model_id,
        "task": "visual_question_answering",
        "image": str(image_file),
        "question": question,
        "answer": response,
        "device": str(device),
    }
