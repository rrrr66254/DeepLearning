"""Stage 1 - Qwen2.5-VL: look at the image and write the song.

A single vision-language model both *sees* the picture (mood, scene, palette)
and *writes* a structured song specification plus lyrics. Output is JSON so the
downstream music model gets a clean text prompt.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import torch
from PIL import Image

import config
from pipeline.manager import free

_SYSTEM = (
    "You are a synesthetic songwriter. You look at an image and translate what "
    "you see and feel into a song. Pay attention to mood, colors, light, motion "
    "and atmosphere - not just the literal objects."
)

_INSTRUCTION = (
    "Look at this image and compose a song inspired by it.\n"
    "Respond with ONLY a JSON object, no extra text, with these keys:\n"
    '  "title": a short song title,\n'
    '  "mood": 3-5 emotion words describing the feeling,\n'
    '  "genre": one music genre that fits,\n'
    '  "tempo_bpm": an integer beats-per-minute,\n'
    '  "instruments": a short comma-separated list of instruments,\n'
    '  "tags": comma-separated style tags for a music model, including genre, '
    "mood, the main instruments, vocal type (e.g. soft female vocals) and tempo,\n"
    '  "lyrics": song lyrics with structure tags on their own lines, like '
    "[verse] and [chorus]. Write two verses and a chorus, 12 to 16 lines total, "
    "separated by \\n. Make them singable and evocative of the image,\n"
    '  "cover_prompt": one vivid English sentence describing album cover art for '
    "this song (subject, mood, colors, art style) for an image-generation model.\n"
)


def prompt_text() -> str:
    """The exact prompt sent to Qwen2.5-VL (shown in the UI)."""
    return f"[system]\n{_SYSTEM}\n\n[user]\n{_INSTRUCTION}"


@dataclass
class Song:
    title: str = "Untitled"
    mood: str = ""
    genre: str = ""
    tempo_bpm: int = 90
    instruments: str = ""
    tags: str = ""
    cover_prompt: str = ""
    lyrics: str = ""
    raw: str = field(default="", repr=False)


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model response."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    blob = match.group(0) if match else text
    for candidate in (blob, re.sub(r",\s*([}\]])", r"\1", blob)):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    # The model sometimes truncates the JSON (long lyrics) or emits raw newlines
    # inside strings, both invalid JSON. Salvage fields by regex instead.
    return _salvage(text)


def _salvage(text: str) -> dict:
    data: dict = {}
    for key in ("title", "mood", "genre", "instruments", "tags", "cover_prompt"):
        m = re.search(rf'"{key}"\s*:\s*"([^"]*)"', text)
        if m:
            data[key] = m.group(1)
            continue
        arr = re.search(rf'"{key}"\s*:\s*\[(.*?)(?:\]|$)', text, re.DOTALL)
        if arr:
            data[key] = re.findall(r'"([^"]*)"', arr.group(1))
    tempo = re.search(r'"tempo_bpm"\s*:\s*(\d+)', text)
    if tempo:
        data["tempo_bpm"] = int(tempo.group(1))

    # Lyrics may be an array whose lines contain "]" (e.g. "[Verse 1]"), so grab
    # everything after `"lyrics": [` up to the next field (cover_prompt) or the
    # end of the (possibly truncated) text, then pull out the quoted lines.
    start = re.search(r'"lyrics"\s*:\s*\[', text)
    if start:
        tail = text[start.end():]
        cut = re.search(r'"cover_prompt"', tail)
        if cut:
            tail = tail[: cut.start()]
        lines = re.findall(r'"((?:[^"\\]|\\.)*)"', tail)
        data["lyrics"] = [m.replace("\\n", "\n") for m in lines]
    else:
        s = re.search(r'"lyrics"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
        if s:
            data["lyrics"] = s.group(1).replace("\\n", "\n")
    return data


def _to_song(answer: str) -> Song:
    data = _extract_json(answer)

    def as_text(value, joiner=", ") -> str:
        if isinstance(value, list):
            return joiner.join(str(v) for v in value)
        return str(value or "")

    def as_int(value, default=90) -> int:
        try:
            return int(re.sub(r"[^0-9]", "", str(value)) or default)
        except ValueError:
            return default

    return Song(
        title=as_text(data.get("title", "Untitled")).strip() or "Untitled",
        mood=as_text(data.get("mood")).strip(),
        genre=as_text(data.get("genre")).strip(),
        tempo_bpm=as_int(data.get("tempo_bpm")),
        instruments=as_text(data.get("instruments")).strip(),
        tags=as_text(data.get("tags")).strip(),
        cover_prompt=as_text(data.get("cover_prompt")).strip(),
        lyrics=as_text(data.get("lyrics"), joiner="\n").strip(),
        raw=answer,
    )


def write_song(image_path: str) -> Song:
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info

    processor = AutoProcessor.from_pretrained(config.VLM_MODEL)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.VLM_MODEL, torch_dtype=config.DTYPE, device_map=config.DEVICE
    )

    messages = [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": Image.open(image_path).convert("RGB")},
                {"type": "text", "text": _INSTRUCTION},
            ],
        },
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(config.DEVICE)

    # The VLM occasionally returns a response we can't fully parse (empty lyrics
    # or cover prompt). Retry once at a lower temperature before giving up, so a
    # live demo never ends up with a "la la la" song.
    best: Song | None = None
    for attempt in range(2):
        temperature = config.VLM_TEMPERATURE if attempt == 0 else 0.4
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=config.VLM_MAX_NEW_TOKENS,
                do_sample=True,
                temperature=temperature,
            )
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
        answer = processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        best = _to_song(answer)
        if best.lyrics and best.cover_prompt:
            break

    if config.FREE_AFTER_STAGE:
        model = model.cpu()
        free(model, processor)

    return best
