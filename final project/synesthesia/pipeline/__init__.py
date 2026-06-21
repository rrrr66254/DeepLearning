"""Synesthesia pipeline: image -> song + album cover.

Three HuggingFace foundation models, one task, each a different modality:
  1. Qwen2.5-VL       - sees the image, writes lyrics + style tags
  2. ACE-Step         - sings the lyrics into a full song (separate `ace` env)
  3. Stable Diffusion - paints an album cover
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterator

from pipeline import cover, lyricist, singer
from pipeline.lyricist import Song
from pipeline.manager import vram_summary


@dataclass
class Stage:
    """One streamed pipeline event."""
    kind: str                       # "vlm" | "song" | "cover"
    song: Song | None = None
    audio_path: str | None = None
    image_path: str | None = None
    note: str = ""


def stream_song(
    image_path: str,
    out_dir: str,
    *,
    duration: int | None = None,
    infer_step: int | None = None,
    sd_steps: int | None = None,
    seed: int | None = None,
    log=print,
) -> Iterator[Stage]:
    """Run the pipeline, yielding a Stage after each model finishes."""
    os.makedirs(out_dir, exist_ok=True)
    log(f"[0/3] start  {vram_summary()}")

    log("[1/3] Qwen2.5-VL: looking at image, writing lyrics + tags...")
    song = lyricist.write_song(image_path)
    log(f"      title={song.title!r} genre={song.genre!r}")
    yield Stage(kind="vlm", song=song, note=vram_summary())

    log("[2/3] ACE-Step: singing the song...")
    song_wav = os.path.join(out_dir, "song.wav")
    singer.sing(song.tags, song.lyrics, song_wav, duration=duration,
                infer_step=infer_step, seed=seed)
    log(f"      saved {song_wav}  {vram_summary()}")
    yield Stage(kind="song", song=song, audio_path=song_wav, note=vram_summary())

    log("[3/3] Stable Diffusion: painting the album cover...")
    cover_png = os.path.join(out_dir, "cover.png")
    cover.make_cover(song.cover_prompt, cover_png, steps=sd_steps)
    log(f"      saved {cover_png}  {vram_summary()}")
    yield Stage(kind="cover", song=song, image_path=cover_png, note=vram_summary())


def image_to_song(image_path: str, out_dir: str, log=print) -> dict:
    """Run the whole pipeline and return paths (CLI convenience)."""
    result = {"song": None, "audio_path": None, "image_path": None}
    for stage in stream_song(image_path, out_dir, log=log):
        if stage.song is not None:
            result["song"] = stage.song
        if stage.audio_path:
            result["audio_path"] = stage.audio_path
        if stage.image_path:
            result["image_path"] = stage.image_path
    return result
