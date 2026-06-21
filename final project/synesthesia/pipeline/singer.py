"""Stage 2 - ACE-Step: turn lyrics + style tags into a sung song.

ACE-Step pins transformers==4.50, incompatible with the Qwen2.5-VL stage, so it
lives in a separate `ace` conda env and is invoked as a subprocess. Lyrics are
passed via a temp file to avoid shell-quoting issues; the song is written to
`out_path`.
"""
from __future__ import annotations

import os
import subprocess
import tempfile

import config


def sing(tags: str, lyrics: str, out_path: str, duration: int | None = None) -> str:
    duration = duration or config.ACE_DURATION_S
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    fd, lyrics_file = tempfile.mkstemp(suffix=".txt")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(lyrics or "[verse]\nla la la")

    cmd = [
        config.ACE_ENV_PYTHON,
        config.ACE_SINGER_SCRIPT,
        "--prompt", tags or "pop, emotional, soft vocals",
        "--lyrics-file", lyrics_file,
        "--out", out_path,
        "--duration", str(duration),
        "--infer-step", str(config.ACE_INFER_STEP),
        "--seed", str(config.ACE_SEED),
        "--cpu-offload", "1" if config.ACE_CPU_OFFLOAD else "0",
    ]
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    try:
        subprocess.run(cmd, check=True, env=env)
    finally:
        os.unlink(lyrics_file)

    if not os.path.exists(out_path):
        raise RuntimeError("ACE-Step did not produce an output file")
    return out_path
