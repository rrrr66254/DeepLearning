"""Stage 2 - ACE-Step: turn lyrics + style tags into a sung song.

ACE-Step pins transformers==4.50, incompatible with the Qwen2.5-VL stage, so it
lives in a separate `ace` conda env and is invoked as a subprocess. Lyrics are
passed via a temp file to avoid shell-quoting issues; the song is written to
`out_path`.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

import config


def _ace_launcher() -> list[str]:
    """How to launch python in the ACE-Step env.

    Prefer an explicit python.exe if ACE_ENV_PYTHON points to one; otherwise
    resolve the env by name via `conda run -n <ACE_ENV_NAME>` so the project is
    portable across machines without editing any path.
    """
    py = config.ACE_ENV_PYTHON
    if py and os.path.exists(py):
        return [py]
    conda = shutil.which("conda") or "conda"
    return [conda, "run", "--no-capture-output", "-n", config.ACE_ENV_NAME, "python"]


def sing(
    tags: str,
    lyrics: str,
    out_path: str,
    duration: int | None = None,
    infer_step: int | None = None,
    seed: int | None = None,
) -> str:
    duration = duration or config.ACE_DURATION_S
    infer_step = infer_step or config.ACE_INFER_STEP
    seed = config.ACE_SEED if seed is None else seed
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    fd, lyrics_file = tempfile.mkstemp(suffix=".txt")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(lyrics or "[verse]\nla la la")

    cmd = [
        *_ace_launcher(),
        config.ACE_SINGER_SCRIPT,
        "--prompt", tags or "pop, emotional, soft vocals",
        "--lyrics-file", lyrics_file,
        "--out", out_path,
        "--duration", str(duration),
        "--infer-step", str(infer_step),
        "--seed", str(seed),
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
