"""Standalone ACE-Step singer — runs inside the isolated `ace` conda env.

The main pipeline (Qwen2.5-VL + Stable Diffusion, transformers 5.x) calls this
as a subprocess because ACE-Step pins transformers==4.50. Reads lyrics from a
file (avoids shell-quoting issues) and writes a sung song to --out.

Invoked by pipeline/singer.py; not meant to be imported by the main env.
"""
from __future__ import annotations

import argparse
import os


def _patch_torchaudio_save() -> None:
    """torchaudio 2.10 routes save() through TorchCodec, which has no Windows
    wheel. ACE-Step asks for the soundfile backend anyway, so write the wav with
    soundfile directly and skip TorchCodec entirely.
    """
    import soundfile as sf
    import torch
    import torchaudio

    def _save(uri, src, sample_rate=None, channels_first=True, **kwargs):
        wav = src
        if hasattr(wav, "detach"):
            wav = wav.detach().cpu().to(torch.float32).numpy()
        if wav.ndim == 2 and channels_first:
            wav = wav.T  # (channels, frames) -> (frames, channels)
        sf.write(uri, wav, int(sample_rate))

    torchaudio.save = _save


def main() -> None:
    p = argparse.ArgumentParser(description="ACE-Step: lyrics + tags -> sung song")
    p.add_argument("--prompt", required=True, help="style tags, comma separated")
    p.add_argument("--lyrics-file", required=True, help="utf-8 file with lyrics")
    p.add_argument("--out", required=True, help="output wav path")
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument("--infer-step", type=int, default=27)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint", default="", help="empty = auto-download from HF")
    p.add_argument("--cpu-offload", type=int, default=1)
    p.add_argument("--device-id", type=int, default=0)
    args = p.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.device_id))
    _patch_torchaudio_save()

    from acestep.pipeline_ace_step import ACEStepPipeline

    with open(args.lyrics_file, encoding="utf-8") as fh:
        lyrics = fh.read()

    pipe = ACEStepPipeline(
        checkpoint_dir=args.checkpoint,
        dtype="bfloat16",
        cpu_offload=bool(args.cpu_offload),
    )
    pipe(
        audio_duration=args.duration,
        prompt=args.prompt,
        lyrics=lyrics,
        infer_step=args.infer_step,
        manual_seeds=[args.seed],
        save_path=args.out,
    )
    print(f"ACE_SINGER_SAVED {args.out}")


if __name__ == "__main__":
    main()
