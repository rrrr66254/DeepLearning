# Synesthesia — Image → Song + Album Cover

Give it a picture. It writes a song about it, **sings** it, and paints an album
cover for it.

One **task** (image → song), three HuggingFace **foundation models**, each owning
a different modality:

| Stage | Model | Modality | Job |
|---|---|---|---|
| 1. See & write | [`Qwen/Qwen2.5-VL-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) | vision + language | reads the image's mood, writes lyrics + style tags |
| 2. Sing | [`ACE-Step/ACE-Step-v1-3.5B`](https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B) | music + voice | turns lyrics + tags into a full **sung** song |
| 3. Paint | [`stabilityai/sdxl-turbo`](https://huggingface.co/stabilityai/sdxl-turbo) | image generation | paints an album cover |

All weights come from the HuggingFace Hub and run **locally** — nothing is sent
to an API.

## Pipeline

```
image ─▶ Qwen2.5-VL ─▶ { title, mood, genre, tempo, lyrics, tags, cover_prompt }
                            │                              │
                   lyrics + tags                     cover_prompt
                            ▼                              ▼
                      ACE-Step  ─▶ song.wav        Stable Diffusion ─▶ cover.png
```

To stay inside a 12 GB GPU, each model is loaded for its stage and freed before
the next (`FREE_AFTER_STAGE` in `config.py`).

## Why two conda envs

ACE-Step pins `transformers==4.50`, which is incompatible with the `transformers`
5.x that Qwen2.5-VL needs. So ACE-Step lives in its own `ace` env and is called
as a subprocess (`ace_singer.py`); the main `synesthesia` env runs Qwen2.5-VL and
Stable Diffusion.

## Setup

Requires an NVIDIA GPU with a working CUDA build of PyTorch. Developed on an
RTX 5070 Ti (Blackwell, `torch 2.10.0+cu128`).

```bash
# --- main env: Qwen2.5-VL + Stable Diffusion ---
conda create -n synesthesia python=3.11 -y
conda activate synesthesia
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# --- isolated env: ACE-Step singer ---
conda create -n ace python=3.11 -y
conda activate ace
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
# install ACE-Step from GitHub (the PyPI `ace-step` package is broken).
# PYTHONUTF8=1 avoids a setup.py decode error on non-UTF-8 locales (e.g. Korean Windows).
PYTHONUTF8=1 pip install git+https://github.com/ace-step/ACE-Step.git
```

Notes:
- The singer env is resolved by **name** (`conda run -n ace`), so no path editing
  is needed as long as the env is called `ace`. To use a different name set the
  `ACE_ENV_NAME` env var; to point at a specific interpreter set `ACE_ENV_PYTHON`.
- `ace_singer.py` patches `torchaudio.save` to use soundfile, so you do **not**
  need TorchCodec (which has no Windows wheel).

## Run

**Web app (recommended for the live demonstration)** — upload an image and watch
each model's output stream into the page as it finishes:

```bash
python server.py         # opens http://localhost:8000
```

The page shows, in order: Qwen2.5-VL's lyrics + tags, ACE-Step's sung song
(playable + downloadable), and Stable Diffusion's album cover. Sliders let you
set the song length, singing/cover quality (diffusion steps), and the seed before
generating; their defaults come from `config.py`.

Command line:

```bash
python demo.py examples/sunset.jpg --out outputs/cli
```

Model weights download automatically from the HuggingFace Hub on first run
(Qwen2.5-VL ~7 GB, ACE-Step ~3.5 GB, SDXL-Turbo ~7 GB) and are cached afterward.

## Config

All model IDs and hyperparameters live in `config.py`:

- `ACE_DURATION_S` — length of the song in seconds
- `ACE_INFER_STEP` — ACE-Step diffusion steps (quality vs. speed)
- `ACE_CPU_OFFLOAD` — `False` keeps ACE-Step resident (~12 s load) vs. `True` (~150 s, less VRAM)
- `SD_MODEL` / `SD_STEPS` — album-cover model and steps
- `FREE_AFTER_STAGE` — free each model after its stage to fit 12 GB

## Layout

```
synesthesia/
  server.py         # FastAPI web app (streams each model's output)
  web/index.html    # browser UI: upload, per-model cards, audio + cover
  demo.py           # CLI
  ace_singer.py     # ACE-Step entry point, run inside the `ace` env
  config.py         # model IDs + hyperparameters
  pipeline/
    lyricist.py     # 1) Qwen2.5-VL       — see + write
    singer.py       # 2) ACE-Step         — sing (subprocess to `ace` env)
    cover.py        # 3) Stable Diffusion — album cover
    manager.py      # VRAM helpers
    __init__.py     # stream_song() / image_to_song() orchestrator
  requirements.txt
  REPORT.md         # full project write-up (design, decisions, challenges)
```
