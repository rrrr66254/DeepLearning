"""Central configuration for the Synesthesia image-to-song pipeline.

All model IDs and tunable hyperparameters live here so the pipeline modules
stay free of hard-coded constants.

Three HuggingFace foundation models, each a different modality:
  1. Qwen2.5-VL    (vision + language) - sees the image, writes lyrics + tags
  2. ACE-Step      (music + singing)   - turns lyrics + tags into a sung song
  3. Stable Diffusion (image gen)      - paints an album cover
"""
from __future__ import annotations

import os

# Some conda installs ship duplicate OpenMP runtimes; allow them so imports
# don't hard-crash on Windows.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

HERE = os.path.dirname(os.path.abspath(__file__))

# --- Devices / dtypes -------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# Free each model from VRAM after its stage finishes. Keeps peak usage low
# enough to fit a 12 GB laptop GPU.
FREE_AFTER_STAGE = True

# --- (1) Vision-language: Qwen2.5-VL ----------------------------------------
VLM_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
VLM_MAX_NEW_TOKENS = 1024
VLM_TEMPERATURE = 0.7

# --- (2) Singer: ACE-Step (runs in the isolated `ace` conda env) ------------
# ACE-Step pins transformers==4.50, so it lives in its own env and is called
# as a subprocess. The model weights download from the HuggingFace Hub.
# By default the env is resolved by name via `conda run -n ace` (portable, no
# machine-specific path). Set ACE_ENV_PYTHON to a python.exe to override.
ACE_ENV_NAME = os.environ.get("ACE_ENV_NAME", "ace")
ACE_ENV_PYTHON = os.environ.get("ACE_ENV_PYTHON", "")
ACE_SINGER_SCRIPT = os.path.join(HERE, "ace_singer.py")
ACE_DURATION_S = 25
ACE_INFER_STEP = 27
ACE_SEED = 42
# False keeps ACE-Step resident on the GPU: ~12 s load vs ~150 s with offload.
# The VLM stage is freed before ACE-Step runs, so 12 GB is enough.
ACE_CPU_OFFLOAD = False

# --- (3) Album cover: Stable Diffusion --------------------------------------
SD_MODEL = "stabilityai/sdxl-turbo"
SD_STEPS = 3
SD_GUIDANCE = 0.0  # turbo models use no classifier-free guidance
SD_SIZE = 512

# --- Paths ------------------------------------------------------------------
OUTPUT_DIR = "outputs"
