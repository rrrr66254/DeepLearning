"""Standalone midterm demo for Depth Anything V2 Small."""

from __future__ import annotations

import argparse
from pathlib import Path

from models.depth_model import run_depth
from utils.save_utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Depth Anything V2 Small on one classroom image.")
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--model-id", default="depth-anything/Depth-Anything-V2-Small-hf", help="HF model id")
    parser.add_argument("--depth-output", default="outputs/depth/depth_result.png", help="Where to save depth map")
    parser.add_argument("--json-output", default="outputs/depth/result.json", help="Where to save JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_depth(image_path=args.image, model_id=args.model_id, save_depth_to=args.depth_output)
    json_path = save_json(result, Path(args.json_output))

    print("\n=== Depth Demo Result ===")
    print(f"Depth map: {result['depth_map']}")
    print(f"Shape    : {result['shape']}")
    print(f"JSON     : {json_path}")


if __name__ == "__main__":
    main()
