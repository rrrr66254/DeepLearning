"""Standalone midterm demo for Florence-2 object-level analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

from models.florence_model import run_florence
from utils.save_utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Florence-2 object detection on one classroom image.")
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--model-id", default="microsoft/Florence-2-base-ft", help="HF model id")
    parser.add_argument("--task-prompt", default="<OD>", help="Florence task prompt")
    parser.add_argument("--json-output", default="outputs/florence/result.json", help="Where to save JSON")
    parser.add_argument(
        "--image-output",
        default="outputs/florence/detection_result.jpg",
        help="Where to save annotated visualization",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_florence(
        image_path=args.image,
        model_id=args.model_id,
        task_prompt=args.task_prompt,
        save_annotated_to=args.image_output,
    )
    json_path = save_json(result, Path(args.json_output))

    print("\n=== Florence Demo Result ===")
    print(f"Objects : {result['objects']}")
    print(f"JSON    : {json_path}")
    print(f"Image   : {result['output_image']}")


if __name__ == "__main__":
    main()
