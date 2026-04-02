"""Standalone midterm demo for Qwen visual understanding."""

from __future__ import annotations

import argparse
from pathlib import Path

from models.qwen_model import run_qwen
from utils.save_utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen3-VL-2B-Instruct on one classroom image.")
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--question", required=True, help="Question to ask about the image")
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-2B-Instruct", help="HF model id")
    parser.add_argument("--output", default="outputs/qwen/result.json", help="Where to save JSON output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_qwen(image_path=args.image, question=args.question, model_id=args.model_id)
    output_path = save_json(result, Path(args.output))

    print("\n=== Qwen Demo Result ===")
    print(f"Question: {result['question']}")
    print(f"Answer  : {result['answer']}")
    print(f"Saved   : {output_path}")


if __name__ == "__main__":
    main()
