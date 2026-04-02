"""Integrated final demo: semantic + object + spatial understanding."""

from __future__ import annotations

import argparse
from pathlib import Path

from models.depth_model import run_depth
from models.florence_model import run_florence
from models.qwen_model import run_qwen
from utils.save_utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full classroom image understanding pipeline.")
    parser.add_argument("--image", required=True, help="Path to classroom image")
    parser.add_argument(
        "--question",
        default="What is happening in this classroom?",
        help="Question for Qwen semantic reasoning",
    )
    parser.add_argument("--output-dir", default="outputs", help="Base output folder")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    qwen_result = run_qwen(args.image, args.question)
    florence_result = run_florence(args.image, save_annotated_to=output_dir / "florence" / "detection_result.jpg")
    depth_result = run_depth(args.image, save_depth_to=output_dir / "depth" / "depth_result.png")

    save_json(qwen_result, output_dir / "qwen" / "result.json")
    save_json(florence_result, output_dir / "florence" / "result.json")
    save_json(depth_result, output_dir / "depth" / "result.json")

    combined = {
        "title": "Multi-Level Classroom Image Understanding System",
        "input_image": str(Path(args.image).resolve()),
        "semantic_layer": qwen_result,
        "object_layer": florence_result,
        "spatial_layer": depth_result,
    }
    combined_path = save_json(combined, output_dir / "combined_result.json")

    print("\n=== Unified Demo Complete ===")
    print(f"Combined result saved to: {combined_path}")


if __name__ == "__main__":
    main()
