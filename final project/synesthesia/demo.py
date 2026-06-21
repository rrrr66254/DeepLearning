"""Command-line demo: turn an image into a sung song + album cover.

Run:  python demo.py path/to/image.jpg --out outputs/cli
"""
from __future__ import annotations

import argparse

import config
from pipeline import image_to_song


def main() -> None:
    parser = argparse.ArgumentParser(description="Image -> song + cover (Synesthesia)")
    parser.add_argument("image", help="path to an input image")
    parser.add_argument("--out", default="outputs/cli", help="output directory")
    args = parser.parse_args()

    print(f"device={config.DEVICE} dtype={config.DTYPE}")
    result = image_to_song(args.image, args.out)
    song = result["song"]

    print("\n=== SONG ===")
    print(f"title : {song.title}")
    print(f"genre : {song.genre} | mood: {song.mood}")
    print(f"tags  : {song.tags}")
    print(f"lyrics:\n{song.lyrics}")
    print(f"\nsong  -> {result['audio_path']}")
    print(f"cover -> {result['image_path']}")


if __name__ == "__main__":
    main()
