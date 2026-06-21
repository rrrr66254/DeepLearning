"""Synesthesia web app.

Serves a single-page HTML UI and streams the pipeline stage-by-stage so each
foundation model's output (text + audio) shows up in the browser as soon as it
is ready.

Run:  python server.py   ->   http://localhost:8000
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import uuid

import soundfile as sf
import uvicorn
from fastapi import FastAPI, Form, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import config
from pipeline import stream_song

HERE = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(HERE, "web")
OUT_DIR = os.path.join(HERE, config.OUTPUT_DIR)
UPLOAD_DIR = os.path.join(OUT_DIR, "uploads")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Synesthesia")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(WEB_DIR, "index.html"))


@app.get("/defaults")
def defaults() -> dict:
    """Slider defaults, sourced from config so the UI stays in sync."""
    return {
        "duration": config.ACE_DURATION_S,
        "infer_step": config.ACE_INFER_STEP,
        "sd_steps": config.SD_STEPS,
        "seed": config.ACE_SEED,
    }


def _audio_duration(path: str) -> float:
    info = sf.info(path)
    return round(info.frames / info.samplerate, 1)


@app.post("/generate")
def generate(
    image: UploadFile,
    duration: int = Form(config.ACE_DURATION_S),
    infer_step: int = Form(config.ACE_INFER_STEP),
    sd_steps: int = Form(config.SD_STEPS),
    seed: int = Form(config.ACE_SEED),
) -> StreamingResponse:
    session = uuid.uuid4().hex[:8]
    session_dir = os.path.join(OUT_DIR, session)
    os.makedirs(session_dir, exist_ok=True)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    img_path = os.path.join(UPLOAD_DIR, f"{session}_{image.filename or 'image'}")
    with open(img_path, "wb") as fh:
        fh.write(image.file.read())

    def emit():
        logs: list[str] = []
        try:
            for stage in stream_song(
                img_path, session_dir, duration=duration, infer_step=infer_step,
                sd_steps=sd_steps, seed=seed, log=logs.append,
            ):
                payload: dict = {"kind": stage.kind, "note": stage.note}
                if stage.song is not None:
                    payload["song"] = dataclasses.asdict(stage.song)
                if stage.audio_path:
                    name = os.path.basename(stage.audio_path)
                    payload["audio_url"] = f"/outputs/{session}/{name}"
                    payload["duration"] = _audio_duration(stage.audio_path)
                if stage.image_path:
                    name = os.path.basename(stage.image_path)
                    payload["image_url"] = f"/outputs/{session}/{name}"
                payload["log"] = "\n".join(logs)
                yield json.dumps(payload, ensure_ascii=False) + "\n"
        except Exception as exc:  # surface errors to the browser
            import traceback
            yield json.dumps(
                {"kind": "error", "message": str(exc), "log": traceback.format_exc()}
            ) + "\n"

    return StreamingResponse(emit(), media_type="application/x-ndjson")


app.mount("/outputs", StaticFiles(directory=OUT_DIR), name="outputs")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    print(f"device={config.DEVICE} dtype={config.DTYPE}")
    print(f"open http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
