# Foundation Models Midterm → Final Report

## 1) Project Theme
**Multi-Level Classroom Image Understanding System**

A single classroom image is analyzed by three complementary foundation models to produce:
- semantic understanding,
- object-level structure,
- spatial/depth understanding.

## 2) Model Selection and Roles

### Qwen3-VL-2B-Instruct
- **Role:** semantic understanding + VQA.
- **Input:** one classroom image + a natural-language question.
- **Output:** textual answer and scene-level interpretation.
- **Why chosen:** satisfies Qwen requirement and provides human-readable reasoning.
- **Strengths:** flexible natural-language analysis.
- **Limitations:** output can vary with prompt quality.

### Florence-2-base-ft
- **Role:** object-level visual structure.
- **Input:** classroom image (+ task token such as `<OD>`).
- **Output:** structured object predictions and optional annotated image.
- **Why chosen:** complements scene summary with localization-oriented information.
- **Strengths:** prompt-based vision workflow.
- **Limitations:** output schema depends on task and model version.

### Depth Anything V2 Small
- **Role:** spatial understanding via monocular depth.
- **Input:** classroom image.
- **Output:** depth map visualization and shape metadata.
- **Why chosen:** adds spatial reasoning not provided by captioning/detection models.
- **Strengths:** intuitive and useful for layout interpretation.
- **Limitations:** relative (not absolute metric) depth.

## 3) Midterm Deliverable Mapping
- `qwen_demo.py`: standalone Qwen script.
- `florence_demo.py`: standalone Florence script.
- `depth_demo.py`: standalone depth script.

Each script:
1. loads one image,
2. runs one model,
3. prints result summary,
4. saves structured output under `outputs/`.

## 4) Final Project Integration Strategy
- Keep reusable model logic in `models/`.
- Keep CLI wrappers thin in top-level `*_demo.py` files.
- Use common save/device/image utilities in `utils/`.
- Final demo (`unified_demo.py`) orchestrates all three outputs on one image.

## 5) Reproducibility and Portability
- Uses `pathlib` for path handling.
- Includes CUDA → MPS → CPU fallback.
- Uses open model IDs and no token-based code path by default.
- Provides one requirements file and consistent output layout.

## 6) Current Limitations / Notes
- Large model downloads may take time on first run.
- Memory usage depends on image size and available VRAM.
- Model availability can change upstream; model IDs are configurable via CLI flags.
