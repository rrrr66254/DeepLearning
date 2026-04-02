# Multi-Level Classroom Image Understanding System

This project is structured to satisfy both:
1. **Midterm requirement**: 3 separate foundation model demos (including at least one Qwen model).
2. **Final project requirement**: one integrated pipeline over a single classroom image.

## Selected Models
1. **Qwen/Qwen3-VL-2B-Instruct** (semantic understanding + VQA)
2. **microsoft/Florence-2-base-ft** (object-level structure)
3. **depth-anything/Depth-Anything-V2-Small-hf** (spatial/depth understanding)

## Project Structure

```text
foundation_project/
├── README.md
├── requirements.txt
├── report.md
├── sample_images/
│   ├── README.md
│   └── class_photo.jpg  # add your own image
├── outputs/
│   ├── qwen/
│   ├── florence/
│   └── depth/
├── models/
│   ├── qwen_model.py
│   ├── florence_model.py
│   └── depth_model.py
├── utils/
│   ├── device.py
│   ├── image_io.py
│   └── save_utils.py
├── qwen_demo.py
├── florence_demo.py
├── depth_demo.py
└── unified_demo.py
```

## Setup

### 1) Create environment
```bash
python -m venv .venv
```

### 2) Activate environment
- Linux/macOS:
  ```bash
  source .venv/bin/activate
  ```
- Windows PowerShell:
  ```powershell
  .venv\Scripts\Activate.ps1
  ```

### 3) Install dependencies
```bash
pip install -r requirements.txt
```

## How to Run (Midterm Scripts)

> Add your image as `sample_images/class_photo.jpg` first.

### Depth demo
```bash
python depth_demo.py --image sample_images/class_photo.jpg
```
Outputs:
- `outputs/depth/depth_result.png`
- `outputs/depth/result.json`

### Florence demo
```bash
python florence_demo.py --image sample_images/class_photo.jpg
```
Outputs:
- `outputs/florence/detection_result.jpg`
- `outputs/florence/result.json`

### Qwen demo
```bash
python qwen_demo.py --image sample_images/class_photo.jpg --question "What is happening in this classroom?"
```
Outputs:
- `outputs/qwen/result.json`

## How to Run (Final Integrated Script)

```bash
python unified_demo.py --image sample_images/class_photo.jpg
```
Outputs:
- per-model JSON and images in `outputs/`
- final merged file: `outputs/combined_result.json`

## Reproducibility and Cross-Platform Notes
- Uses `pathlib` for OS-independent path handling.
- Device helper picks `cuda` → `mps` → `cpu` safely.
- Model IDs are CLI-configurable to handle upstream naming/version changes.
- No explicit login/token flow is implemented in scripts.

## Limitations
- First run requires downloading model weights from Hugging Face.
- Performance depends on available RAM/VRAM.
- Some model repositories may change over time; if that happens, pass an updated `--model-id`.
