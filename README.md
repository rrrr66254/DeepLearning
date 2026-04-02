# DeepLearning

This repository now includes a complete project scaffold for the **Foundation Models Midterm → Final Project Plan** under:

- `foundation_project/`

## Quick start

```bash
cd foundation_project
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows (PowerShell): .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run standalone midterm demos

```bash
python depth_demo.py --image sample_images/class_photo.jpg
python florence_demo.py --image sample_images/class_photo.jpg
python qwen_demo.py --image sample_images/class_photo.jpg --question "What is happening in this classroom?"
```

## Run final integrated demo

```bash
python unified_demo.py --image sample_images/class_photo.jpg
```

For full details, see:
- `foundation_project/README.md`
- `foundation_project/report.md`
