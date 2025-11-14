# Tyre OCR Pipeline

This branch contains only the components required to run `python main_ml.py` for tire sidewall OCR:

- `main_ml.py`: entry point orchestrating detection, enhancement, OCR, and ML parsing
- `Yolo/`: ONNX-based YOLOv11 tyre detector wrapper and utilities
- `convert.py`: polar warp + enhancement helpers used before OCR
- `OCR/vision.py`: Google Vision wrapper for text detection (expects credentials via environment variables or file path)
- `ML/`: text post-processing models (manufacturer, size, DOT, etc.)
- `plant_codes.py`: lookup table for DOT plant codes
- `models/Tyre_Detect.onnx`: trained tyre detector weights

## Usage

1. Create a virtual environment and `pip install -r requirements.txt`.
2. Set up Google Vision credentials (e.g., `export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json`).
3. Drop tyre sidewall photos into `sample_image/`.
4. Run `python main_ml.py`.
5. OCR outputs land in `sample_image/<name>_ml.txt` and visualization assets in `doc/img/`.

Non-OCR components (damage detection, apps, deployment scripts, etc.) were intentionally excluded from this branch.
