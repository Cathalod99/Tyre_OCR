# Minimal Tyre OCR ML System

This is a minimal version of the Tyre OCR system containing only the essential files needed to run `main_ml.py`.

## What's Included

### Core Files
- `main_ml.py` - Main ML-based tire OCR processing script
- `convert.py` - Image processing and enhancement
- `plant_codes.py` - Plant code mapping functionality
- `requirements.txt` - Minimal Python dependencies

### ML Modules
- `ML/` - Machine learning processing modules
  - `text_processor.py` - Main text processing orchestrator
  - `size_extractor.py` - Tire size extraction
  - `tire_classifier.py` - Brand and model classification
  - `dot_extractor.py` - DOT code extraction

### Detection & OCR
- `Yolo/` - YOLO tire detection implementation
- `OCR/` - Google Cloud Vision OCR integration

### Models & Data
- `models/Tyre_Detect.onnx` - YOLO tire detection model
- `models/tire_classifier.pkl` - Tire classifier model
- `data/plant_codes.json` - Plant code database
- `google_credentials.json` - Google Cloud Vision API credentials

### Sample Data
- `sample_image/` - Sample tire images for testing

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up credentials:**
   ```bash
   python setup_minimal.py
   ```

3. **Add tire images:**
   Place tire images (JPG, JPEG, PNG) in the `sample_image/` directory

4. **Run the system:**
   ```bash
   python main_ml.py
   ```

## What the System Extracts

Based on the user's requirements, the system extracts:
- **Tire Make** (Manufacturer)
- **Tire Model** 
- **Tire Size** (e.g., 205/55R16)
- **DOT Code** (including plant information)

The system ignores load/speed ratings as requested.

## Output

Results are saved as `{image_name}_ml.txt` files in the `sample_image/` directory, containing:
- Raw OCR text
- Extracted tire information
- Plant information (if DOT code is found)
- Confidence analysis

## Requirements

- Python 3.7+
- Google Cloud Vision API credentials
- Internet connection for OCR processing

## Troubleshooting

- If OCR fails, check that `GOOGLE_APPLICATION_CREDENTIALS` environment variable is set
- If YOLO detection fails, the system will use fallback detection
- Ensure tire images are clear and well-lit for best results
