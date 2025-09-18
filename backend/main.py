from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
import sys
import tempfile
import cv2
import numpy as np
from pathlib import Path
import logging

# Add the parent directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent))

from Yolo.YOLO import YOLOv11
from OCR.vision import detect_text
from convert import warpPolar
from ML.text_processor import build_result
from plant_codes import PLANT_MAP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Tire OCR API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your mobile app's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
yolov11 = None
MODEL_PATH = "../models/Tyre_Detect.onnx"

def lookup_plant_info(plant_code: str) -> dict:
    """
    Look up plant information from plant code.
    """
    if not plant_code:
        return None
    
    clean_code = plant_code.strip().upper()
    plant_info = PLANT_MAP.get(clean_code)
    if plant_info:
        return {
            "Plant Code": clean_code,
            "Factory": plant_info.get("plant", "Unknown"),
            "Country": plant_info.get("country", "Unknown")
        }
    return None

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global yolov11
    try:
        logger.info("Loading YOLO model...")
        yolov11 = YOLOv11(MODEL_PATH, conf_thres=0.2, iou_thres=0.3)
        logger.info("YOLO model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {e}")
        raise e

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Tire OCR API is running"}

@app.post("/analyze-tire")
async def analyze_tire(image: UploadFile = File(...)):
    """
    Analyze a tire image and extract information
    """
    if not yolov11:
        raise HTTPException(status_code=500, detail="YOLO model not loaded")
    
    try:
        # Validate file type
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image data
        image_data = await image.read()
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        
        logger.info(f"Processing image: {image.filename}")
        
        # YOLO tire detection
        ocr_crops, boxes, scores, class_ids = yolov11(img)
        
        if not ocr_crops:
            return JSONResponse(
                status_code=400,
                content={"error": "No tire detected in the image. Please ensure the tire is clearly visible."}
            )
        
        # Process the first detected tire
        (x1, y1, x2, y2) = ocr_crops[0]
        x1, y1, x2, y2 = map(int, (max(0, x1), max(0, y1), max(0, x2), max(0, y2)))
        crop = img[y1:y2, x1:x2, :]
        
        # Create temporary files for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            crop_path = temp_dir / "tire_crop.jpg"
            enhanced_path = temp_dir / "tire_enhanced.jpg"
            
            # Save crop
            cv2.imwrite(str(crop_path), crop)
            
            # Enhance & OCR
            logger.info("Enhancing tire image for better OCR...")
            warpPolar(str(crop_path))
            enhanced = temp_dir / "tire_crop_convert.jpg"
            
            if not enhanced.exists():
                raise HTTPException(status_code=500, detail="Image enhancement failed")
            
            # OCR text extraction
            logger.info(f"Running OCR on enhanced image: {enhanced}")
            try:
                ocr_text = detect_text(str(enhanced))
                logger.info(f"OCR result: {ocr_text}")
                if not ocr_text:
                    return JSONResponse(
                        status_code=400,
                        content={"error": "No text detected in the tire image. Please try a clearer image with better lighting and contrast."}
                    )
            except Exception as ocr_error:
                logger.error(f"OCR failed: {str(ocr_error)}")
                return JSONResponse(
                    status_code=500,
                    content={"error": f"OCR processing failed: {str(ocr_error)}"}
                )
            
            logger.info("Processing with ML models...")
            
            # ML models → dict
            result = build_result(ocr_text)
            if not isinstance(result, dict):
                result = {"_raw": str(result)}
            
            # Add plant information if DOT code is available
            if 'Scan TIN' in result and isinstance(result['Scan TIN'], dict):
                dot_info = result['Scan TIN']
                plant_code = dot_info.get('Plant Code')
                if plant_code:
                    plant_info = lookup_plant_info(plant_code)
                    if plant_info:
                        result['Plant Information'] = plant_info
                        logger.info(f"Found plant: {plant_info['Factory']} ({plant_info['Country']})")
            
            # Clean up the result for mobile app
            cleaned_result = {
                "Manufacturer": result.get('Manufacturer', 'N/A'),
                "Tire model": result.get('Tire model', 'N/A'),
                "Tire size": result.get('Tire size', 'N/A'),
                "Load index and speed rating": result.get('Load index and speed rating', 'N/A'),
                "Scan TIN": {
                    "Full DOT": result.get('Scan TIN', {}).get('Full DOT', 'N/A'),
                    "DOT Code": result.get('Scan TIN', {}).get('DOT Code', 'N/A'),
                    "Week Code": result.get('Scan TIN', {}).get('Week Code', 'N/A'),
                    "Year Code": result.get('Scan TIN', {}).get('Year Code', 'N/A'),
                    "Plant Code": result.get('Scan TIN', {}).get('Plant Code', 'N/A'),
                    "Plant Name": result.get('Scan TIN', {}).get('Plant Name', 'N/A'),
                    "Plant Country": result.get('Scan TIN', {}).get('Plant Country', 'N/A'),
                    "Plant Manufacturer": result.get('Scan TIN', {}).get('Plant Manufacturer', 'N/A'),
                }
            }
            
            # Add plant information if available
            if 'Plant Information' in result:
                plant_info = result['Plant Information']
                cleaned_result['Scan TIN']['Plant Name'] = plant_info.get('Factory', 'N/A')
                cleaned_result['Scan TIN']['Plant Country'] = plant_info.get('Country', 'N/A')
            
            logger.info("Analysis completed successfully")
            return cleaned_result
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
