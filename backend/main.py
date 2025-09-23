from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
import os
import sys
import tempfile
import cv2
import numpy as np
from pathlib import Path
import logging
import hashlib
import time
import uuid
from typing import Optional

# Add the parent directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent))

from Yolo.YOLO import YOLOv11
from OCR.vision import detect_text

# Using Google Cloud Vision API only - no pytesseract needed
from convert import warpPolar
from ML.text_processor import build_result
from plant_codes import PLANT_MAP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for models
yolov11 = None
onnx_session = None
MODEL_PATH = "models/Tyre_Detect.onnx"  # Fixed path for container
MODEL_SHA256 = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models on startup"""
    global yolov11, onnx_session, MODEL_SHA256
    
    # Set up Google Cloud credentials from environment variable if available
    gcp_creds_base64 = os.getenv('GOOGLE_CLOUD_CREDENTIALS_BASE64')
    if gcp_creds_base64:
        try:
            import base64
            import json
            import tempfile
            
            # Decode and write credentials to temp file
            creds_json = base64.b64decode(gcp_creds_base64).decode('utf-8')
            creds_data = json.loads(creds_json)
            
            # Create temp file for credentials
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(creds_data, f)
                temp_creds_path = f.name
            
            # Set environment variable to point to temp file
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = temp_creds_path
            logger.info(f"Google Cloud credentials set from environment variable: {temp_creds_path}")
            
        except Exception as e:
            logger.error(f"Failed to set up Google Cloud credentials from environment: {e}")
    
    try:
        # Check ONNX Runtime availability first
        import onnxruntime as ort
        logger.info(f"ONNX Runtime version: {ort.__version__}")
        logger.info(f"Available providers: {ort.get_available_providers()}")
        
        # Log model path and size for sanity check
        from pathlib import Path
        model_path = Path(MODEL_PATH)
        logger.info("Model path=%s exists=%s size=%s bytes", 
                   model_path, model_path.exists(), 
                   model_path.stat().st_size if model_path.exists() else -1)
        
        # Calculate and log model SHA256 for verification
        if model_path.exists():
            MODEL_SHA256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
            logger.info("YOLO ONNX sha256=%s", MODEL_SHA256)
        
        # Test model loading with Railway-optimized settings
        logger.info("Testing ONNX Runtime with YOLO model...")
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 1
        sess_opts.inter_op_num_threads = 1
        sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        onnx_session = ort.InferenceSession(MODEL_PATH, sess_options=sess_opts, providers=["CPUExecutionProvider"])
        logger.info("ONNX Runtime test successful")
        
        # Log model metadata
        try:
            model_meta = onnx_session.get_modelmeta()
            logger.info("Model opset version=%s", getattr(model_meta, 'version', 'unknown'))
        except Exception as e:
            logger.warning("Could not get model metadata: %s", e)
        
        # Preload test - warm-up inference to catch bad opsets early
        logger.info("Running warm-up inference...")
        import numpy as np
        dummy_input = np.random.rand(1, 3, 640, 640).astype(np.float32)
        _ = onnx_session.run(None, {"images": dummy_input})
        logger.info("Warm-up inference successful")
        
        logger.info("Loading YOLO model...")
        yolov11 = YOLOv11(MODEL_PATH, conf_thres=0.2, iou_thres=0.3)
        logger.info("YOLO model loaded successfully")
    except ImportError as e:
        logger.warning(f"ONNX Runtime not available: {e}")
        logger.warning("YOLO detection will be disabled. Using fallback detection.")
        yolov11 = None
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {e}")
        logger.warning("YOLO detection will be disabled. Using fallback detection.")
        yolov11 = None
    
    yield
    
    # Cleanup code - graceful shutdown
    logger.info("Shutting down application...")
    if onnx_session:
        try:
            onnx_session.close()
            logger.info("ONNX session closed gracefully")
        except Exception as e:
            logger.warning("Error closing ONNX session: %s", e)

app = FastAPI(title="Tire OCR API", version="1.0.0", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your mobile app's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Removed fallback OCR - using Google Cloud Vision API only

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

@app.get("/health")
async def health_check():
    """Health check endpoint - lightweight, no model touch"""
    return {
        "status": "healthy", 
        "message": "Tire OCR API is running",
        "timestamp": time.time()
    }

@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint - asserts model is loaded"""
    yolo_status = "available" if yolov11 else "fallback mode"
    
    # Check ONNX Runtime availability
    try:
        import onnxruntime as ort
        onnx_available = True
        onnx_version = ort.__version__
        onnx_providers = ort.get_available_providers()
    except ImportError:
        onnx_available = False
        onnx_version = None
        onnx_providers = []
    
    return {
        "status": "ready" if yolov11 else "not_ready", 
        "message": "Tire OCR API readiness check",
        "yolo_status": yolo_status,
        "onnx": onnx_available,
        "onnx_version": onnx_version,
        "onnx_providers": onnx_providers,
        "model_sha256": MODEL_SHA256,
        "features": {
            "yolo_detection": yolov11 is not None,
            "ocr_processing": True,
            "ml_models": True
        }
    }

@app.get("/debug/onnx")
async def onnx_debug():
    """Debug endpoint to check ONNX Runtime status"""
    try:
        import onnxruntime as ort
        import platform
        
        # Check model file
        from pathlib import Path
        model_path = Path(MODEL_PATH)
        
        return {
            "onnxruntime": ort.__version__,
            "available_providers": ort.get_available_providers(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "yolo_model_loaded": yolov11 is not None,
            "model_path": MODEL_PATH,
            "model_exists": model_path.exists(),
            "model_size_bytes": model_path.stat().st_size if model_path.exists() else -1
        }
    except Exception as e:
        return {
            "onnxruntime": None, 
            "error": str(e),
            "yolo_model_loaded": False,
            "model_path": MODEL_PATH
        }

@app.get("/test-gcp")
async def test_gcp_credentials():
    """Test Google Cloud Vision API credentials"""
    try:
        gcp_creds = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        logger.info(f"Testing GCP credentials: {gcp_creds}")
        
        if not gcp_creds:
            return {"error": "GOOGLE_APPLICATION_CREDENTIALS not set"}
        
        if not os.path.exists(gcp_creds):
            return {"error": f"Credentials file not found: {gcp_creds}"}
        
        # Test with a simple image
        test_image_path = "sample_image/20240516_130139.jpg"
        if os.path.exists(test_image_path):
            ocr_text = detect_text(test_image_path)
            return {
                "status": "success",
                "credentials_file": gcp_creds,
                "credentials_exists": os.path.exists(gcp_creds),
                "test_ocr_result": ocr_text[:200] if ocr_text else "No text detected"
            }
        else:
            return {"error": "Test image not found"}
            
    except Exception as e:
        logger.error(f"GCP test failed: {str(e)}")
        return {"error": f"GCP test failed: {str(e)}"}

@app.post("/analyze-tire")
async def analyze_tire(request: Request, image: UploadFile = File(...)):
    """
    Analyze a tire image and extract information
    """
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    try:
        # Security checks
        if not image.content_type or not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Check file extension
        if image.filename:
            ext = Path(image.filename).suffix.lower()
            if ext not in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                raise HTTPException(status_code=400, detail="Unsupported image format")
        
        # Size limit check (8MB)
        image_data = await image.read()
        if len(image_data) > 8 * 1024 * 1024:  # 8MB
            raise HTTPException(status_code=400, detail="Image too large (max 8MB)")
        
        logger.info("request_id=%s filename=%s size_bytes=%d content_type=%s", 
                   request_id, image.filename, len(image_data), image.content_type)
        
        # Decode image data
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        
        # Debug logging
        logger.info(f"Image dimensions: {img.shape}")
        logger.info(f"Image data type: {img.dtype}")
        logger.info(f"Image size: {len(image_data)} bytes")
        
        # Preserve image quality - ensure we're working with high quality
        if img.shape[0] < 1000 or img.shape[1] < 1000:
            logger.warning(f"Image resolution is low: {img.shape[0]}x{img.shape[1]}")
        else:
            logger.info(f"Image resolution is good: {img.shape[0]}x{img.shape[1]}")
        
        logger.info(f"Processing image: {image.filename}")
        
        # YOLO tire detection or fallback to full image
        if yolov11:
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
            logger.info("Using YOLO-detected tire region")
        else:
            # Fallback: use the entire image
            crop = img
            logger.info("YOLO not available, using full image for OCR")
        
        # Create temporary files for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            crop_path = temp_dir / "tire_crop.jpg"
            enhanced_path = temp_dir / "tire_enhanced.jpg"
            
            # Save crop with high quality
            cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            # Enhance & OCR
            logger.info("Enhancing tire image for better OCR...")
            logger.info(f"Crop image saved to: {crop_path}")
            logger.info(f"Crop image size: {crop_path.stat().st_size} bytes")
            
            try:
                warpPolar(str(crop_path))
                enhanced = temp_dir / "tire_crop_convert.jpg"
                logger.info(f"Image enhancement completed. Enhanced image: {enhanced}")
                
                if not enhanced.exists():
                    raise HTTPException(status_code=500, detail="Image enhancement failed - enhanced image not created")
                    
                logger.info(f"Enhanced image size: {enhanced.stat().st_size} bytes")
            except Exception as enhance_error:
                logger.error(f"Image enhancement failed: {str(enhance_error)}")
                raise HTTPException(status_code=500, detail=f"Image enhancement failed: {str(enhance_error)}")
            
            # OCR text extraction
            logger.info(f"Running OCR on enhanced image: {enhanced}")
            logger.info(f"Enhanced image exists: {enhanced.exists()}")
            logger.info(f"Enhanced image size: {enhanced.stat().st_size if enhanced.exists() else 'N/A'} bytes")
            
            # Check Google Cloud credentials
            gcp_creds = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
            logger.info(f"Google Cloud credentials path: {gcp_creds}")
            logger.info(f"Credentials file exists: {os.path.exists(gcp_creds) if gcp_creds else 'N/A'}")
            
            # Check if enhanced image was created successfully
            logger.info(f"Enhanced image path: {enhanced}")
            logger.info(f"Enhanced image exists: {enhanced.exists()}")
            if enhanced.exists():
                logger.info(f"Enhanced image size: {enhanced.stat().st_size} bytes")
            
            try:
                ocr_text = detect_text(str(enhanced))
                logger.info(f"OCR result length: {len(ocr_text) if ocr_text else 0}")
                logger.info(f"OCR result preview: {ocr_text[:200] if ocr_text else 'None'}")
                logger.info(f"Full OCR text: {ocr_text}")
                if not ocr_text:
                    return JSONResponse(
                        status_code=400,
                        content={"error": "No text detected in the tire image. Please try a clearer image with better lighting and contrast."}
                    )
            except Exception as ocr_error:
                logger.error(f"Google Cloud Vision OCR failed: {str(ocr_error)}")
                return JSONResponse(
                    status_code=500,
                    content={"error": f"OCR processing failed: {str(ocr_error)}"}
                )
            
            logger.info("Processing with ML models...")
            logger.info(f"Input OCR text for ML processing: {ocr_text[:500]}...")
            
            # ML models → dict
            result = build_result(ocr_text)
            logger.info(f"ML processing result: {result}")
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
            
            # Log successful completion with latency
            latency_ms = int((time.time() - start_time) * 1000)
            logger.info("request_id=%s status=success latency_ms=%d yolo_used=%s", 
                       request_id, latency_ms, yolov11 is not None)
            return cleaned_result
            
    except HTTPException:
        latency_ms = int((time.time() - start_time) * 1000)
        logger.warning("request_id=%s status=client_error latency_ms=%d", request_id, latency_ms)
        raise
    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        logger.error("request_id=%s status=server_error latency_ms=%d error=%s", 
                    request_id, latency_ms, str(e))
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
