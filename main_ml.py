# main_ml.py - ML-based Tyre OCR System
import os, json, cv2, time
from pathlib import Path
from Yolo.YOLO import YOLOv11
from OCR.vision import detect_text
from convert import warpPolar
from ML.text_processor import build_result
from plant_codes import PLANT_MAP

def lookup_plant_info(plant_code: str) -> dict:
    """
    Look up plant information from plant code.
    
    Args:
        plant_code: The plant code to look up (e.g., "10U", "6Y87")
        
    Returns:
        Dictionary with plant and country information, or None if not found
    """
    if not plant_code:
        return None
    
    # Clean the plant code (remove spaces, convert to uppercase)
    clean_code = plant_code.strip().upper()
    
    # Look up in plant map
    plant_info = PLANT_MAP.get(clean_code)
    if plant_info:
        return {
            "Plant Code": clean_code,
            "Factory": plant_info.get("plant", "Unknown"),
            "Country": plant_info.get("country", "Unknown")
        }
    
    return None

MODEL_PATH = "models/Tyre_Detect.onnx"
IMAGE_FOLDER = Path("sample_image")
DOC_IMG_DIR = Path("doc/img")
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# Start timing the entire script
script_start_time = time.time()

DOC_IMG_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_FOLDER.mkdir(parents=True, exist_ok=True)

# Clear previous images in doc/img directory
print("🧹 Clearing previous images from doc/img directory...")
for file_path in DOC_IMG_DIR.glob("*"):
    if file_path.is_file():
        file_path.unlink()
        print(f"  🗑️ Deleted: {file_path.name}")
print("✅ Previous images cleared\n")

# Initialize ML-based models
yolov11 = YOLOv11(MODEL_PATH, conf_thres=0.2, iou_thres=0.3)

print("🚀 ML-based Tyre OCR System")
print("=" * 40)
print("✓ YOLO tire detection model loaded")
print("✓ ML text processor functions loaded")
print("✓ Google Cloud Vision OCR ready")
print()

for image_path in sorted(p for p in IMAGE_FOLDER.iterdir() if p.suffix.lower() in VALID_EXT):
    image_start_time = time.time()
    print(f"📸 Processing: {image_path.name}")
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"  ⚠️ Could not read image: {image_path}")
        continue

    # YOLO tire detection
    ocr_crops, boxes, scores, class_ids = yolov11(img)
    vis = yolov11.draw_detections(img)
    cv2.imwrite(str(DOC_IMG_DIR / f"{image_path.stem}_detect.jpg"), vis)

    if not ocr_crops:
        print("  ⚠️ No OCR crops detected.")
        continue

    for (x1, y1, x2, y2) in ocr_crops:
        x1, y1, x2, y2 = map(int, (max(0,x1), max(0,y1), max(0,x2), max(0,y2)))
        crop = img[y1:y2, x1:x2, :]
        crop_path = DOC_IMG_DIR / f"{image_path.stem}_tyre.jpg"
        cv2.imwrite(str(crop_path), crop)

        # Enhance & OCR
        print("  🔄 Enhancing tire image for better OCR...")
        warpPolar(str(crop_path))
        enhanced_convert = DOC_IMG_DIR / f"{image_path.stem}_tyre_convert.jpg"
        enhanced_square = DOC_IMG_DIR / f"{image_path.stem}_tyre_tyre_square.jpg"
        
        # OCR text extraction from both images
        print("  📝 Running OCR on converted image...")
        ocr_text_convert = detect_text(str(enhanced_convert))
        
        print("  📝 Running OCR on square image...")
        ocr_text_square = detect_text(str(enhanced_square))
        
        # Combine OCR results from both images
        ocr_texts = []
        if ocr_text_convert:
            ocr_texts.append(ocr_text_convert)
        if ocr_text_square:
            ocr_texts.append(ocr_text_square)
        
        if not ocr_texts:
            print("  ⚠️ No characters detected from either image.")
            continue
        
        # Combine the OCR results (join with newline separator)
        ocr_text = "\n---\n".join(ocr_texts)

        print("   Processing with ML models...")
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
                    print(f"  🏭 Found plant: {plant_info['Factory']} ({plant_info['Country']})")

        # Save results
        out_path = IMAGE_FOLDER / f"{image_path.stem}_ml.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("=== ML-BASED TYRE OCR RESULTS ===\n")
            f.write(f"Image: {image_path.name}\n")
            f.write(f"Processing Method: Machine Learning Models\n")
            f.write(f"Timestamp: {os.popen('date').read().strip()}\n\n")
            
            f.write("Raw OCR Text (from both converted and square images):\n")
            f.write("-" * 30 + "\n")
            f.write(ocr_text if isinstance(ocr_text, str) else str(ocr_text))
            f.write("\n\n")
            
            # Add info about which images were used
            f.write("OCR Sources:\n")
            f.write("-" * 30 + "\n")
            if ocr_text_convert:
                f.write(f"✓ Converted image ({enhanced_convert.name}): {len(ocr_text_convert)} characters\n")
            else:
                f.write(f"✗ Converted image ({enhanced_convert.name}): No text detected\n")
            if ocr_text_square:
                f.write(f"✓ Square image ({enhanced_square.name}): {len(ocr_text_square)} characters\n")
            else:
                f.write(f"✗ Square image ({enhanced_square.name}): No text detected\n")
            f.write("\n")
            
            f.write("Extracted Tire Information:\n")
            f.write("-" * 30 + "\n")
            json.dump(result, f, indent=4, ensure_ascii=False)
            
            # Add plant information summary if available
            if 'Plant Information' in result:
                plant_info = result['Plant Information']
                f.write(f"\n\nPlant Information Summary:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Plant Code: {plant_info.get('Plant Code', 'N/A')}\n")
                f.write(f"Factory: {plant_info.get('Factory', 'N/A')}\n")
                f.write(f"Country: {plant_info.get('Country', 'N/A')}\n")
            
            # Add confidence analysis
            if "confidence" in result:
                f.write(f"\n\nConfidence Analysis:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Manufacturer Detection Confidence: {result['confidence']:.2%}\n")
                if result['confidence'] > 0.7:
                    f.write("✅ High confidence detection\n")
                elif result['confidence'] > 0.5:
                    f.write("⚠️ Medium confidence detection\n")
                else:
                    f.write("❌ Low confidence detection\n")

        print(f"  ✅ Results saved to {out_path.name}")
        
        # Display results
        print("  📊 Extracted Information:")
        print(f"     Manufacturer: {result.get('Manufacturer', 'N/A')}")
        print(f"     Model: {result.get('Tire model', 'N/A')}")
        print(f"     Size: {result.get('Tire size', 'N/A')}")
        if 'Scan TIN' in result and isinstance(result['Scan TIN'], dict):
            dot_info = result['Scan TIN']
            print(f"     DOT Code: {dot_info.get('DOT Code', 'N/A')}")
            print(f"     Week/Year: {dot_info.get('Week Code', 'N/A')}/{dot_info.get('Year Code', 'N/A')}")
        
        # Display plant information if available
        if 'Plant Information' in result:
            plant_info = result['Plant Information']
            print(f"     Factory: {plant_info.get('Factory', 'N/A')}")
            print(f"     Country: {plant_info.get('Country', 'N/A')}")
        
        if "confidence" in result:
            print(f"     Confidence: {result['confidence']:.1%}")
    
    # Calculate and display image processing time
    image_processing_time = time.time() - image_start_time
    print(f"  ⏱️ Image processing time: {image_processing_time:.2f} seconds")

# Calculate and display total script execution time
total_execution_time = time.time() - script_start_time
print("\n🎉 ML-based processing complete!")
print(f"📁 Results saved in: {IMAGE_FOLDER}")
print(f"🖼️ Processed images in: {DOC_IMG_DIR}")
print(f"⏱️ Total execution time: {total_execution_time:.2f} seconds")
print(f"⏱️ Average time per image: {total_execution_time / max(1, len(list(IMAGE_FOLDER.glob('*')))):.2f} seconds")
