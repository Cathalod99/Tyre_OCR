# main.py - GPT-based Tyre OCR System
import os, json, cv2
from pathlib import Path
from Yolo.YOLO import YOLOv11
from OCR.vision import detect_text
from convert import warpPolar
from OpenAI.gpt import get_tyre_info  # returns a dict

MODEL_PATH = "models/Tyre_Detect.onnx"
IMAGE_FOLDER = Path("sample_image")
DOC_IMG_DIR = Path("doc/img")
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

DOC_IMG_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_FOLDER.mkdir(parents=True, exist_ok=True)

# Clear previous images in doc/img directory
print("🧹 Clearing previous images from doc/img directory...")
for file_path in DOC_IMG_DIR.glob("*"):
    if file_path.is_file():
        file_path.unlink()
        print(f"  🗑️ Deleted: {file_path.name}")
print("✅ Previous images cleared\n")

# Initialize models
yolov11 = YOLOv11(MODEL_PATH, conf_thres=0.2, iou_thres=0.3)

for image_path in sorted(p for p in IMAGE_FOLDER.iterdir() if p.suffix.lower() in VALID_EXT):
    print(f"Processing: {image_path.name}")
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"⚠️ Could not read image: {image_path}")
        continue

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
        enhanced = DOC_IMG_DIR / f"{image_path.stem}_tyre_convert.jpg"
        print(f"  📁 Enhanced image saved to: {enhanced}")
        ocr_text = detect_text(str(enhanced))
        if not ocr_text:
            print("  ⚠️ No characters detected.")
            continue

        # LLM + rules → dict
        result = get_tyre_info(ocr_text)
        if not isinstance(result, dict):
            s = str(result).strip().strip("`").replace("json", "", 1)
            try:
                result = json.loads(s)
            except Exception:
                result = {"_raw": s}

        # Save alongside image
        out_path = IMAGE_FOLDER / f"{image_path.stem}.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("OCR Result:\n")
            f.write(ocr_text if isinstance(ocr_text, str) else str(ocr_text))
            f.write("\n\nFinal Result JSON:\n")
            json.dump(result, f, indent=4, ensure_ascii=False)

        print(f"OCR result and final JSON result have been saved to {out_path.name}.")