import cv2
from Yolo import YOLOv11
from OCR import *
from convert import warpPolar
from OpenAI import *
import json
import os
# Initialize yolov8 object detector

model_path = "models/Tyre_Detect.onnx"
yolov11_detector = YOLOv11(model_path, conf_thres=0.2, iou_thres=0.3)

image_folder = "sample_image"
file_list = os.listdir(image_folder)
valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

for file_name in file_list:
    if not os.path.splitext(file_name)[1].lower() in valid_extensions:
        continue
    print(f"Processing: {file_name}")
    img_path = os.path.join(image_folder, file_name)
    # Read image
    img = cv2.imread(img_path)
    # Detect Objects
    ocr_crops,boxes, scores, class_ids = yolov11_detector(img)

    # Draw detections
    combined_img = yolov11_detector.draw_detections(img)

    # cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
    # cv2.imshow("Detected Objects", combined_img)
    cv2.imwrite(f"doc/img/{file_name.split('.')[0]}_detect.jpg", combined_img)

    for ocr_crop in ocr_crops:
        [x1, y1, x2, y2] = ocr_crop
        if (x1 < 0): 
            x1 = 0
        if (x2 < 0): 
            x2 = 0
        if (y1 < 0): 
            y1 = 0
        if (y2 < 0): 
            y2 = 0
        tyre_crop = img[int(y1):int(y2), int(x1):int(x2), :]
        # license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
        # license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
        # print(license_plate_text)
        # cv2.namedWindow("Tyre", cv2.WINDOW_NORMAL)
        # cv2.imshow("Tyre", tyre_crop)
        cv2.imwrite(f"doc/img/{file_name.split('.')[0]}_tyre.jpg", tyre_crop)
        warpPolar(f"doc/img/{file_name.split('.')[0]}_tyre.jpg")
        
        ocr_result = detect_text(f"doc/img/{file_name.split('.')[0]}_tyre_convert.jpg")
        if ocr_result == None:
            print("Not detected any character!!!")
            continue
        # print(ocr_result)
        final_result = get_tyre_info(ocr_result)  # This returns a string\
        # print(final_result)
        final_result = final_result.strip("```")
        final_result = final_result.replace("json", "")
        final_result_json = json.loads(final_result)  # Convert string to JSON

        text_name = file_name.split(".")[0] + ".txt"
        text_path = os.path.join(image_folder, text_name)
        with open(text_path, "w", encoding="utf-8") as file:
            # Write the OCR result
            file.write("OCR Result:\n")
            file.write(ocr_result)
            file.write("\n\n")  # Add spacing between sections

            # Write the final result JSON
            file.write("Final Result JSON:\n")
            json.dump(final_result_json, file, indent=4, ensure_ascii=False)  # Write JSON in a readable format

        print(f"OCR result and final JSON result have been saved to {text_name}.")

        # print(final_result_json)
    # cv2.waitKey(0)
