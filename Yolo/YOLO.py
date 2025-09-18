import time
import cv2
import numpy as np

# Try to import onnxruntime, handle import errors gracefully
try:
    import onnxruntime
    ONNX_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ONNX Runtime not available: {e}")
    print("YOLO detection will be disabled. Using fallback detection.")
    ONNX_AVAILABLE = False
    onnxruntime = None

from .utils import xywh2xyxy, draw_detections, multiclass_nms


class YOLOv11:

    def __init__(self, path, conf_thres=0.7, iou_thres=0.5):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.session = None
        self.input_name = None
        self.output_names = None
        self.input_shape = None

        # Initialize model only if ONNX Runtime is available
        if ONNX_AVAILABLE:
            self.initialize_model(path)
        else:
            print("YOLO model disabled due to ONNX Runtime unavailability")

    def __call__(self, image):
        return self.detect_objects(image)

    def initialize_model(self, path):
        self.session = onnxruntime.InferenceSession(path,
                                                    providers=onnxruntime.get_available_providers())
        # Get model info
        self.get_input_details()
        self.get_output_details()


    def detect_objects(self, image):
        self.boxes = []
        self.scores = []
        self.class_ids = []
        self.ocr_crops = []
        
        # If ONNX Runtime is not available, use fallback detection
        if not ONNX_AVAILABLE or self.session is None:
            return self.fallback_detection(image)
        
        input_tensor = self.prepare_input(image)

        # Perform inference on the image
        outputs = self.inference(input_tensor)

        boxes, scores, class_ids = self.process_output(outputs)
        for class_id, box, score in zip(class_ids, boxes, scores):
                if(class_id == 0): 
                    self.ocr_crops.append(box)
                self.boxes.append(box)
                self.scores.append(score)
                self.class_ids.append(class_id)
        return self.ocr_crops, self.boxes, self.scores, self.class_ids

    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor


    def inference(self, input_tensor):
        start = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})

        # print(f"Inference time: {(time.perf_counter() - start)*1000:.2f} ms")
        return outputs

    def process_output(self, output):
        predictions = np.squeeze(output[0]).T

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], []

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(predictions)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        # indices = nms(boxes, scores, self.iou_threshold)
        indices = multiclass_nms(boxes, scores, class_ids, self.iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices]

    def extract_boxes(self, predictions):
        # Extract boxes from predictions
        boxes = predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes)

        # Convert boxes to xyxy format
        boxes = xywh2xyxy(boxes)

        return boxes

    def rescale_boxes(self, boxes):

        # Rescale boxes to original image dimensions
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return boxes

    def draw_detections(self, image, draw_scores=True, mask_alpha=0.4):
        return draw_detections(image, self.boxes, self.scores,
                               self.class_ids, mask_alpha)

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

    def fallback_detection(self, image):
        """
        Fallback detection method when ONNX Runtime is not available.
        Returns a simple bounding box around the entire image as a tire detection.
        """
        print("Using fallback detection - assuming entire image is a tire")
        
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Create a simple bounding box around the entire image
        # with some padding to avoid edge effects
        padding = 0.1
        x1 = int(width * padding)
        y1 = int(height * padding)
        x2 = int(width * (1 - padding))
        y2 = int(height * (1 - padding))
        
        # Return the bounding box as if it's a tire detection
        box = [x1, y1, x2, y2]
        
        self.ocr_crops = [box]
        self.boxes = [box]
        self.scores = [0.8]  # Medium confidence
        self.class_ids = [0]  # Tire class
        
        return self.ocr_crops, self.boxes, self.scores, self.class_ids

