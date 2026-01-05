from pathlib import Path
import cv2
import numpy as np
import time

# ===========================================
# YOLOv8 HAND DETECTOR
# ===========================================
class YOLOv8HandDetector:
    def __init__(self, model_path="hand_detection_yolo/yolo11n.pt"):
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.available = True
            print(f"✓ YOLOv8 Hand Detection loaded: {model_path}")
        except Exception as e:
            self.available = False
            print(f"✗ YOLOv8 Hand Detection not available: {e}")
    
    def detect(self, image_input):
        """
        image_input: either file path (str/Path) or numpy array
        """
        if not self.available:
            return None

        # --- accept path or array ---
        if isinstance(image_input, (str, Path)):
            img = cv2.imread(str(image_input))
            if img is None:
                raise FileNotFoundError(f"Could not read image: {image_input}")
        else:
            img = image_input.copy()
            if img is None or img.size == 0:
                raise ValueError("Empty image array provided")

        start_time = time.time()
        results = self.model(img, conf=0.25, verbose=False)
        inference_time = time.time() - start_time

        hand_boxes = []
        output_img = img.copy()
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())

                    cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(output_img, f"Hand: {conf:.2f}", 
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.6, (0, 255, 255), 2)

                    hand_boxes.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf
                    })


     

        return {
            'model': 'YOLOv8 Hand Detection',
            'num_hands_detected': len(hand_boxes),
            'hands': hand_boxes,
            'inference_time': inference_time,
        }

# ===========================================
# CROPPING FUNCTION
# ===========================================
detector = YOLOv8HandDetector("hand_detection_yolo/yolo11n.pt")

def crop_hand(image, detector=detector, pad=20, output_path=None):
    """
    Detect and crop hand from image using YOLO.
    Returns cropped BGR image and hand_detected flag.

    Picks the hand with the highest confidence if multiple hands are detected.
    """
    if detector is None:
        raise ValueError("A YOLO hand detector instance is required.")

    # Load image if path
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image))
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image}")
    else:
        img = image.copy()
        if img is None or img.size == 0:
            raise ValueError("Empty image array provided")

    result = detector.detect(img)

    if result and result['num_hands_detected'] > 0:
        # Pick the hand with the highest confidence
        best_hand = max(result['hands'], key=lambda h: h['confidence'])
        x1, y1, x2, y2 = best_hand['bbox']

        # Add padding & clamp to image dimensions
        h, w, _ = img.shape
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)

        cropped = img[y1:y2, x1:x2]

        if output_path:
            cv2.imwrite(str(output_path), cropped)

        return cropped, True
    else:
        # No hand detected → return original image
        return img, False
