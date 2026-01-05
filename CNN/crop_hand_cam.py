import mediapipe as mp
import cv2
from pathlib import Path
import numpy as np

def crop_hand_cam(image, min_detection_confidence=0.3, pad=20):
    """
    Detect and crop hand from image.
    Returns cropped RGB image and hand_detected flag.
    Accepts:
        - OpenCV image array (H, W, 3)
        - Image path (str / Path)
    """
    # Load image if path
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image))
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image}")
    else:
        img = image.copy()
        if img is None or img.size == 0:
            raise ValueError("Empty image array provided")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    mp_hands = mp.solutions.hands
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence
    ) as hands:

        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            # Get bounding box of hand
            x_min, y_min = w, h
            x_max, y_max = 0, 0

            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    x_min, y_min = min(x_min, x), min(y_min, y)
                    x_max, y_max = max(x_max, x), max(y_max, y)

            # Add padding & clamp
            x_min, y_min = max(0, x_min - pad), max(0, y_min - pad)
            x_max, y_max = min(w, x_max + pad), min(h, y_max + pad)

            cropped = img[y_min:y_max, x_min:x_max]
            cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

            # Debug: make sure crop is not empty
            if cropped_rgb.size == 0:
                return img_rgb, False

            return cropped_rgb, True

        else:
            # No hand detected → return full image
            return img_rgb, False