# -----------------------------
# MediaPipe hand crop function
# -----------------------------
import mediapipe as mp
from pathlib import Path
import cv2

def crop_hand(image):
    """
    Detect and crop hand from image.
    Accepts either:
        - image path (str / Path)
        - OpenCV image array (H, W, 3)
    Returns cropped RGB image and hand_detected flag
    """
    # Read image if path
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image))
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image}")
    else:
        # Assume image is numpy array
        img = image.copy()
        if img is None or img.size == 0:
            raise ValueError("Empty image array provided")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(static_image_mode=True,
                        max_num_hands=1,
                        min_detection_confidence=0.5) as hands:
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            h, w, _ = img.shape
            x_min, y_min = w, h
            x_max, y_max = 0, 0

            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    x_min, y_min = min(x_min, x), min(y_min, y)
                    x_max, y_max = max(x_max, x), max(y_max, y)

            # Add padding
            pad = 20
            x_min, y_min = max(0, x_min - pad), max(0, y_min - pad)
            x_max, y_max = min(w, x_max + pad), min(h, y_max + pad)

            cropped = img[y_min:y_max, x_min:x_max]
            cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            return cropped_rgb, True
        else:
            return img_rgb, False

