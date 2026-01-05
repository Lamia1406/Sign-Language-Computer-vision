# =====================================================
# Batch Prediction with Hand Cropping + HOG + LinearSVC
# =====================================================

import cv2
import pickle
from pathlib import Path
import numpy as np
import mediapipe as mp
from skimage.feature import hog

# -----------------------------
# Load trained model
# -----------------------------
MODEL_FILE = "svm_hog_selected.pkl"

with open(MODEL_FILE, "rb") as f:
    data = pickle.load(f)

svm_clf = data["model"]
scaler = data["scaler"]
selector = data["selector"]
label_map = data["label_map"]
inv_label_map = {v: k for k, v in label_map.items()}
hog_params = data["hog_params"]

# -----------------------------
# MediaPipe Hands detector
# -----------------------------
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

# -----------------------------
# Hand cropping function
# -----------------------------
def crop_hand(img, padding=20):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(img_rgb)
    if not results.multi_hand_landmarks:
        return img_rgb, False

    h, w, _ = img.shape
    x_min, y_min = w, h
    x_max, y_max = 0, 0
    for lm in results.multi_hand_landmarks[0].landmark:
        x, y = int(lm.x * w), int(lm.y * h)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x)
        y_max = max(y_max, y)

    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(w, x_max + padding)
    y_max = min(h, y_max + padding)

    cropped = img[y_min:y_max, x_min:x_max]
    if cropped.size == 0:
        return img_rgb, False

    return cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB), True

# -----------------------------
# HOG feature extraction
# -----------------------------
def extract_hog_features(img_rgb, size=(64,64)):
    img_gray = cv2.cvtColor(cv2.resize(img_rgb, size), cv2.COLOR_RGB2GRAY)
    features = hog(img_gray, **hog_params)
    return features

# -----------------------------
# Predict single image
# -----------------------------
def predict_image(img_path):
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"⚠ Failed to read {img_path}")
        return None

    cropped, hand_detected = crop_hand(img_bgr)
    if not hand_detected:
        print(f"⚠ No hand detected in {img_path}")
        return None

    feat = extract_hog_features(cropped)
    feat_scaled = scaler.transform([feat])
    feat_sel = selector.transform(feat_scaled)

    pred_id = svm_clf.predict(feat_sel)[0]
    pred_letter = inv_label_map[pred_id]
    return pred_letter, cropped

# -----------------------------
# Batch prediction
# -----------------------------
def batch_predict(folder_path="example_signs"):
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Folder not found: {folder_path}")
        return

    images = [f for f in folder.iterdir() if f.suffix.lower() in {".jpg",".jpeg",".png",".bmp"}]
    if not images:
        print("No images found.")
        return

    correct = 0
    for img_path in images:
        true_label = img_path.stem.split("_")[0].lower()
        result = predict_image(img_path)
        if result is None:
            continue

        pred_letter, cropped_img = result
        is_correct = pred_letter.lower() == true_label
        if is_correct:
            correct += 1

        # Show cropped hand
        cv2.imshow(f"{img_path.name} - Pred: {pred_letter}", cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(500)  # 500ms per image
        cv2.destroyAllWindows()

        print(f"{img_path.name}: True={true_label}, Pred={pred_letter}, Correct={is_correct}")

    acc = correct / len(images) * 100
    print(f"\nBatch Accuracy: {acc:.2f}%")

# -----------------------------
# Run batch prediction
# -----------------------------
if __name__ == "__main__":
    batch_predict("example_signs")
