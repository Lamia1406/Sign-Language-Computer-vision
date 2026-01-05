import streamlit as st
import numpy as np
from CNN.crop_hand import crop_hand
from CNN.crop_hand_cam import crop_hand_cam
from HOG_SVM.hog_features import extract_hog_features
def predict_sign_hog(
    image,
    svm_clf,
    scaler,
    selector,
    inv_label_map,
    hog_params,
    mode="batch"
):
    """
    Predict sign language using HOG + LinearSVC
    Returns SVM-specific decision information (NOT probabilities)
    """
    try:
        # 1️⃣ Crop hand
        if mode == "batch":
            cropped_img, hand_detected = crop_hand(image)
        else:
            cropped_img, hand_detected = crop_hand_cam(image)

        if not hand_detected:
            return {
                'prediction': None,
                'svm_margin': None,
                'top3': None,
                'decision_scores': None,
                'cropped_image': cropped_img,
                'hand_detected': False,
                'model_type': 'hog+svm'
            }

        # 2️⃣ HOG features
        feat = extract_hog_features(cropped_img, hog_params)
        feat_scaled = scaler.transform([feat])
        feat_sel = selector.transform(feat_scaled)

        # 3️⃣ Decision function (CORE of LinearSVC)
        scores = svm_clf.decision_function(feat_sel).flatten()

        pred_idx = np.argmax(scores)
        pred_class = inv_label_map[pred_idx]
        margin = float(scores[pred_idx])

        # 4️⃣ Top-3 predictions
        top3_idx = scores.argsort()[-3:][::-1]
        top3 = [(inv_label_map[i], float(scores[i])) for i in top3_idx]

        return {
            'prediction': pred_class,
            'svm_margin': margin,              # ← NOT probability
            'top3': top3,
            'decision_scores': scores,
            'cropped_image': cropped_img,
            'hand_detected': True,
            'model_type': 'hog+svm'
        }

    except Exception as e:
        st.error(f"Error during HOG prediction: {e}")
        return None
