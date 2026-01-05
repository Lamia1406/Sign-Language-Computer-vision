
from CNN.crop_hand import crop_hand
from CNN.crop_hand_cam import crop_hand_cam
from fastai.vision.all import PILImage
import streamlit as st


def predict_sign(image, learn, mode="batch"):
    """Predict sign language from image or frame"""
    try:
        if mode == "batch":
            cropped_img, hand_detected = crop_hand(image)
        if mode == "cam":
            cropped_img, hand_detected = crop_hand_cam(image)
        
        pil_img = PILImage.create(cropped_img)
        pred_class, pred_idx, outputs = learn.predict(pil_img)
        return {
            'prediction': pred_class,
            'confidence': float(outputs[pred_idx]),
            'all_probs': outputs,
            'cropped_image': cropped_img,
            'hand_detected': hand_detected
        }
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None
