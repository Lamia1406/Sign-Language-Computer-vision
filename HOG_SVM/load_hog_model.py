import pickle
import streamlit as st

HOG_MODEL_FILE = "HOG_SVM/weights/svm_hog_selected.pkl"


@st.cache_resource
def load_hog_model():
    with open(HOG_MODEL_FILE, "rb") as f:
        data = pickle.load(f)

    return {
        "svm": data["model"],
        "scaler": data["scaler"],
        "selector": data["selector"],
        "label_map": data["label_map"],
        "inv_label_map": {v: k for k, v in data["label_map"].items()},
        "hog_params": data["hog_params"]
        
    }
