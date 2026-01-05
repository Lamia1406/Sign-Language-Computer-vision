from CNN.components.styled_radio_container import styled_radio_container
import streamlit as st
from fastai.vision.all import load_learner

import platform
import pathlib

from CNN.webcam import webcam
from CNN.batch_prediction import batch_prediction
from HOG_SVM.hog_batch_prediction import hog_batch_prediction
from HOG_SVM.hog_webcam import hog_webcam
from HOG_SVM.load_hog_model import load_hog_model

import sys

st.write("Python version:", sys.version)
# -----------------------------
# Fix pathlib on Windows
# -----------------------------
if platform.system() == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath

# -----------------------------
# Page config with light theme
# -----------------------------
st.set_page_config(
    page_title="Arabic Sign Language Recognition",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for styling radio buttons
st.markdown("""
    <style>
       
        
        .stRadio > div > label {
            padding: 0.75rem 1rem;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s;
            width: 400px;
            
        }
        
        .stRadio > div > label:hover {
            border-color: #0066cc;
            background-color: #ff3300;
            color: red;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Arabic Sign Language Recognition")

# -----------------------------
# Load CNN model
# -----------------------------
MODEL_FILE = "CNN/weights/arabic_sign_resnet342.pkl"

@st.cache_resource
def load_model(path):
    return load_learner(path, cpu=True)

learn = load_model(MODEL_FILE)

col1, col2 = st.columns(2)

# -----------------------------
# Input Mode (Batch / Webcam)
# -----------------------------
folder_icon_svg = """
<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#00ccff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <path stroke="none" d="M0 0h24v24H0z" fill="none"/>
    <path d="M5 19l2.757 -7.351a1 1 0 0 1 .936 -.649h12.307a1 1 0 0 1 .986 1.164l-.996 5.211a2 2 0 0 1 -1.964 1.625h-14.026a2 2 0 0 1 -2 -2v-11a2 2 0 0 1 2 -2h4l3 3h7a2 2 0 0 1 2 2v2" />
</svg>
"""
robot_icon_svg ="""
<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#00ccff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="icon icon-tabler icons-tabler-outline icon-tabler-robot-face">
<path stroke="none" d="M0 0h24v24H0z" fill="none"/>
<path d="M6 5h12a2 2 0 0 1 2 2v12a2 2 0 0 1 -2 2h-12a2 2 0 0 1 -2 -2v-12a2 2 0 0 1 2 -2z" /><path d="M9 16c1 .667 2 1 3 1s2 -.333 3 -1" /><path d="M9 7l-1 -4" /><path d="M15 7l1 -4" /><path d="M9 12v-1" /><path d="M15 12v-1" /></svg>
"""
with col1:
    input_mode = styled_radio_container(
        title="Input Mode",
        icon_svg=folder_icon_svg,
        options=["Batch Prediction", "Webcam"],
        key="input_mode",
    )

# -----------------------------
# Model Mode (CNN / HOG+SVM)
# -----------------------------
with col2:
    model_mode = styled_radio_container(
        title="Model",
        icon_svg=robot_icon_svg,
        options=["CNN", "HOG + SVM"],
        key="model_mode",
    )

st.divider()

# ======================================================
# 🔀 ROUTER
# ======================================================

# Normalize the input_mode to match original logic
input_mode_value = "Batch" if input_mode == "Batch Prediction" else "Webcam"
model_mode_value = "CNN" if model_mode == "CNN" else "HOG"

if model_mode_value == "CNN":

    if input_mode_value == "Batch":
        batch_prediction(learn)

    elif input_mode_value == "Webcam":
        webcam(learn)

elif model_mode_value == "HOG":

    hog_bundle = load_hog_model()
    if input_mode_value == "Batch":
        hog_batch_prediction(hog_bundle)

    else:
        hog_webcam(hog_bundle)

