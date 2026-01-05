import streamlit as st
import cv2
import numpy as np
from pathlib import Path


# =============================
# PREDICTION CARD (HOG ONLY)
# =============================
def prediction_card(result):
    status = "Correct" if result["is_correct"] else "Incorrect"
    status_color = "#16a34a" if result["is_correct"] else "#dc2626"
    status_bg = "#dcfce7" if result["is_correct"] else "#fee2e2"
    status_icon = "✓" if result["is_correct"] else "✕"

    if result["SVM Margin"] is None:
        margin_html = "—"
    else:
        margin_html = f"{result['SVM Margin']:.2f}"
    with st.container():
        col_img, col_content = st.columns([1.2, 3], gap="medium")

        with col_img:
            st.image(result["path"], width=190)

        with col_content:
            st.html(f"""
            <div style="
                padding: 0.85rem 1rem;
                border: 1px solid #e5e7eb;
                border-radius: 14px;
                background: #ffffff;
                box-shadow: 0 2px 10px rgba(0,0,0,0.05);
                display: flex;
                flex-direction: column;
                gap: 0.4rem;
            ">

                <div style="
                    align-self: flex-start;
                    background:{status_bg};
                    color:{status_color};
                    padding:0.25rem 0.65rem;
                    border-radius:999px;
                    font-size:0.8rem;
                    font-weight:600;
                ">
                    {status_icon} {status}
                </div>
                <div style="font-size:1.05rem; font-weight:600; color:#111827;">
                    Prediction:
                    <span style="color:#2563eb;">
                        {result['Prediction']}
                    </span>
                </div>
               <!-- Confidence -->
                <div style="font-size:0.9rem; color:#374151;">
                    SVM margin:
                    <strong>{margin_html}</strong>
                </div>


                <div style="font-size:0.85rem; color:#6b7280;">
                    Top-3 margins:
                    <span>{result['Top-3 (Margins)']}</span>
                </div>

            </div>
            """)

        st.divider()
