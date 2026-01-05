import streamlit as st
import cv2
import numpy as np
from HOG_SVM.predict_sign import predict_sign_hog


def camera(hog_bundle):
    """
    Real-time HOG + LinearSVM webcam inference
    """

    # =========================
    # Unpack bundle
    # =========================
    svm_clf = hog_bundle["svm"]
    scaler = hog_bundle["scaler"]
    selector = hog_bundle["selector"]
    inv_label_map = hog_bundle["inv_label_map"]
    hog_params = hog_bundle["hog_params"]

    # =========================
    # Session state
    # =========================
    if "gesture_buffer" not in st.session_state:
        st.session_state.gesture_buffer = []

    # =========================
    # Layout
    # =========================
    col_video, col_predictions = st.columns([2, 1])

    with col_video:
        FRAME_WINDOW = st.empty()

    with col_predictions:
        st.markdown("### Prediction")
        main_prediction = st.empty()
        st.markdown("#### Top 3 Alternatives")
        alt_predictions = st.empty()

    status_text = st.empty()

    # =========================
    # Camera loop
    # =========================
    if not st.session_state.camera_running:
        status_text.success("Click **Start Camera** to begin.")
        return

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    status_text.info("📹 Camera running — show a hand to get predictions")

    while st.session_state.camera_running and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            status_text.error("Failed to read from camera")
            break

        # -------------------------
        # Prediction
        # -------------------------
        res = predict_sign_hog(
            frame,
            svm_clf,
            scaler,
            selector,
            inv_label_map,
            hog_params,
            mode="cam"
        )

        hand_detected = res and res["hand_detected"]

        # -------------------------
        # NO HAND → NO PREDICTION
        # -------------------------
        if not hand_detected:
            st.session_state.gesture_buffer.clear()

            frame_display = cv2.flip(frame, 1)
            cv2.putText(
                frame_display,
                "⚠ No hand detected",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

            FRAME_WINDOW.image(
                cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB),
                channels="RGB",
                use_container_width=True
            )

            main_prediction.markdown(
                "<div style='color:#dc2626; font-weight:600;'>No hand detected</div>",
                unsafe_allow_html=True
            )
            alt_predictions.markdown("*—*")

            continue

        # -------------------------
        # VALID HAND PREDICTION
        # -------------------------
        pred_class = res["prediction"]
        margin = res["svm_margin"]
        top3 = res["top3"]

        # Margin → UI confidence (visual only)
        pseudo_conf = 1 / (1 + np.exp(-margin))
        pseudo_conf *= 100

        # -------------------------
        # Gesture lock buffer
        # -------------------------
        buffer = st.session_state.gesture_buffer
        buffer.append((pred_class, pseudo_conf))
        if len(buffer) > 5:
            buffer.pop(0)

        locked_pred = max(
            set(p[0] for p in buffer),
            key=[p[0] for p in buffer].count
        )
        locked_conf = np.mean([p[1] for p in buffer])

        # -------------------------
        # Frame display
        # -------------------------
        frame_display = cv2.flip(frame, 1)

        color = (
            (0, 255, 0) if locked_conf >= 85
            else (0, 165, 255) if locked_conf >= 65
            else (255, 0, 0)
        )

        cv2.putText(
            frame_display,
            f"✓ {locked_pred} ({locked_conf:.1f}%)",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

        # Confidence bar
        bar_width = int(frame_display.shape[1] * locked_conf / 100)
        cv2.rectangle(
            frame_display,
            (0, frame_display.shape[0] - 18),
            (bar_width, frame_display.shape[0] - 8),
            color,
            -1
        )

        FRAME_WINDOW.image(
            cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB),
            channels="RGB",
            use_container_width=True
        )

        # -------------------------
        # Prediction card
        # -------------------------
        conf_color = (
            "green" if locked_conf >= 85
            else "orange" if locked_conf >= 65
            else "red"
        )

        main_prediction.markdown(f"""
        <div style="
            border: 3px solid {conf_color};
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            background-color: #f8f9fa;
            margin-bottom: 20px;
        ">
            <h1 style="margin: 0; color: #333;">{locked_pred}</h1>
                <h3 style="margin: 10px 0 0 0; color: {conf_color};">{locked_conf:.1f}%</h3>
        </div>
        """, unsafe_allow_html=True)

        # -------------------------
        # Top-3 (decision margins)
        # -------------------------
        alt_html = ""
        for i, (label, score) in enumerate(top3[1:], 1):
            conf = 1 / (1 + np.exp(-score)) * 100
            alt_html += f"""
            <div style="
                        border: 1px solid #ddd;
                        border-radius: 8px;
                        padding: 12px;
                        margin-bottom: 10px;
                        background-color: white;
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                    ">
                <span style=" color: #666; font-weight: 600; font-size: 1.1rem;" ><b>{i}. {label}</b></span>
                <span style="color: #666; font-size: 0.95rem;">{conf:.1f}%</span>
            </div>
            """

        alt_predictions.markdown(alt_html or "*—*", unsafe_allow_html=True)

    cap.release()
    st.session_state.camera_running = False
