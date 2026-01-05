import streamlit as st
import cv2
import numpy as np
from CNN.predict_sign import predict_sign

def camera(learn):
    # Create two columns: one for video, one for predictions
    col_video, col_predictions = st.columns([2, 1])
    
    with col_video:
        FRAME_WINDOW = st.empty()
    
    with col_predictions:
        st.markdown("### Predictions")
        main_prediction = st.empty()
        st.markdown("#### Top 3 Alternatives")
        alt_predictions = st.empty()
    
    status_text = st.empty()

    if st.session_state.camera_running:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        status_text.info("📹 Camera running... Press 'Stop Camera' to exit.")

        while st.session_state.camera_running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                status_text.error("Failed to read from camera")
                break

            # --- Prediction ---
            res = predict_sign(frame, learn, "cam")

            pred_class = "N/A"
            confidence = 0
            hand_bbox = None
            hand_detected = False
            top_predictions = []

            if res:
                pred_class = res.get("prediction", "N/A")
                confidence = res.get("confidence", 0) * 100
                hand_detected = res.get("hand_detected", True)
                hand_bbox = res.get("hand_bbox", None)
                # Get top predictions if available
                top_predictions = res.get("top_predictions", [])

            # --- Gesture lock buffer (last 5 frames) ---
            buffer = st.session_state.gesture_buffer
            buffer.append((pred_class, confidence))
            if len(buffer) > 5:
                buffer.pop(0)

            # Take the most frequent prediction in buffer
            locked_pred = max(set([p[0] for p in buffer]), key=[p[0] for p in buffer].count)
            locked_conf = np.mean([p[1] for p in buffer])

            # --- Hand-centered framing ---
            if hand_bbox:
                x, y, w, h = hand_bbox
                pad = 30
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(frame.shape[1], x + w + pad)
                y2 = min(frame.shape[0], y + h + pad)
                frame_cropped = frame[y1:y2, x1:x2]
            else:
                frame_cropped = frame

            # Mirror for user
            frame_display = cv2.flip(frame_cropped, 1)

            # --- Draw overlay ---
            color = (
                (0, 255, 0) if locked_conf >= 90
                else (0, 165, 255) if locked_conf >= 75
                else (255, 0, 0)
            )
            status_icon = "✓" if hand_detected else "⚠"
            cv2.putText(frame_display,
                        f"{status_icon} {locked_pred} ({locked_conf:.1f}%)",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # Confidence bar
            bar_width = int(frame_display.shape[1] * locked_conf / 100)
            cv2.rectangle(frame_display, (0, frame_display.shape[0]-20),
                          (bar_width, frame_display.shape[0]-10),
                          color, -1)
            cv2.rectangle(frame_display, (0, frame_display.shape[0]-20),
                          (frame_display.shape[1], frame_display.shape[0]-10),
                          (200, 200, 200), 2)

            FRAME_WINDOW.image(cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB),
                               channels="RGB",
                               use_container_width=True)

            # --- Update predictions display ---
            # Main prediction
            conf_color = "green" if locked_conf >= 90 else "orange" if locked_conf >= 75 else "red"
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

            # Top 3 alternatives
            if top_predictions and len(top_predictions) > 1:
                alt_html = ""
                for i, (label, conf) in enumerate(top_predictions[1:4], 1):  # Skip first (main), take next 3
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
                        <span style="font-weight: 600; font-size: 1.1rem;">{i}. {label}</span>
                        <span style="color: #666; font-size: 0.95rem;">{conf * 100:.1f}%</span>
                    </div>
                    """
                alt_predictions.markdown(alt_html, unsafe_allow_html=True)
            else:
                alt_predictions.markdown("*No alternative predictions available*")

        cap.release()
        st.session_state.camera_running = False
    else:
        status_text.success("Click 'Start Camera' to begin.")