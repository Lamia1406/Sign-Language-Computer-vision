# import streamlit as st
# from pathlib import Path
# import cv2
# from CNN.components.prediction_card import prediction_card
# from HOG_SVM.predict_sign import predict_sign_hog   # your HOG-style predict_sign
# import numpy as np

# def get_top3_string_hog(outputs, label_map):
#     if outputs is None:
#         return "N/A"

#     outputs = np.asarray(outputs).ravel()  # 🔥 FIX

#     top3_idx = outputs.argsort()[-3:][::-1]

#     return ", ".join(
#         f"{label_map[int(i)]} ({outputs[int(i)]:.1%})"
#         for i in top3_idx
#     )


# def hog_batch_prediction(hog_bundle, folder_path="example_signs"):
#     svm = hog_bundle["svm"]
#     scaler = hog_bundle["scaler"]
#     selector = hog_bundle["selector"]
#     inv_label_map = hog_bundle["inv_label_map"]
#     hog_params = hog_bundle["hog_params"]
#     # ============================
#     # Paths & validation
#     # ============================
#     folder = Path(folder_path)

#     if not folder.exists():
#         st.error(f"Folder '{folder_path}' not found.")
#         return

#     image_files = [
#         f for f in folder.iterdir()
#         if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
#     ]

#     if not image_files:
#         st.warning("No images found.")
#         return

#     # ============================
#     # Batch prediction
#     # ============================
#     results = []
#     correct_count = 0

#     for img_path in image_files:
#         img_bgr = cv2.imread(str(img_path))

#         if img_bgr is None:
#             continue

#         res = predict_sign_hog(
#             img_bgr,
#             svm_clf=svm,
#             scaler=scaler,
#             selector=selector,
#             inv_label_map=inv_label_map,
#             hog_params=hog_params
#         )
#         if res and res["hand_detected"]:
#             pred = res["prediction"]
#             conf = res["confidence"]
#             top3 = get_top3_string_hog(res["all_probs"], inv_label_map)

#             true_label = img_path.stem.split("_")[0].lower()
#             is_correct = pred.lower() == true_label

#             if is_correct:
#                 correct_count += 1
#         else:
#             pred, conf, top3 = "NO HAND", 0.0, "N/A"
#             is_correct = False

#         results.append({
#             "path": img_path,
#             "Prediction": pred,
#             "Confidence": f"{conf:.2%}",
#             "Top-3": top3,
#             "is_correct": is_correct,
#             "cropped_image": res["cropped_image"] if res else None
#         })

#     # ============================
#     # Accuracy
#     # ============================
#     acc = correct_count / len(image_files) * 100

#     # ============================
#     # Header row (same as CNN)
#     # ============================
#     st.html(f"""
# <div style="
#     display: flex;
#     align-items: center;
#     justify-content: space-between;
#     margin-bottom: 1rem;
#     flex-wrap: wrap;
#     gap: 1rem;
# ">
#     <h3 style="margin: 0;">Batch Prediction Results (HOG)</h3>

#     <div style="
#         border: 1px solid #e0e0e0;
#         border-radius: 10px;
#         padding: 0.75rem 1.25rem;
#         background-color: #ffffff;
#         display: flex;
#         align-items: center;
#         gap: 0.75rem;
#         box-shadow: 0 2px 6px rgba(0,0,0,0.05);
#     ">
#         <div style="
#             display: flex;
#             align-items: center;
#             gap: 1rem;
#         ">
#             <div style="font-weight: 600; font-size: 0.95rem; color: #333;">
#                 Overall Accuracy
#             </div>
#             <div style="font-size: 1.4rem; font-weight: 700; color: #00ff33;">
#                 {acc:.2f}%
#             </div>
#         </div>
#     </div>
# </div>
# """)

#     st.divider()

#     # ============================
#     # Display predictions
#     # ============================
#     for result in results:
#         prediction_card(result)


import streamlit as st
from pathlib import Path
import cv2
import numpy as np

from HOG_SVM.components.prediction_card import prediction_card
from HOG_SVM.predict_sign import predict_sign_hog


def hog_batch_prediction(hog_bundle, folder_path="example_signs"):
    # ============================
    # Unpack model bundle
    # ============================
    svm = hog_bundle["svm"]
    scaler = hog_bundle["scaler"]
    selector = hog_bundle["selector"]
    inv_label_map = hog_bundle["inv_label_map"]
    hog_params = hog_bundle["hog_params"]

    # ============================
    # Validate folder
    # ============================
    folder = Path(folder_path)

    if not folder.exists():
        st.error(f"Folder '{folder_path}' not found.")
        return

    image_files = [
        f for f in folder.iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    ]

    if not image_files:
        st.warning("No images found.")
        return

    # ============================
    # Batch prediction
    # ============================
    results = []
    correct_count = 0

    for img_path in image_files:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        res = predict_sign_hog(
            img_bgr,
            svm_clf=svm,
            scaler=scaler,
            selector=selector,
            inv_label_map=inv_label_map,
            hog_params=hog_params,
            mode="batch"
        )

        # ----------------------------
        # NO HAND DETECTED
        # ----------------------------
        if res is None or not res["hand_detected"]:
            results.append({
                "path": img_path,
                "Prediction": "NO HAND",
                "SVM Margin": None,
                "Top-3 (Margins)": "N/A",
                "is_correct": False,
                "cropped_image": None,
                "model_type": "hog+svm"
            })
            continue

        # ----------------------------
        # VALID PREDICTION
        # ----------------------------
        pred = res["prediction"]
        margin = res["svm_margin"]

        top3_str = ", ".join(
            f"{lbl} ({score:.2f})"
            for lbl, score in res["top3"]
        )

        true_label = img_path.stem.split("_")[0].lower()
        is_correct = pred.lower() == true_label

        if is_correct:
            correct_count += 1

        results.append({
            "path": img_path,
            "Prediction": pred,
            "SVM Margin": margin,
            "Top-3 (Margins)": top3_str,
            "is_correct": is_correct,
            "cropped_image": res["cropped_image"],
            "model_type": "hog+svm"
        })

    # ============================
    # Accuracy
    # ============================
    acc = (correct_count / len(image_files)) * 100

    # ============================
    # Header
    # ============================
    st.html(f"""
<div style="
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1rem;
    flex-wrap: wrap;
    gap: 1rem;
">
    <h3 style="margin: 0;">Batch Prediction Results</h3>

    <div style="
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 0.75rem 1.25rem;
        background-color: #ffffff;
        display: flex;
        align-items: center;
            
        gap: 0.75rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    ">
        <div style="
            display: flex;
    align-items: center;
            gap: 1rem;
            ">
            <div style="font-weight: 600; font-size: 0.95rem; color: #333;">
                Overall Accuracy
            </div>
            <div style="font-size: 1.4rem; font-weight: 700; color: #00ff33;">
                {acc:.2f}%
            </div>
        </div>
    </div>
</div>
""")

    st.divider()

    # ============================
    # Display results
    # ============================
    for result in results:
        prediction_card(result)
