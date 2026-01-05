import streamlit as st
from pathlib import Path
from CNN.predict_sign import predict_sign
from CNN.components.prediction_card import prediction_card


def get_top3_string(outputs, learn):
    """Return top-3 predictions as a string."""
    top3_idx = outputs.argsort(descending=True)[:3]
    return ", ".join(
        f"{learn.dls.vocab[i]} ({outputs[i]:.1%})"
        for i in top3_idx
    )


def batch_prediction(learn):

    # ============================
    # Paths & validation
    # ============================
    folder = Path("example_signs")

    if not folder.exists():
        st.error("Folder 'example_signs' not found.")
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
        res = predict_sign(img_path, learn)

        if res:
            pred = res["prediction"]
            conf = res["confidence"]
            top3 = get_top3_string(res["all_probs"], learn)
            true_label = img_path.stem.split("_")[0].lower()
            is_correct = true_label == pred.lower()
            if is_correct:
                correct_count += 1
        else:
            pred, conf, top3 = "ERROR", 0, "N/A"
            is_correct = False

        results.append({
            "path": img_path,
            "Prediction": pred,
            "Confidence": f"{conf:.2%}",
            "Top-3": top3,
            "is_correct": is_correct
        })

    # ============================
    # Accuracy
    # ============================
    acc = correct_count / len(image_files) * 100

    

    # ============================
    # Header row (space-between)
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
    # Display predictions
    # ============================
    for idx, result in enumerate(results):
        prediction_card(result)