import streamlit as st

from core.models import SAMPLE_DIR, load_pneumonia_model, require_models
from core.xray import load_xray, overlay, predict_with_cam

st.title("🫁 Pneumonia screening from chest X-rays")
st.caption("EfficientNet-B0 CNN · pediatric chest X-rays (ages 1–5) · Grad-CAM explanation")

require_models("pneumonia")
model, default_threshold = load_pneumonia_model()


@st.cache_data(show_spinner="Analysing X-ray…", max_entries=64)
def analyse(data: bytes):
    m, _ = load_pneumonia_model()
    img = load_xray(data)
    prob, cam = predict_with_cam(m, img)
    return img, prob, cam


with st.sidebar:
    threshold = st.slider(
        "Decision threshold", 0.0, 1.0, round(default_threshold, 2), 0.01,
        help=f"Default {default_threshold:.2f} was chosen on validation data to keep sensitivity ≥ 95 % "
             "(few missed pneumonias, at the cost of more false alarms).",
    )
    alpha = st.slider("Heatmap opacity", 0.0, 1.0, 0.4, 0.05)

source = st.radio("Input", ["Sample from the test set", "Upload an X-ray"], horizontal=True)
truth = None
if source.startswith("Sample"):
    samples = sorted((SAMPLE_DIR / "xray").glob("*.jpg"))
    if not samples:
        st.warning("No sample images found — run `python scripts/prepare_samples.py`.")
        st.stop()
    path = st.selectbox("Sample image", samples, format_func=lambda p: p.stem.replace("_", " #").title())
    data = path.read_bytes()
    truth = path.stem.split("_")[0].upper()
else:
    upload = st.file_uploader("Frontal (AP/PA) chest X-ray — JPEG or PNG", type=["jpg", "jpeg", "png"])
    if upload is None:
        st.info("Upload an image to analyse it.")
        st.stop()
    data = upload.getvalue()

img, prob, cam = analyse(data)
flagged = prob >= threshold

left, right = st.columns(2)
with left:
    st.image(data, caption="Input X-ray", width="stretch")
with right:
    st.image(overlay(img, cam, alpha), caption="Grad-CAM — regions driving the pneumonia score",
             width="stretch")

m1, m2, m3 = st.columns(3)
m1.metric("Pneumonia probability", f"{prob:.1%}")
m2.metric("Triage decision", "⚠️ PNEUMONIA suspected" if flagged else "✅ No pneumonia flagged")
if truth:
    m3.metric("Ground truth (expert grading)", truth,
              delta="model agrees" if (truth == "PNEUMONIA") == flagged else "model disagrees",
              delta_color="normal" if (truth == "PNEUMONIA") == flagged else "inverse")
st.progress(prob, text=f"score {prob:.3f} vs threshold {threshold:.2f}")

if flagged:
    st.warning("Flagged for **priority radiologist review**. The screening threshold favours sensitivity, "
               "so some flagged X-rays will be normal.")
else:
    st.success("Below the screening threshold. Routine reading order.")

with st.expander("How to read this"):
    st.markdown(
        """
* The **probability** is the model's estimated chance of pneumonia (bacterial or viral).
* The **threshold** turns it into a decision. Lower it to catch more pneumonia (more false alarms); raise it for fewer false alarms (more misses).
* **Grad-CAM** highlights the image regions that pushed the score up. Trustworthy heatmaps cover the lung fields;
  heatmaps on borders, text markers or outside the body suggest the model is using a shortcut.
* Trained only on pediatric AP X-rays — adult or lateral images are out of distribution.
"""
    )
