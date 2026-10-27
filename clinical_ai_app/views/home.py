import streamlit as st

from core.models import MODEL_FILES, missing_files

st.title("🩺 Clinical AI Lab")
st.markdown(
    "Three deep-learning models for clinical diagnostics, trained on public, expert-labelled datasets. "
    "Pick a tool in the sidebar."
)

c1, c2, c3 = st.columns(3)
with c1, st.container(border=True):
    st.subheader("🫁 Pneumonia X-ray")
    st.markdown(
        "**CNN** — a convolutional neural network trained on 5,856 pediatric chest X-rays (Kermany et al., *Cell* 2018). "
        "Flags likely pneumonia at a sensitivity-first triage threshold and shows a **Grad-CAM** heatmap."
    )
    st.page_link("views/xray.py", label="Open X-ray screening", icon="➡️")
with c2, st.container(border=True):
    st.subheader("💓 ECG arrhythmia")
    st.markdown(
        "**RNN** — Conv1D + bidirectional LSTMs over each heartbeat plus RR-interval rhythm features, trained on the "
        "MIT-BIH Arrhythmia Database and tested on unseen patients. Labels every beat N / S / V / F (AAMI)."
    )
    st.page_link("views/ecg.py", label="Open ECG analysis", icon="➡️")
with c3, st.container(border=True):
    st.subheader("🧪 Synthetic X-rays")
    st.markdown(
        "**Conditional GAN** — projection discriminator, spectral norm, hinge loss, DiffAugment and an EMA generator. "
        "Generates NORMAL or PNEUMONIA chest X-rays on demand for data augmentation research."
    )
    st.page_link("views/gan.py", label="Open X-ray generator", icon="➡️")

st.divider()
st.subheader("Model status")
names = {"pneumonia": "Pneumonia CNN", "ecg": "ECG CNN-BiLSTM", "gan": "X-ray GAN generator"}
for key, (files, notebook) in MODEL_FILES.items():
    missing = missing_files(key)
    if missing:
        st.markdown(f"❌ **{names[key]}** — missing {', '.join(f'`{m}`' for m in missing)} (train with `{notebook}`)")
    else:
        st.markdown(f"✅ **{names[key]}** — loaded from `models/`")

with st.expander("About the data & limitations"):
    st.markdown(
        """
* **Chest X-rays:** children aged 1–5 from a single hospital (Guangzhou). Performance on adults or other hospitals is unknown.
* **ECG:** MIT-BIH, 48 half-hour two-lead recordings at 360 Hz. The model uses lead MLII; other leads or devices may behave differently.
  Supraventricular (S) beats remain hard to detect on new patients.
* **GAN:** 64 × 64 images — far below diagnostic resolution. Synthetic images are for research, never for clinical use.
* All demo inputs bundled with this app come from the **held-out test sets**, so the models have never seen them.
"""
    )
