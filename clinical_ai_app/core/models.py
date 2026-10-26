"""Model loading (cached once per server process) and model-file checks."""
import json
from pathlib import Path

import numpy as np
import streamlit as st

APP_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = APP_DIR / "models"
SAMPLE_DIR = APP_DIR / "sample_data"

MODEL_FILES = {
    "pneumonia": (["pneumonia_cnn.keras", "pneumonia_cnn_threshold.json"], "clinical_cnn_rnn.ipynb"),
    "ecg": (["ecg_cnn_bilstm.keras", "ecg_rr_scaler.npz"], "clinical_cnn_rnn.ipynb"),
    "gan": (["xray_cgan_generator_ema.keras"], "xray_conditional_gan.ipynb"),
}


def missing_files(key):
    files, _ = MODEL_FILES[key]
    return [f for f in files if not (MODEL_DIR / f).exists()]


def require_models(key):
    """Stop the page with instructions if the trained model files are not in models/."""
    missing = missing_files(key)
    if missing:
        notebook = MODEL_FILES[key][1]
        st.error(
            f"Model files not found in `models/`: {', '.join(f'`{m}`' for m in missing)}.\n\n"
            f"Run `{notebook}` on Colab, download the results zip, and copy these files into "
            f"`clinical_ai_app/models/`."
        )
        st.stop()


@st.cache_resource(show_spinner="Loading pneumonia CNN…")
def load_pneumonia_model():
    import keras

    model = keras.models.load_model(MODEL_DIR / "pneumonia_cnn.keras", compile=False)
    threshold = json.loads((MODEL_DIR / "pneumonia_cnn_threshold.json").read_text())["threshold"]
    return model, float(threshold)


@st.cache_resource(show_spinner="Loading ECG CNN-BiLSTM…")
def load_ecg_model():
    import keras

    model = keras.models.load_model(MODEL_DIR / "ecg_cnn_bilstm.keras", compile=False)
    scaler = np.load(MODEL_DIR / "ecg_rr_scaler.npz")
    return model, scaler["mean"], scaler["std"]


@st.cache_resource(show_spinner="Loading X-ray GAN generator…")
def load_generator():
    import keras

    return keras.models.load_model(MODEL_DIR / "xray_cgan_generator_ema.keras", compile=False)
