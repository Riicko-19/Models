# 🩺 Clinical AI Lab — Streamlit app

Interactive demo of three clinical deep-learning models:

| Page | Model | Task |
|---|---|---|
| 🫁 Pneumonia X-ray | EfficientNet-B0 CNN + Grad-CAM | Flag pneumonia on pediatric chest X-rays |
| 💓 ECG Arrhythmia | Conv1D + BiLSTM + RR features | Label every heartbeat N / S / V / F (AAMI) |
| 🧪 Synthetic X-rays | Conditional GAN | Generate NORMAL / PNEUMONIA chest X-rays |

> ⚠️ Educational demo — not a medical device.

## 1. Add the trained models
Run the two notebooks on Colab (`clinical_cnn_rnn.ipynb`, `xray_conditional_gan.ipynb`). Each one downloads a zip at the end.
Copy these files into `models/`:

```
models/
├── pneumonia_cnn.keras
├── pneumonia_cnn_threshold.json
├── ecg_cnn_bilstm.keras
├── ecg_rr_scaler.npz
└── xray_cgan_generator_ema.keras
```

A page whose model is missing shows instructions instead of crashing.

## 2. Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

`sample_data/` already contains demo inputs (held-out test X-rays and 5-minute ECGs of unseen MIT-BIH patients).
To rebuild them: `python scripts/prepare_samples.py`.

## 3. Deploy to Streamlit Community Cloud (free)
1. Commit the model files in `models/` and push (about 55 MB in total; every file is under GitHub's 100 MB limit).
2. Go to <https://share.streamlit.io> → **Create app** → repo `Riicko-19/Models`, branch `main`, main file `clinical_ai_app/app.py`.
3. Under **Advanced settings**, choose **Python 3.12**.
4. Click **Deploy**. The first build installs TensorFlow and takes a few minutes.

## ECG upload format
CSV with one lead in millivolts (MLII / lead II is best); if there are several columns the last one is used.
Set the sampling rate on the page — the signal is resampled to 360 Hz. Example: `sample_data/ecg/example_upload.csv`.
