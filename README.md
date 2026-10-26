# 🩺 Clinical AI — CNN, RNN & GAN for medical diagnostics

Three deep-learning models for real clinical problems, trained on public expert-labelled datasets, plus a Streamlit app to use them.

| Notebook | Problem | Data | Model |
|---|---|---|---|
| [`clinical_cnn_rnn.ipynb`](clinical_cnn_rnn.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Riicko-19/Models/blob/main/clinical_cnn_rnn.ipynb) | **A.** Pneumonia screening from pediatric chest X-rays<br>**B.** Arrhythmia detection from ECG heartbeats | Kermany et al. 2018 (5,856 X-rays)<br>MIT-BIH Arrhythmia DB (~100k beats) | EfficientNet-B0 transfer learning + Grad-CAM<br>Conv1D + BiLSTM + RR-interval features, inter-patient evaluation |
| [`xray_conditional_gan.ipynb`](xray_conditional_gan.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Riicko-19/Models/blob/main/xray_conditional_gan.ipynb) | Medical-imaging data scarcity & class imbalance | Same chest X-rays | Conditional GAN (projection D, spectral norm, hinge loss, DiffAugment, EMA) evaluated with FID, a memorisation check and TSTR / augmentation experiments |
| [`clinical_ai_app/`](clinical_ai_app) | Interactive demo of all three models | — | Streamlit |

## Train
Open a notebook in Colab → *Runtime → Change runtime type → T4 GPU* → *Run all*. Datasets (~1.3 GB) are downloaded
automatically from the Hugging Face Hub, and each notebook downloads a zip of trained models at the end.

## Deploy the app
Copy the trained model files into `clinical_ai_app/models/` (see [`clinical_ai_app/models/README.md`](clinical_ai_app/models/README.md)),
commit and push, then on [share.streamlit.io](https://share.streamlit.io) create an app from this repo with main file
**`clinical_ai_app/app.py`** and Python 3.12. Details in [`clinical_ai_app/README.md`](clinical_ai_app/README.md).

> ⚠️ Educational project — not a medical device.
