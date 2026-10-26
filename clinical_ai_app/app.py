import streamlit as st

st.set_page_config(page_title="Clinical AI Lab", page_icon="🩺", layout="wide")

pages = [
    st.Page("views/home.py", title="Overview", icon="🏠", default=True),
    st.Page("views/xray.py", title="Pneumonia X-ray (CNN)", icon="🫁"),
    st.Page("views/ecg.py", title="ECG Arrhythmia (RNN)", icon="💓"),
    st.Page("views/gan.py", title="Synthetic X-rays (GAN)", icon="🧪"),
]

with st.sidebar:
    st.caption("⚠️ Educational demo — not a medical device. Do not use for diagnosis.")

st.navigation(pages).run()
