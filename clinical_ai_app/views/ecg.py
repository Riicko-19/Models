import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.ecg import AAMI, CLASS_INFO, CLASSES, FS, classify, clean, detect_r_peaks, match_annotations, to_fs
from core.models import SAMPLE_DIR, load_ecg_model, require_models

COLORS = {"N": "#2ca02c", "S": "#ff7f0e", "V": "#d62728", "F": "#9467bd"}

st.title("💓 Arrhythmia detection from ECG")
st.caption("Conv1D + BiLSTM over each heartbeat with RR-interval features · MIT-BIH · AAMI classes N / S / V / F")

require_models("ecg")
load_ecg_model()


@st.cache_data(show_spinner="Filtering signal and classifying beats…", max_entries=16)
def analyse(sig, fs, ann_samples=None):
    model, rr_mean, rr_std = load_ecg_model()
    sig_clean = clean(to_fs(sig, fs))
    peaks = ann_samples if ann_samples is not None else detect_r_peaks(sig_clean)
    return sig_clean, classify(model, rr_mean, rr_std, sig_clean, peaks)


source = st.radio("Input", ["Demo patient (MIT-BIH test set)", "Upload a CSV"], horizontal=True)
ann = None
if source.startswith("Demo"):
    files = sorted((SAMPLE_DIR / "ecg").glob("mitbih_*.npz"))
    if not files:
        st.warning("No demo recordings found — run `python scripts/prepare_samples.py`.")
        st.stop()
    demos = {f: np.load(f) for f in files}
    f = st.selectbox("Recording", files, format_func=lambda p: f"Record {p.stem.split('_')[1]} — {demos[p]['description']}")
    d = demos[f]
    sig, fs = d["signal"], int(d["fs"])
    ann = (d["samples"], d["symbols"])
    use_ann = st.toggle("Use the cardiologist's R-peak positions (instead of automatic beat detection)", value=False)
    is_beat = np.isin(ann[1], list(AAMI))
    sig_clean, beats = analyse(sig, fs, ann[0][is_beat] if use_ann else None)
else:
    st.markdown("CSV with one ECG lead in **millivolts** (lead II / MLII works best). "
                "If there are several columns, the last one is used. Example: `sample_data/ecg/example_upload.csv`.")
    upload = st.file_uploader("ECG CSV", type=["csv", "txt"])
    fs = st.number_input("Sampling rate (Hz)", 100, 2000, 360, help="The signal is resampled to 360 Hz internally.")
    if upload is None:
        st.info("Upload a recording (at least ~10 seconds) to analyse it.")
        st.stop()
    try:
        # header=None + to_numeric: a header row (if any) becomes NaN and is dropped; no sample is lost otherwise
        text = upload.getvalue().decode("utf-8", errors="ignore")
        head = text.splitlines()[:5]
        sep = next((d for d in [",", ";", "\t"] if any(d in line for line in head)), r"\s+")
        df_in = pd.read_csv(io.StringIO(text), header=None, sep=sep, engine="python")
        sig = df_in.iloc[:, -1].pipe(pd.to_numeric, errors="coerce").dropna().to_numpy(np.float64)
    except Exception as e:
        st.error(f"Could not read the CSV: {e}")
        st.stop()
    if len(sig) < 10 * fs:
        st.error(f"Recording too short: {len(sig) / fs:.1f}s (need ≥ 10 s).")
        st.stop()
    sig_clean, beats = analyse(sig, int(fs))

if beats.empty:
    st.error("No heartbeats could be detected in this signal.")
    st.stop()
if ann is not None:
    beats = match_annotations(beats, *ann)

# ---- summary
duration = len(sig_clean) / FS
rr = np.diff(beats["sample"]) / FS
counts = beats["pred"].value_counts().reindex(CLASSES, fill_value=0)
abnormal = int(counts[["S", "V", "F"]].sum())
cols = st.columns(4)
cols[0].metric("Duration", f"{duration / 60:.1f} min")
cols[1].metric("Beats analysed", f"{len(beats)}")
cols[2].metric("Median heart rate", f"{60 / np.median(rr):.0f} bpm" if len(rr) else "–")
cols[3].metric("Abnormal beats", f"{abnormal} ({abnormal / len(beats):.1%})")

cols = st.columns(4)
for c, cls in zip(cols, CLASSES):
    c.metric(f"{cls} — {CLASS_INFO[cls]}", int(counts[cls]), f"{60 * counts[cls] / duration:.1f}/min", delta_color="off")

if counts["V"] >= 1:
    st.warning(f"**{counts['V']} ventricular ectopic beats** detected ({60 * counts['V'] / duration:.1f} per minute).")
if counts["S"] >= 1:
    st.info(f"**{counts['S']} supraventricular ectopic beats** detected — S beats are the least reliable class on new patients.")

# ---- interactive strip
st.subheader("ECG strip")
c1, c2 = st.columns([1, 3])
win = c1.selectbox("Window", [10, 20, 30, 60], index=1, format_func=lambda s: f"{s} s")
start = c2.slider("Start (s)", 0.0, duration - win, 0.0, 1.0) if duration > win + 1 else 0.0
a, b = int(start * FS), int(min(duration, start + win) * FS)
shown = beats[(beats["sample"] >= a) & (beats["sample"] < b)]

fig = go.Figure()
fig.add_trace(go.Scatter(x=np.arange(a, b) / FS, y=sig_clean[a:b], mode="lines", line=dict(color="#444", width=1),
                         name="filtered ECG", hoverinfo="skip"))
for cls in CLASSES:
    sel = shown[shown["pred"] == cls]
    if sel.empty:
        continue
    hover = [f"predicted {p} ({c:.0%})" + (f"<br>cardiologist: {t}" if ann is not None else "")
             for p, c, t in zip(sel["pred"], sel["confidence"], sel.get("true", [None] * len(sel)))]
    fig.add_trace(go.Scatter(x=sel["time_s"], y=sig_clean[sel["sample"]] + 0.4, mode="markers+text",
                             marker=dict(color=COLORS[cls], size=9, symbol="triangle-down"),
                             text=sel["pred"], textposition="top center", textfont=dict(color=COLORS[cls]),
                             name=f"{cls} — {CLASS_INFO[cls]}", hovertext=hover, hoverinfo="text"))
fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="time (s)",
                  yaxis=dict(showticklabels=False, title="z-score"), legend=dict(orientation="h", y=-0.25))
st.plotly_chart(fig, width="stretch")

# ---- agreement with the cardiologist (demo records)
if ann is not None and beats["true"].notna().any():
    st.subheader("Agreement with the cardiologist's labels")
    m = beats.dropna(subset=["true"])
    st.caption(f"{len(m)} of {len(beats)} detected beats matched an annotated beat within 100 ms.")
    left, right = st.columns([2, 3])
    left.metric("Beat-level agreement", f"{(m['true'] == m['pred']).mean():.1%}")
    sens = {c: ((m["true"] == c) & (m["pred"] == c)).sum() / max((m["true"] == c).sum(), 1)
            for c in CLASSES if (m["true"] == c).any()}
    left.dataframe(pd.DataFrame({"beats": m["true"].value_counts().reindex(sens), "sensitivity": sens}),
                   column_config={"sensitivity": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")})
    right.dataframe(pd.crosstab(m["true"].rename("cardiologist ↓ / model →"), m["pred"])
                    .reindex(index=CLASSES, columns=CLASSES, fill_value=0).loc[lambda d: d.sum(axis=1) > 0])

# ---- beat table
st.subheader("Abnormal beats")
cols = ["time_s", "pred", "confidence"] + (["true"] if ann is not None else []) + CLASSES
st.dataframe(beats.loc[beats["pred"] != "N", cols].round(3), width="stretch", hide_index=True)
st.download_button("Download all beat predictions (CSV)", beats.to_csv(index=False).encode(), "ecg_beats.csv", "text/csv")
