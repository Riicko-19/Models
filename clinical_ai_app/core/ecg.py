"""ECG preprocessing, R-peak detection and beat classification — mirrors clinical_cnn_rnn.ipynb (Part B)."""
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.signal import butter, filtfilt, find_peaks, resample_poly

FS = 360
CLASSES = ["N", "S", "V", "F"]
CLASS_INFO = {
    "N": "Normal / bundle-branch block",
    "S": "Supraventricular ectopic (SVEB)",
    "V": "Ventricular ectopic (VEB)",
    "F": "Fusion",
}
AAMI = {**dict.fromkeys("NLRej", "N"), **dict.fromkeys("AaJS", "S"), **dict.fromkeys("VE", "V"),
        "F": "F", **dict.fromkeys("/fQ", "Q")}
BEFORE, AFTER, DOWN = int(0.25 * FS), int(0.45 * FS), 2
_b_lp, _a_lp = butter(3, 35 / (FS / 2), btype="low")
_b_qrs, _a_qrs = butter(2, [5 / (FS / 2), 15 / (FS / 2)], btype="band")


def to_fs(sig, fs):
    """Resample a signal to the 360 Hz the model was trained on."""
    sig = np.asarray(sig, np.float64)
    if fs == FS:
        return sig
    from fractions import Fraction
    frac = Fraction(FS, int(round(fs))).limit_denominator(1000)
    return resample_poly(sig, frac.numerator, frac.denominator)


def clean(sig):
    baseline = median_filter(median_filter(sig, int(0.2 * FS) | 1), int(0.6 * FS) | 1)
    x = filtfilt(_b_lp, _a_lp, sig - baseline)
    return ((x - x.mean()) / x.std()).astype(np.float32)


def detect_r_peaks(sig_clean):
    """Pan-Tompkins-style detector: QRS band-pass → squared slope → 150 ms integration → peaks."""
    x = filtfilt(_b_qrs, _a_qrs, sig_clean)
    w = int(0.15 * FS)
    energy = np.convolve(np.diff(x, prepend=x[0]) ** 2, np.ones(w) / w, mode="same")
    peaks, _ = find_peaks(energy, distance=int(0.25 * FS), height=0.3 * np.percentile(energy, 99))
    r = int(0.075 * FS)
    refined = [lo + int(np.argmax(np.abs(sig_clean[lo:hi])))
               for lo, hi in ((max(0, p - r), min(len(sig_clean), p + r)) for p in peaks)]
    return np.unique(refined)


def beat_inputs(sig_clean, peaks):
    """Beat windows + RR-interval features for every beat that has both neighbours."""
    rr = np.diff(peaks) / FS
    mean_rr = rr.mean()
    X, R, POS = [], [], []
    for i in range(1, len(peaks) - 1):
        s = peaks[i]
        if s - BEFORE < 0 or s + AFTER > len(sig_clean):
            continue
        pre, post, local = rr[i - 1], rr[i], rr[max(0, i - 10):i].mean()
        X.append(sig_clean[s - BEFORE:s + AFTER:DOWN])
        R.append([pre, post, local, pre / mean_rr, post / mean_rr, pre / local, post / pre])
        POS.append(s)
    return np.array(X, np.float32), np.array(R, np.float32), np.array(POS)


def classify(model, rr_mean, rr_std, sig_clean, peaks):
    X, R, pos = beat_inputs(sig_clean, peaks)
    if len(pos) == 0:
        return pd.DataFrame(columns=["sample", "time_s", "pred", "confidence", *CLASSES])
    probs = model.predict({"beat": X[..., None], "rr": (R - rr_mean) / rr_std}, batch_size=1024, verbose=0)
    df = pd.DataFrame(probs, columns=CLASSES)
    df.insert(0, "sample", pos)
    df.insert(1, "time_s", pos / FS)
    df.insert(2, "pred", [CLASSES[i] for i in probs.argmax(1)])
    df.insert(3, "confidence", probs.max(1))
    return df


def match_annotations(beats, ann_samples, ann_symbols, tol_s=0.1):
    """Attach the cardiologist's AAMI label to each detected beat (nearest annotation within tol)."""
    keep = np.array([AAMI.get(s) in CLASSES for s in ann_symbols])
    a_s, a_c = np.asarray(ann_samples)[keep], np.array([AAMI[s] for s in np.asarray(ann_symbols)[keep]])
    if len(a_s) == 0 or beats.empty:
        return beats.assign(true=None)
    idx = np.clip(np.searchsorted(a_s, beats["sample"].to_numpy()), 1, len(a_s) - 1)
    left, right = a_s[idx - 1], a_s[idx]
    nearest = np.where(np.abs(beats["sample"] - left) <= np.abs(beats["sample"] - right), idx - 1, idx)
    ok = np.abs(a_s[nearest] - beats["sample"].to_numpy()) <= tol_s * FS
    return beats.assign(true=np.where(ok, a_c[nearest], None))
