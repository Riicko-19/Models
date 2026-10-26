"""Build the small demo files in sample_data/ from the original datasets.

Only *test-set* data is used, so every demo input is unseen by the models:
  * X-rays: from the official test split of the Kermany et al. chest X-ray dataset
  * ECGs:   5-minute MLII excerpts of MIT-BIH records from the inter-patient test set (DS2)

Usage:  python scripts/prepare_samples.py [--data-dir path/with/downloads]
"""
import argparse
import io
import json
import os
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

HF = "https://huggingface.co/datasets"
XRAY_TEST = f"{HF}/hf-vision/chest-xray-pneumonia/resolve/main/data/test-00000-of-00001.parquet"
ECG_URL = f"{HF}/epr-labs/mit-bih-arrhythmia-database/resolve/main/data/train-00000-of-00001.parquet"
OUT = Path(__file__).resolve().parent.parent / "sample_data"

ECG_RECORDS = {
    "100": "Mostly normal sinus rhythm",
    "200": "Frequent ventricular ectopic (V) beats",
    "213": "Fusion (F) and ventricular beats",
    "232": "Many supraventricular (S) beats, sinus bradycardia",
    "233": "Frequent V beats, including couplets",
}
ECG_MINUTES = 5
N_XRAYS_PER_CLASS = 4


def fetch(url, path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print("downloading", url)
        urllib.request.urlretrieve(url, path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data", help="where the notebooks' downloads live (or will be saved)")
    data = Path(ap.parse_args().data_dir)

    # ---- X-rays
    xdir = OUT / "xray"
    xdir.mkdir(parents=True, exist_ok=True)
    table = pq.read_table(fetch(XRAY_TEST, data / "chest_xray" / "test-00000-of-00001.parquet"))
    labels = np.array(table.column("label").to_pylist())
    images = table.column("image").to_pylist()
    rng = np.random.default_rng(7)
    for cls, name in enumerate(["NORMAL", "PNEUMONIA"]):
        for k, i in enumerate(rng.choice(np.where(labels == cls)[0], N_XRAYS_PER_CLASS, replace=False)):
            im = Image.open(io.BytesIO(images[i]["bytes"])).convert("L")
            im.thumbnail((640, 640))
            im.save(xdir / f"{name.lower()}_{k + 1}.jpg", quality=90)
    print("x-rays ->", xdir)

    # ---- ECGs
    edir = OUT / "ecg"
    edir.mkdir(parents=True, exist_ok=True)
    table = pq.read_table(fetch(ECG_URL, data / "mitbih" / "mitbih.parquet"))
    signals = table.column("signals").combine_chunks()
    n = ECG_MINUTES * 60 * 360
    for i, hdr in enumerate(table.column("header").to_pylist()):
        hdr = json.loads(hdr)
        rec = hdr["record_name"]
        if rec not in ECG_RECORDS:
            continue
        sig = signals[i].values.flatten().to_numpy(zero_copy_only=False).reshape(-1, hdr["n_sig"])
        mlii = sig[:n, hdr["sig_name"].index("MLII")].astype(np.float32)
        samples = np.array(table.column("annotation_sample")[i].as_py())
        symbols = np.array(table.column("annotation_symbol")[i].as_py())
        keep = samples < n
        np.savez_compressed(edir / f"mitbih_{rec}.npz", signal=mlii, fs=360, samples=samples[keep],
                            symbols=symbols[keep], description=ECG_RECORDS[rec])
        if rec == "233":   # also an example of the CSV upload format: 60 s, time + millivolts
            t = np.arange(60 * 360) / 360
            np.savetxt(edir / "example_upload.csv", np.c_[t, mlii[:60 * 360]], delimiter=",",
                       header="time_s,mv", comments="", fmt=["%.4f", "%.4f"])
    print("ecgs   ->", edir)


if __name__ == "__main__":
    main()
