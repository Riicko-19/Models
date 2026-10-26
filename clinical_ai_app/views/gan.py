import io
import zipfile

import numpy as np
import streamlit as st
from PIL import Image

from core.models import load_generator, require_models

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

st.title("🧪 Synthetic chest X-rays (conditional GAN)")
st.caption("Projection-discriminator cGAN with spectral norm, hinge loss, DiffAugment and an EMA generator")
st.info("Every image on this page is **synthetic** — generated from random noise, not taken from a patient.")

require_models("gan")
G = load_generator()
LATENT = int(G.inputs[0].shape[-1])
NATIVE = int(G.outputs[0].shape[1])


@st.cache_data(max_entries=32)
def generate(z_seed: int, n: int, cls: int):
    import tensorflow as tf

    z = np.random.default_rng(z_seed).standard_normal((n, LATENT)).astype("float32")
    imgs = load_generator()([tf.constant(z), tf.fill([n, 1], cls)], training=False).numpy()
    return ((imgs[..., 0] + 1) * 127.5).clip(0, 255).astype(np.uint8)


@st.cache_data(max_entries=16)
def interpolate(seed_a: int, seed_b: int, cls: int, steps: int):
    import tensorflow as tf

    za = np.random.default_rng(seed_a).standard_normal(LATENT)
    zb = np.random.default_rng(seed_b).standard_normal(LATENT)
    z = np.stack([(1 - t) * za + t * zb for t in np.linspace(0, 1, steps)]).astype("float32")
    imgs = load_generator()([tf.constant(z), tf.fill([steps, 1], cls)], training=False).numpy()
    return ((imgs[..., 0] + 1) * 127.5).clip(0, 255).astype(np.uint8)


def upscale(a, size=192):
    return Image.fromarray(a).resize((size, size), Image.BICUBIC)


def show_row(imgs, cols, captions=None):
    for i in range(0, len(imgs), cols):
        row = st.columns(cols)
        for j, im in enumerate(imgs[i:i + cols]):
            row[j].image(upscale(im), caption=captions[i + j] if captions else None, width="stretch")


with st.sidebar:
    mode = st.radio("Class", ["NORMAL", "PNEUMONIA", "Both (same noise)"])
    n = st.slider("Number of images", 4, 24, 8, 4)
    st.session_state.setdefault("seed", 42)
    seed = st.number_input("Random seed", 0, 1_000_000, key="seed")
    st.button("🎲 New random seed", width="stretch",
              on_click=lambda: st.session_state.update(seed=int(np.random.default_rng().integers(1_000_000))))

st.subheader("Generated images")
st.caption(f"Native resolution {NATIVE}×{NATIVE}, shown upscaled. Seed {seed}.")
if mode.startswith("Both"):
    st.markdown("Each column uses the **same noise vector** — only the class label changes.")
    normal, pneu = generate(int(seed), n, 0), generate(int(seed), n, 1)
    st.markdown("**NORMAL**")
    show_row(list(normal), min(n, 8))
    st.markdown("**PNEUMONIA**")
    show_row(list(pneu), min(n, 8))
    batch = [(f"normal_{i:02d}.png", im) for i, im in enumerate(normal)] + \
            [(f"pneumonia_{i:02d}.png", im) for i, im in enumerate(pneu)]
else:
    cls = CLASS_NAMES.index(mode)
    imgs = generate(int(seed), n, cls)
    show_row(list(imgs), 8 if n >= 8 else n)
    batch = [(f"{mode.lower()}_{i:02d}.png", im) for i, im in enumerate(imgs)]

buf = io.BytesIO()
with zipfile.ZipFile(buf, "w") as zf:
    for name, im in batch:
        png = io.BytesIO()
        Image.fromarray(im).save(png, format="PNG")
        zf.writestr(name, png.getvalue())
st.download_button(f"Download these {len(batch)} images (PNG, {NATIVE}×{NATIVE})", buf.getvalue(),
                   f"synthetic_xrays_seed{seed}.zip", "application/zip")

st.divider()
st.subheader("Latent-space interpolation")
st.markdown("Walking in a straight line between two noise vectors. Smooth transitions mean the GAN learned a "
            "continuous space of anatomies rather than memorising individual training images.")
c1, c2, c3, c4 = st.columns(4)
seed_a = c1.number_input("Seed A", 0, 1_000_000, 1)
seed_b = c2.number_input("Seed B", 0, 1_000_000, 2)
icls = c3.selectbox("Class ", CLASS_NAMES, index=1)
steps = c4.slider("Steps", 4, 12, 8)
show_row(list(interpolate(int(seed_a), int(seed_b), CLASS_NAMES.index(icls), steps)), steps,
         [f"{t:.2f}" for t in np.linspace(0, 1, steps)])
