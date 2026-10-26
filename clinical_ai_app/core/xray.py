"""Chest X-ray preprocessing, prediction and Grad-CAM — mirrors clinical_cnn_rnn.ipynb (Part A)."""
import io

import numpy as np
from matplotlib import colormaps
from PIL import Image

IMG_SIZE = 224
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


def load_xray(data: bytes, size: int = IMG_SIZE) -> np.ndarray:
    """Decode an image file to a (size, size, 1) uint8 grayscale array, exactly as in training."""
    im = Image.open(io.BytesIO(data))
    im.draft("L", (size * 2, size * 2))      # no-op for non-JPEG files
    im = im.convert("L").resize((size, size), Image.BILINEAR)
    return np.asarray(im, np.uint8)[..., None]


def _parts(model):
    """The EfficientNet backbone and classification-head layers inside the saved model."""
    import keras

    base = next(l for l in model.layers if isinstance(l, keras.Model) and l.name != "augment")
    return base, model.get_layer("gap"), model.get_layer("pneumonia_prob")


def predict_with_cam(model, img_u8: np.ndarray):
    """Return (pneumonia probability, Grad-CAM heatmap in [0, 1] at IMG_SIZE)."""
    import tensorflow as tf

    base, gap, head = _parts(model)
    x = tf.cast(img_u8[None], tf.float32)
    x = tf.concat([x, x, x], axis=-1)
    with tf.GradientTape() as tape:
        feats = base(x, training=False)
        tape.watch(feats)
        prob = head(gap(feats))
    grads = tape.gradient(prob, feats)[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = tf.nn.relu(tf.reduce_sum(feats[0] * weights, axis=-1))
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    cam = tf.image.resize(cam[..., None], (IMG_SIZE, IMG_SIZE))[..., 0].numpy()
    return float(prob[0, 0]), cam


def overlay(img_u8: np.ndarray, cam: np.ndarray, alpha: float = 0.4, out_size: int = 448) -> Image.Image:
    gray = np.repeat(img_u8.astype(np.float32) / 255.0, 3, axis=-1)
    heat = colormaps["jet"](cam)[..., :3]
    blend = (1 - alpha) * gray + alpha * heat
    return Image.fromarray((blend * 255).astype(np.uint8)).resize((out_size, out_size), Image.BILINEAR)
