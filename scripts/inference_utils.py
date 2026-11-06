"""
Shared inference helpers for the course extension (notebooks 05/06 and app/app.py).

Everything here is ADDITIVE: it imports the existing data_pipeline.py and
gradcam.py unchanged and re-uses their preprocessing contract, so notebooks
and the app never carry a second copy of that logic.

What this adds on top of the trained FP32 model (best_model.keras):
  1. Input-quality gate      - rejects photos the model is known to handle badly.
  2. Confidence thresholding - "uncertain, consult an expert" instead of a forced label.
  3. Advisory lookup         - curated, plain-language guidance per class.
  4. Grad-CAM overlay        - reuses scripts/gradcam.py.
  5. Analysis helpers        - calibration, coverage-vs-accuracy, controlled corruption.

Decision-support only: this is NOT a diagnosis.
"""
import json
import os
import sys
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_SCRIPTS_DIR, ".."))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import data_pipeline as dp  # noqa: E402  (existing module, unchanged)

ADVISORY_PATH = os.path.join(ROOT, "advisory", "advisory_table.csv")
GATE_CONFIG_PATH = os.path.join(ROOT, "advisory", "gate_config.json")
ALLOWED_CATEGORIES = {"healthy", "fungal", "oomycete", "bacterial", "viral", "pest"}
ADVISORY_COLUMNS = [
    "class_name", "crop", "condition", "is_healthy", "category", "cause",
    "what_to_do_now", "prevention", "when_to_seek_help", "source_name",
    "source_url", "source_status",
]

DISCLAIMER = (
    "Decision support only, not a diagnosis. The model knows 15 classes of Potato, Tomato and "
    "Pepper leaves from lab-style photos. Confirm with a local agriculture extension officer "
    "before spraying or removing crops."
)
GRADCAM_NOTE = (
    "Warm colours mark where the model looked most for this prediction. This shows attention, "
    "not proof of the cause."
)
UNKNOWN_ADVICE = {
    "crop": "unknown", "condition": "uncertain result", "is_healthy": False, "category": "unknown",
    "cause": "The model is not confident enough to name a condition.",
    "what_to_do_now": "Retake the photo (single leaf, good daylight, in focus, plain background) "
                      "and show the plant to a local agriculture extension officer.",
    "prevention": "", "when_to_seek_help": "Now, if the plant is visibly declining.",
    "source_name": "", "source_url": "", "source_status": "",
}


# --------------------------------------------------------------------------------------
# Image helpers
# --------------------------------------------------------------------------------------
def _decode_bytes(data):
    """Decode image bytes with TensorFlow, the SAME decoder the training pipeline uses.
    PIL/libjpeg decodes JPEGs to slightly different pixels, so paths and uploads go through TF."""
    import tensorflow as tf
    return tf.io.decode_image(data, channels=3, expand_animations=False).numpy().astype(np.uint8)


def to_uint8_rgb(img):
    """Accept a file path, raw bytes, a file-like upload, a PIL image or an ndarray; return an
    HxWx3 uint8 RGB array. Paths/bytes/uploads are decoded with TensorFlow to match training."""
    from PIL import Image
    if isinstance(img, (str, os.PathLike)):
        with open(img, "rb") as f:
            return _decode_bytes(f.read())
    if isinstance(img, (bytes, bytearray)):
        return _decode_bytes(bytes(img))
    if hasattr(img, "read") and not isinstance(img, Image.Image):
        data = img.read()
        if hasattr(img, "seek"):
            img.seek(0)
        return _decode_bytes(data)
    if isinstance(img, Image.Image):
        return np.asarray(img.convert("RGB"), dtype=np.uint8)
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return np.clip(arr, 0, 255).astype(np.uint8)


def resize_for_model(rgb_uint8):
    """Exactly the training/eval preprocessing: tf.image.resize to 224x224, float32, RAW 0-255
    scale (the model has no /255 step in the data pipeline; do NOT normalise)."""
    import tensorflow as tf
    out = tf.image.resize(tf.convert_to_tensor(rgb_uint8), (dp.IMG_SIZE, dp.IMG_SIZE))
    return tf.cast(out, tf.float32).numpy()


def _as_uint8(resized_float):
    return np.clip(resized_float, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------------------
# 1. Input-quality gate
# --------------------------------------------------------------------------------------
@dataclass
class GateConfig:
    """Provisional defaults. Notebook 05 recalibrates them from the clean VAL split and
    saves advisory/gate_config.json, which GateConfig.load() then prefers."""
    min_side: int = 100
    min_sharpness: float = 100.0      # variance of the Laplacian (low = blurry)
    max_noise_sigma: float = 15.0     # Immerkaer noise estimate, in gray levels
    min_brightness: float = 40.0
    max_brightness: float = 215.0

    @classmethod
    def load(cls, path=GATE_CONFIG_PATH):
        if path and os.path.exists(path):
            with open(path) as f:
                raw = json.load(f)
            return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})
        return cls()

    def save(self, path=GATE_CONFIG_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)


def image_quality_metrics(rgb224):
    """Sharpness, noise level and brightness of a 224x224 uint8 RGB image."""
    import cv2
    gray = cv2.cvtColor(rgb224, cv2.COLOR_RGB2GRAY).astype(np.float32)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    h, w = gray.shape
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    conv = cv2.filter2D(gray, -1, kernel)[1:-1, 1:-1]
    noise_sigma = float(np.sqrt(np.pi / 2) / (6 * (h - 2) * (w - 2)) * np.abs(conv).sum())
    return {"sharpness": sharpness, "noise_sigma": noise_sigma, "brightness": float(gray.mean())}


def evaluate_gate(metrics, min_side, cfg):
    """Pure decision logic: metrics + original shortest side -> (passed, reasons)."""
    reasons = []
    if min_side < cfg.min_side:
        reasons.append(f"image too small ({min_side}px, need at least {cfg.min_side}px)")
    if metrics["sharpness"] < cfg.min_sharpness:
        reasons.append("image looks blurry")
    if metrics["noise_sigma"] > cfg.max_noise_sigma:
        reasons.append("image looks grainy or noisy")
    if metrics["brightness"] < cfg.min_brightness:
        reasons.append("image is too dark")
    if metrics["brightness"] > cfg.max_brightness:
        reasons.append("image is too bright or washed out")
    return (not reasons), reasons


def quality_gate(image, cfg=None):
    cfg = cfg or GateConfig.load()
    rgb = to_uint8_rgb(image)
    min_side = int(min(rgb.shape[:2]))
    metrics = image_quality_metrics(_as_uint8(resize_for_model(rgb)))
    metrics["min_side"] = min_side
    passed, reasons = evaluate_gate(metrics, min_side, cfg)
    return {"passed": passed, "reasons": reasons, "metrics": metrics}


# --------------------------------------------------------------------------------------
# Controlled, version-independent corruption (explicit severities)
# --------------------------------------------------------------------------------------
def corrupt_image(rgb224, noise_sigma=0.0, blur_len=0, brightness_delta=0.0, seed=0):
    """Blur -> noise -> brightness, same order as evaluate.py, but with EXPLICIT severities in
    gray levels / pixels so results do not depend on a library's default parameters."""
    import cv2
    out = rgb224.astype(np.float32)
    if blur_len and blur_len > 1:
        k = np.zeros((blur_len, blur_len), np.float32)
        k[blur_len // 2, :] = 1.0 / blur_len  # horizontal motion blur
        out = cv2.filter2D(out, -1, k)
    if noise_sigma and noise_sigma > 0:
        rng = np.random.default_rng(seed)
        out = out + rng.normal(0.0, noise_sigma, out.shape).astype(np.float32)
    if brightness_delta:
        out = out + float(brightness_delta)
    return np.clip(out, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------------------
# 2. Confidence thresholding + analysis helpers
# --------------------------------------------------------------------------------------
def decide(probs, threshold, class_names=None):
    """Top-1 decision with an explicit 'uncertain' state below the threshold."""
    class_names = class_names or dp.SELECTED_CLASSES
    probs = np.asarray(probs, dtype=np.float64)
    order = np.argsort(probs)[::-1]
    top = int(order[0])
    conf = float(probs[top])
    return {
        "pred_idx": top,
        "label": class_names[top],
        "confidence": conf,
        "status": "ok" if conf >= threshold else "uncertain",
        "top3": [(class_names[int(i)], float(probs[int(i)])) for i in order[:3]],
    }


def coverage_accuracy_table(frame, thresholds=None, conf_col="confidence", correct_col="correct"):
    """Selective prediction: for each threshold, what fraction is answered and how accurate."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.30, 1.0, 0.05), 2).tolist() + [0.99]
    rows = []
    for t in thresholds:
        accepted = frame[conf_col] >= t
        n = int(accepted.sum())
        rows.append({
            "threshold": t, "n_accepted": n,
            "coverage": n / len(frame) if len(frame) else float("nan"),
            "selective_accuracy": float(frame.loc[accepted, correct_col].mean()) if n else float("nan"),
        })
    return pd.DataFrame(rows)


def pick_threshold(table, target_accuracy=0.95, min_coverage=0.5):
    """Lowest threshold meeting the accuracy target while still answering min_coverage of
    inputs. Returns None when no threshold qualifies (caller must handle it)."""
    ok = table[(table["selective_accuracy"] >= target_accuracy) & (table["coverage"] >= min_coverage)]
    return None if ok.empty else float(ok["threshold"].min())


def expected_calibration_error(conf, correct, n_bins=10):
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, bins[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def predict_frame(model, df, cfg=None, corrupt=None, batch_size=32):
    """Run the model over manifest rows (needs File_Path, label_idx) and return a DataFrame with
    prediction, confidence, margin, entropy and quality metrics per image. `corrupt` is either a dict
    of corrupt_image() kwargs or a callable(uint8 224x224 RGB) -> uint8 image, applied at the same
    point evaluate.py corrupts (after the resize)."""
    cfg = cfg or GateConfig.load()
    rows, batch, meta = [], [], []

    def flush():
        if not batch:
            return
        _, probs = model.predict(np.stack(batch), verbose=0)
        for m, p in zip(meta, probs):
            order = np.argsort(p)[::-1]
            m.update({
                "pred_idx": int(order[0]), "confidence": float(p[order[0]]),
                "margin": float(p[order[0]] - p[order[1]]),
                "entropy": float(-(p * np.log(p + 1e-12)).sum()),
            })
            m["correct"] = bool(m["pred_idx"] == m["label_idx"])
            rows.append(m)
        batch.clear()
        meta.clear()

    for _, r in df.iterrows():
        rgb = to_uint8_rgb(r["File_Path"])
        min_side = int(min(rgb.shape[:2]))
        x = resize_for_model(rgb)
        if corrupt is not None and callable(corrupt):
            x = np.asarray(corrupt(_as_uint8(x)), dtype=np.uint8).astype(np.float32)
        elif corrupt:
            x = corrupt_image(_as_uint8(x), **corrupt).astype(np.float32)
        q = image_quality_metrics(_as_uint8(x))
        passed, _ = evaluate_gate(q, min_side, cfg)
        batch.append(x)
        meta.append({"File_Path": r["File_Path"], "Class_Name": r["Class_Name"],
                     "label_idx": int(r["label_idx"]), "gate_passed": passed, **q})
        if len(batch) == batch_size:
            flush()
    flush()
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# 3. Advisory table
# --------------------------------------------------------------------------------------
def load_advisory(path=None):
    df = pd.read_csv(path or ADVISORY_PATH, dtype=str, keep_default_na=False)
    df["is_healthy"] = df["is_healthy"].str.lower().isin(["true", "1", "yes"])
    return df.set_index("class_name", drop=False)


def validate_advisory(df, class_names=None):
    """Return a list of problems (empty list = valid)."""
    class_names = class_names or dp.SELECTED_CLASSES
    problems = []
    missing = [c for c in class_names if c not in df.index]
    extra = [c for c in df.index if c not in class_names]
    if missing:
        problems.append(f"classes without advisory entry: {missing}")
    if extra:
        problems.append(f"advisory rows for unknown classes: {extra}")
    for col in ADVISORY_COLUMNS:
        if col not in df.columns:
            problems.append(f"missing column: {col}")
    if problems:
        return problems
    for cls, r in df.iterrows():
        if r["category"] not in ALLOWED_CATEGORIES:
            problems.append(f"{cls}: invalid category '{r['category']}'")
        if bool(r["is_healthy"]) != cls.lower().endswith("healthy"):
            problems.append(f"{cls}: is_healthy flag inconsistent with class name")
        for col in ["crop", "condition", "cause", "what_to_do_now", "source_name"]:
            if not str(r[col]).strip():
                problems.append(f"{cls}: empty '{col}'")
    return problems


def lookup_advisory(df, class_name):
    """Never raises: unknown class -> generic 'consult an expert' advice."""
    if class_name in df.index:
        return df.loc[class_name].to_dict()
    return dict(UNKNOWN_ADVICE, class_name=class_name)


# --------------------------------------------------------------------------------------
# 4. Grad-CAM overlay and full single-image analysis
# --------------------------------------------------------------------------------------
def gradcam_overlay(rgb224, heatmap, alpha=0.45):
    import cv2
    big = cv2.resize(heatmap.astype(np.float32), (rgb224.shape[1], rgb224.shape[0]))
    heat = cv2.applyColorMap(np.uint8(255 * np.clip(big, 0, 1)), cv2.COLORMAP_JET)[:, :, ::-1]
    return np.uint8((1 - alpha) * rgb224 + alpha * heat)


def analyze_image(model, image, advisory=None, cfg=None, threshold=0.8, with_gradcam=True):
    """Full pipeline for one photo: gate -> predict -> threshold -> advisory -> Grad-CAM.
    status is one of: 'rejected_quality', 'uncertain', 'ok'."""
    advisory = advisory if advisory is not None else load_advisory()
    rgb = to_uint8_rgb(image)
    gate = quality_gate(rgb, cfg)
    result = {"gate": gate, "disclaimer": DISCLAIMER}
    if not gate["passed"]:
        result.update(
            status="rejected_quality",
            message="Photo quality too low: " + "; ".join(gate["reasons"]) +
                    ". Retake one leaf in good daylight, in focus, filling the frame.",
        )
        return result

    x = resize_for_model(rgb)
    _, probs = model.predict(x[None], verbose=0)
    decision = decide(probs[0], threshold)
    result.update(decision)
    result["advisory"] = lookup_advisory(advisory, decision["label"]) if decision["status"] == "ok" \
        else dict(UNKNOWN_ADVICE)
    if decision["status"] == "uncertain":
        result["message"] = ("Not confident enough to name a condition. Closest matches are listed; "
                             "retake the photo or consult an expert.")
    if with_gradcam:
        from gradcam import LAST_CONV_LAYER_NAME, make_gradcam_heatmap
        heatmap, _ = make_gradcam_heatmap(model, x[None], LAST_CONV_LAYER_NAME,
                                          pred_index=decision["pred_idx"])
        result["heatmap"] = heatmap
        result["overlay"] = gradcam_overlay(_as_uint8(x), heatmap)
        result["gradcam_note"] = GRADCAM_NOTE
    return result
