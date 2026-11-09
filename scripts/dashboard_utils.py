"""
Dashboard utility functions for Plant Disease Detection Streamlit app.
"""
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf

import data_pipeline as dp
import gradcam
import inference_utils as iu


ROOT = Path(__file__).resolve().parent.parent


def load_model(model_path: Union[str, Path] = "best_model.keras") -> tf.keras.Model:
    """Load the trained Keras model."""
    path = Path(model_path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}")
    return tf.keras.models.load_model(str(path))


def load_advisory_table(table_path: Union[str, Path] = "advisory/advisory_table.csv") -> pd.DataFrame:
    """Load treatment advisory table."""
    path = Path(table_path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(str(path))


def load_advisory_merged(
    table_path: Union[str, Path] = "advisory/advisory_table.csv",
    details_path: Union[str, Path] = "advisory/advisory_details.csv",
) -> pd.DataFrame:
    """Load and left-join advisory_table with advisory_details on class_name.

    Returns a merged DataFrame. Rows with no matching detail entry will have
    NaN in symptom/treatment columns — callers should check for that and show
    'No advisory entry for this class'.
    """
    t_path = Path(table_path)
    if not t_path.is_absolute():
        t_path = ROOT / t_path
    d_path = Path(details_path)
    if not d_path.is_absolute():
        d_path = ROOT / d_path

    if not t_path.exists():
        return pd.DataFrame()

    table = pd.read_csv(str(t_path))
    if not d_path.exists():
        import warnings
        warnings.warn(f"advisory_details.csv not found at {d_path}; symptom/treatment columns will be missing.")
        return table

    details = pd.read_csv(str(d_path))
    merged = table.merge(details, on="class_name", how="left", suffixes=("", "_detail"))

    # Warn about any class missing from details
    missing = merged[merged["symptoms"].isna()]["class_name"].tolist()
    if missing:
        import warnings
        warnings.warn(f"No advisory_details row for: {missing}")

    return merged


def load_gate_config(config_path: Union[str, Path] = "advisory/gate_config.json") -> iu.GateConfig:
    """Load quality gate configuration."""
    path = Path(config_path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        return iu.GateConfig()
    return iu.GateConfig.load(str(path))


def load_safeguards_report(report_path: Union[str, Path] = "advisory/safeguards_report.json") -> Optional[Dict[str, Any]]:
    """Load safeguards report tolerantly handling NaNs or nulls."""
    path = Path(report_path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        return None
    try:
        raw = path.read_text(encoding="utf-8")
        # Replace unquoted NaN with null if present
        clean_raw = raw.replace(": NaN", ": null").replace(":NaN", ":null")
        return json.loads(clean_raw)
    except Exception:
        return None


def load_field_gap_report(report_path: Union[str, Path] = "advisory/field_gap_report.json") -> Optional[Dict[str, Any]]:
    """Load field gap report if it exists."""
    path = Path(report_path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def validate_image_upload(
    file_bytes: bytes,
    filename: str = "upload.jpg",
    max_size_mb: float = 10.0,
    max_megapixels: float = 25.0,
) -> Tuple[bool, str, Optional[np.ndarray]]:
    """Validate format (JPG/PNG), file size limit, and image dimensions before decoding.
    Returns (is_valid, error_msg, rgb_uint8)."""
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > max_size_mb:
        return False, f"File size ({size_mb:.1f} MB) exceeds limit of {max_size_mb} MB.", None

    lower_name = filename.lower()
    if not (lower_name.endswith(".jpg") or lower_name.endswith(".jpeg") or lower_name.endswith(".png")):
        return False, "Unsupported format. Please upload a JPG or PNG image.", None

    try:
        pil_img = Image.open(io.BytesIO(file_bytes))
        width, height = pil_img.size
        megapixels = (width * height) / 1_000_000.0
        if megapixels > max_megapixels:
            return False, f"Resolution ({megapixels:.1f} MP) exceeds {max_megapixels} MP limit.", None

        # Convert to uint8 RGB
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        rgb = np.array(pil_img, dtype=np.uint8)
        return True, "", rgb
    except Exception as e:
        return False, f"Corrupted or invalid image: {e}", None


def get_top_predictions(
    model: tf.keras.Model,
    rgb_uint8: np.ndarray,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """Get top K disease predictions with probabilities and margins."""
    x = iu.resize_for_model(rgb_uint8)
    batch = np.expand_dims(x, axis=0)
    outputs = model(batch, training=False)
    probs = outputs[1].numpy()[0]  # disease head output

    sorted_indices = np.argsort(probs)[::-1]
    top_preds = []
    for rank, idx in enumerate(sorted_indices[:top_k]):
        class_name = dp.SELECTED_CLASSES[idx]
        conf = float(probs[idx])
        top_preds.append({
            "rank": rank + 1,
            "class_idx": int(idx),
            "class_name": class_name,
            "confidence": conf,
        })
    return top_preds


def generate_gradcam(
    model: tf.keras.Model,
    rgb_uint8: np.ndarray,
    target_class_idx: int,
    layer_name: str = "activation_17",
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate Grad-CAM heatmap and blended overlay for a given class index.
    Returns (heatmap_uint8, overlay_uint8)."""
    x = iu.resize_for_model(rgb_uint8)
    batch = np.expand_dims(x, axis=0)
    heatmap, _ = gradcam.make_gradcam_heatmap(model, batch, layer_name, pred_index=target_class_idx)

    # Resize heatmap to input image size
    h, w = rgb_uint8.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(rgb_uint8, 0.6, heatmap_rgb, 0.4, 0)
    return heatmap_rgb, overlay


def compute_blend(image_rgb: np.ndarray, heatmap_rgb: np.ndarray, alpha: float) -> np.ndarray:
    """Blend image and heatmap with custom alpha opacity (0.0 = only image, 1.0 = only heatmap)."""
    alpha = max(0.0, min(1.0, float(alpha)))
    return cv2.addWeighted(heatmap_rgb, alpha, image_rgb, 1.0 - alpha, 0)


def process_batch(
    model: tf.keras.Model,
    files: List[Tuple[str, bytes]],
    cfg: Optional[iu.GateConfig] = None,
    threshold: float = 0.30,
) -> pd.DataFrame:
    """Run batch prediction on up to 12 files."""
    cfg = cfg or iu.GateConfig()
    results = []
    for fname, data in files[:12]:
        valid, err, rgb = validate_image_upload(data, filename=fname)
        if not valid or rgb is None:
            results.append({
                "Filename": fname,
                "Status": f"REJECTED: {err}",
                "Gate_Passed": False,
                "Prediction": "N/A",
                "Confidence": 0.0,
                "Action": "Re-take photo",
            })
            continue

        min_side = min(rgb.shape[:2])
        x = iu.resize_for_model(rgb)
        q = iu.image_quality_metrics(iu._as_uint8(x))
        passed, reasons = iu.evaluate_gate(q, min_side, cfg)

        top_preds = get_top_predictions(model, rgb, top_k=1)
        top = top_preds[0]
        conf = top["confidence"]
        pred_class = top["class_name"]

        if not passed:
            status = f"GATE_REJECT ({', '.join(reasons)})"
            action = "Re-take photo (poor quality)"
        elif conf < threshold:
            status = f"UNCERTAIN (conf {conf:.2f} < {threshold:.2f})"
            action = "Inspect manually / consult expert"
        else:
            status = "ACCEPTED"
            action = "See advisory for " + pred_class

        results.append({
            "Filename": fname,
            "Status": status,
            "Gate_Passed": passed,
            "Prediction": pred_class,
            "Confidence": round(conf, 4),
            "Action": action,
        })
    return pd.DataFrame(results)
