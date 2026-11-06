"""
Plant Leaf Health Assistant - Streamlit demo (course extension).

Run from the repo root:   streamlit run app/app.py

Uses the FP32 model (best_model.keras) and the shared pipeline in scripts/inference_utils.py:
quality gate -> prediction -> confidence threshold -> advisory -> Grad-CAM.
Decision support only; not a diagnosis.
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
MODEL_PATH = Path(os.environ.get("PLANT_MODEL_PATH", ROOT / "best_model.keras"))
REPORT_PATH = ROOT / "advisory" / "safeguards_report.json"
MAX_UPLOAD_MB = 10

st.set_page_config(page_title="Plant Leaf Health Assistant", layout="wide")
st.title("Plant Leaf Health Assistant")
st.caption("Upload a photo of one Potato, Tomato or Pepper leaf. The app checks the photo quality, "
           "identifies the condition if it is confident, and explains what to do next.")

st.warning("Validated on clean, lab-style leaf photos (92.8% test accuracy). In an initial check on "
           "in-the-wild web photos (PlantDoc, 1,098 images) accuracy fell to about 19%, and the "
           "confidence score did not flag the mistakes. Treat results on real field photos as "
           "unreliable and confirm with an extension officer.")

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}. Train the model (notebook 02 / scripts/train.py) or "
             "set the PLANT_MODEL_PATH environment variable to the .keras file.")
    st.stop()

import inference_utils as iu  # noqa: E402  (imports TensorFlow, so it comes after the model check)


@st.cache_resource(show_spinner="Loading model...")
def get_model(path):
    import tensorflow as tf
    return tf.keras.models.load_model(path)


@st.cache_data
def get_advisory():
    return iu.load_advisory()


def default_threshold():
    if REPORT_PATH.exists():
        try:
            return float(json.loads(REPORT_PATH.read_text())["threshold"])
        except (ValueError, KeyError):
            pass
    return 0.80


model = get_model(str(MODEL_PATH))
advisory = get_advisory()
problems = iu.validate_advisory(advisory)
if problems:
    st.error("Advisory table problem: " + "; ".join(problems))
    st.stop()
gate_cfg = iu.GateConfig.load()
calibrated = os.path.exists(iu.GATE_CONFIG_PATH)

with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Confidence needed to give an answer", 0.30, 0.99, default_threshold(), 0.01,
                          help="Below this the app says 'uncertain' instead of guessing.")
    st.caption("Photo-quality gate: " + ("calibrated on validation images." if calibrated
               else "provisional defaults (run notebook 05 to calibrate)."))
    st.caption("Known limits: 15 classes of Potato, Tomato and Pepper leaves, trained on lab-style photos. "
               "Other crops or non-leaf images are not recognised as such.")

uploaded = st.file_uploader("Leaf photo", type=["jpg", "jpeg", "png"])
if uploaded is None:
    st.info("Upload a clear, in-focus photo of a single leaf in daylight to begin.")
    st.stop()
if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
    st.error(f"File is larger than {MAX_UPLOAD_MB} MB. Please upload a smaller photo.")
    st.stop()

try:
    with st.spinner("Analysing..."):
        result = iu.analyze_image(model, uploaded, advisory=advisory, cfg=gate_cfg, threshold=threshold)
except Exception as exc:  # unreadable or unsupported image
    st.error(f"Could not read this image ({type(exc).__name__}). Try a different JPG or PNG photo.")
    st.stop()

left, right = st.columns(2)
with left:
    st.subheader("Your photo")
    st.image(uploaded, use_container_width=True)

if result["status"] == "rejected_quality":
    with right:
        st.subheader("Photo rejected")
        st.warning(result["message"])
else:
    with left:
        st.subheader("Where the model looked")
        st.image(result["overlay"], use_container_width=True)
        st.caption(result["gradcam_note"])
    with right:
        adv = result["advisory"]
        if result["status"] == "uncertain":
            st.subheader("Not sure")
            st.warning(result["message"])
        elif adv["is_healthy"]:
            st.subheader("Looks healthy")
            st.success(f"{adv['crop']}: no visible disease symptoms (confidence {result['confidence']:.0%}).")
        else:
            st.subheader(f"{adv['crop']}: {adv['condition']}")
            st.error(f"Likely {adv['category']} problem (confidence {result['confidence']:.0%}).")
        st.markdown("**Closest matches**")
        st.dataframe(pd.DataFrame(result["top3"], columns=["Class", "Probability"]).round(3),
                     hide_index=True, use_container_width=True)
        st.markdown("**What may be causing it**  \n" + adv["cause"])
        st.markdown("**What to do now**  \n" + adv["what_to_do_now"])
        if adv.get("prevention"):
            st.markdown("**How to prevent it**  \n" + adv["prevention"])
        st.markdown("**When to get expert help**  \n" + adv["when_to_seek_help"])
        if adv.get("source_name"):
            src = adv["source_name"] + (f" ({adv['source_url']})" if adv.get("source_url") else "")
            st.caption("Guidance source: " + src)

with st.expander("Photo-quality checks"):
    m = result["gate"]["metrics"]
    st.dataframe(pd.DataFrame({
        "check": ["sharpness (higher = sharper)", "noise level (lower = cleaner)", "brightness", "shortest side (px)"],
        "this photo": [round(m["sharpness"]), round(m["noise_sigma"], 1), round(m["brightness"]), m["min_side"]],
        "accepted range": [f">= {gate_cfg.min_sharpness:.0f}", f"<= {gate_cfg.max_noise_sigma:.1f}",
                           f"{gate_cfg.min_brightness:.0f} to {gate_cfg.max_brightness:.0f}", f">= {gate_cfg.min_side}"],
    }), hide_index=True, use_container_width=True)
st.caption(result["disclaimer"])
