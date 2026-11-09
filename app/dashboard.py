"""
Plant Disease Detection & Field Reality Dashboard
Streamlit application for model inference, Grad-CAM explainability, batch scoring,
lab safeguards audit, and real-world domain gap reporting.
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# Configure repo path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import dashboard_utils as du
import inference_utils as iu

st.set_page_config(
    page_title="Plant Disease Detection Dashboard",
    page_icon="🌿",
    layout="wide",
)

st.title("🌿 Plant Disease Diagnosis & Reliability Dashboard")
st.caption("Interpretable Concept Bottleneck Model with Input Quality Safeguards & Field Domain Evaluation")


@st.cache_resource
def get_cached_model():
    return du.load_model()


@st.cache_data
def get_cached_advisory():
    return du.load_advisory_table()


@st.cache_data
def get_cached_gate_config():
    return du.load_gate_config()


try:
    model = get_cached_model()
    advisory_df = get_cached_advisory()
    gate_cfg = get_cached_gate_config()
except Exception as e:
    st.error(f"Failed to load essential model or config: {e}")
    st.stop()

# Tabs
tab_classify, tab_batch, tab_perf, tab_field = st.tabs([
    "🔍 Classify & Explain",
    "📁 Batch Processing",
    "📊 Lab Performance",
    "🌍 Field Reality (PlantDoc)",
])

# ── TAB 1: CLASSIFY & EXPLAIN ────────────────────────────────────────────────
with tab_classify:
    st.header("Single Leaf Diagnosis & Explainability")
    col_input, col_view = st.columns([1, 2])

    with col_input:
        st.subheader("1. Provide an Image")
        source_mode = st.radio("Choose input source:", ["Upload Photo", "Pick Sample Preset"], horizontal=True)

        selected_bytes = None
        selected_name = "upload.jpg"

        if source_mode == "Upload Photo":
            uploaded_file = st.file_uploader("Upload leaf photo (JPG/PNG, max 10MB, max 25MP)", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                selected_bytes = uploaded_file.read()
                selected_name = uploaded_file.name
        else:
            demo_dir = ROOT / "app" / "samples"
            if not demo_dir.exists() or not list(demo_dir.glob("*.jpg")):
                demo_dir = ROOT / "app" / "demo_samples"
            samples = sorted(list(demo_dir.glob("*.jpg"))) if demo_dir.exists() else []
            if samples:
                sample_map = {p.stem.replace("_", " "): p for p in samples}
                choice = st.selectbox("Choose a sample preset:", list(sample_map.keys()))
                if choice:
                    selected_path = sample_map[choice]
                    selected_bytes = selected_path.read_bytes()
                    selected_name = selected_path.name
            else:
                st.info("No sample presets found. Run `scripts/make_demo_samples.py`.")

        confidence_threshold = st.slider(
            "Confidence Decision Threshold",
            min_value=0.10,
            max_value=0.95,
            value=0.30,
            step=0.05,
            help="Images below this threshold are flagged as UNCERTAIN.",
        )

    with col_view:
        if selected_bytes is not None:
            valid, err_msg, rgb_img = du.validate_image_upload(selected_bytes, filename=selected_name)
            if not valid or rgb_img is None:
                st.error(f"❌ Invalid Image: {err_msg}")
            else:
                # Run Quality Gate
                min_side = min(rgb_img.shape[:2])
                x_norm = iu.resize_for_model(rgb_img)
                q_metrics = iu.image_quality_metrics(iu._as_uint8(x_norm))
                passed, reasons = iu.evaluate_gate(q_metrics, min_side, gate_cfg)

                if passed:
                    st.success("✅ **Quality Gate: PASSED** (Sharpness, brightness, and noise within acceptable bounds)")
                else:
                    st.warning(f"⚠️ **Quality Gate: REJECTED** ({', '.join(reasons)}) — Field diagnosis may be unreliable.")

                # Predictions
                top_preds = du.get_top_predictions(model, rgb_img, top_k=3)
                top1 = top_preds[0]

                st.subheader("2. Diagnosis Results")
                metric_c1, metric_c2, metric_c3 = st.columns(3)
                metric_c1.metric("Predicted Disease", top1["class_name"].replace("_", " "))
                metric_c2.metric("Top Confidence", f"{top1['confidence']*100:.1f}%")
                status_label = "Confirmed" if (passed and top1["confidence"] >= confidence_threshold) else (
                    "Uncertain" if passed else "Quality Rejected"
                )
                metric_c3.metric("Status", status_label)

                # Grad-CAM selection
                st.subheader("3. Visual Explainability (Grad-CAM)")
                class_options = [f"{p['rank']}. {p['class_name']} ({p['confidence']*100:.1f}%)" for p in top_preds]
                selected_cam_str = st.radio(
                    "Select class to compute Grad-CAM heatmap for:",
                    class_options,
                    horizontal=True,
                )
                selected_rank = int(selected_cam_str.split(".")[0])
                target_class_idx = top_preds[selected_rank - 1]["class_idx"]

                alpha = st.slider("Heatmap Overlay Opacity", min_value=0.0, max_value=1.0, value=0.45, step=0.05)

                heatmap_rgb, base_overlay = du.generate_gradcam(model, rgb_img, target_class_idx=target_class_idx)
                custom_overlay = du.compute_blend(rgb_img, heatmap_rgb, alpha=alpha)

                img_c1, img_c2, img_c3 = st.columns(3)
                img_c1.image(rgb_img, caption="Original Input Photo", use_container_width=True)
                img_c2.image(heatmap_rgb, caption=f"Grad-CAM Heatmap ({top_preds[selected_rank - 1]['class_name']})", use_container_width=True)
                img_c3.image(custom_overlay, caption=f"Blended Overlay (opacity: {alpha:.2f})", use_container_width=True)

                # Treatment Advisory
                st.subheader("4. Agronomic Advisory")
                if not advisory_df.empty and "class_name" in advisory_df.columns:
                    adv_row = advisory_df[advisory_df["class_name"] == top1["class_name"]]
                    if not adv_row.empty:
                        rec = adv_row.iloc[0]
                        st.markdown(f"**Condition:** {rec.get('condition_name', top1['class_name'])}")
                        st.markdown(f"**Symptoms:** {rec.get('symptoms', 'N/A')}")
                        st.markdown(f"**Organic Treatment:** {rec.get('organic_treatment', 'N/A')}")
                        st.markdown(f"**Chemical Treatment:** {rec.get('chemical_treatment', 'N/A')}")
                    else:
                        st.info(f"Healthy plant or standard monitoring recommended for {top1['class_name']}.")
                else:
                    st.info("Advisory table is loading or healthy baseline.")
        else:
            st.info("👈 Upload an image or select a sample preset from the left panel to begin.")

# ── TAB 2: BATCH PROCESSING ──────────────────────────────────────────────────
with tab_batch:
    st.header("Batch Leaf Scoring")
    st.write("Upload up to 12 leaf images for automated quality gating and batch disease screening.")

    batch_files = st.file_uploader(
        "Upload images (max 12)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if batch_files:
        if len(batch_files) > 12:
            st.warning("Only the first 12 images will be processed.")
        pairs = [(f.name, f.read()) for f in batch_files[:12]]
        with st.spinner("Processing batch..."):
            batch_df = du.process_batch(model, pairs, cfg=gate_cfg)
        st.dataframe(batch_df, use_container_width=True)

        csv_data = batch_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Batch Results (CSV)",
            data=csv_data,
            file_name="batch_disease_predictions.csv",
            mime="text/csv",
        )

# ── TAB 3: MODEL PERFORMANCE (LAB) ──────────────────────────────────────────
with tab_perf:
    st.header("Committed Model Evaluation (Lab Dataset)")
    st.markdown("""
    **Clean Test Accuracy**: **92.8%** (410 / 442 correctly classified on full committed test split).
    *Note:* The 5-image gap between `eval_test.json` (92.8%) and preliminary runs was verified to stem from on-demand cloud sync during earlier execution.
    """)

    safeguards = du.load_safeguards_report()
    if safeguards:
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Committed Accuracy", "92.8%")
        col_m2.metric("Calibrated Threshold", f"{safeguards.get('threshold', 0.30):.2f}")
        col_m3.metric("Test ECE", f"{safeguards.get('test_ece', 0.034):.4f}")
        col_m4.metric("Answered Coverage", f"{safeguards.get('clean_test_answered_share', 0.98)*100:.1f}%")

        if "gate_report" in safeguards and safeguards["gate_report"]:
            st.subheader("Corruption Sweep & Quality Gate Stress Testing")
            sweep_table = pd.DataFrame(safeguards["gate_report"])
            st.dataframe(sweep_table, use_container_width=True)
    else:
        st.info("Safeguards report not found. Run notebook 05 to view full calibration figures.")

# ── TAB 4: FIELD REALITY (PLANTDOC) ──────────────────────────────────────────
with tab_field:
    st.header("Real-World Field Reality Check (PlantDoc)")
    field_report = du.load_field_gap_report()

    if field_report is None:
        st.info("⚠️ **Field gap report not yet generated.** Run `notebooks/07_Field_Reality_Check.ipynb` to evaluate against the real PlantDoc dataset.")
    else:
        pipeline = field_report.get("pipeline_control", {})
        repro = field_report.get("reproduction_gate", {})
        targets = field_report.get("field_target_metrics", {})
        negatives = field_report.get("out_of_distribution_negatives", {})

        c1, c2, c3 = st.columns(3)
        c1.metric("Lab Control Accuracy", f"{pipeline.get('lab_control_accuracy', 0.0)*100:.1f}%", f"Status: {pipeline.get('status', 'N/A')}")
        c2.metric("Field Target Accuracy", f"{targets.get('overall_accuracy', 0.0)*100:.1f}%", f"Gate: {repro.get('status', 'N/A')}")
        c3.metric("Field Calibration ECE", f"{targets.get('ece', 0.0):.3f}")

        st.subheader("Field Metrics Summary")
        st.markdown(f"""
        - **Overall Field Accuracy**: **{targets.get('overall_accuracy', 0.0)*100:.1f}%** (95% CI: [{targets.get('ci_95', [0, 0])[0]*100:.1f}%, {targets.get('ci_95', [0, 0])[1]*100:.1f}%])
        - **Exact vs Approximate Mapping**: {targets.get('exact_accuracy', 0.0)*100:.1f}% vs {targets.get('approx_accuracy', 0.0)*100:.1f}%
        - **Quality Gate Rejection on Field**: {targets.get('gate_rejection_rate', 0.0)*100:.1f}%
        - **Other-Crop Leaves Confidently Answered**: {negatives.get('other_crop_answered_at_08', 0.0)*100:.1f}%
        - **AUROC (Confidence / Margin / Entropy)**: {negatives.get('auroc_confidence', 0.5):.2f} / {negatives.get('auroc_margin', 0.5):.2f} / {negatives.get('auroc_entropy', 0.5):.2f}
        """)

        # Display figures if available
        fig_dir = ROOT / "advisory" / "field_figures"
        if fig_dir.exists():
            st.subheader("Field Reality Visualizations")
            figures = sorted(list(fig_dir.glob("*.png")))
            for i in range(0, len(figures), 2):
                cols = st.columns(2)
                cols[0].image(str(figures[i]), use_container_width=True)
                if i + 1 < len(figures):
                    cols[1].image(str(figures[i + 1]), use_container_width=True)
