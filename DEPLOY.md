# Streamlit Dashboard Deployment Guide

This guide covers deploying the Plant Disease Diagnosis & Reliability Dashboard (`app/dashboard.py`).

## 1. Prerequisites & Repo Structure

Ensure your repository has the following files committed:
- `app/dashboard.py` (entrypoint)
- `app/requirements.txt`
- `best_model.keras` (whitelisted via `!best_model.keras` in `.gitignore`, ~7.7 MB)
- `advisory/advisory_table.csv`
- `advisory/gate_config.json`
- `advisory/safeguards_report.json`
- `advisory/field_gap_report.json` (optional, created by notebook 07)
- `scripts/dashboard_utils.py`, `scripts/gradcam.py`, `scripts/inference_utils.py`, `scripts/data_pipeline.py`

---

## 2. Deploying to Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and connect your GitHub repository.
2. Configure application settings:
   - **Repository**: `your-username/plant_disease_detection`
   - **Branch**: `main` (or your deployment branch)
   - **Main file path**: `app/dashboard.py`
3. **Advanced Settings (Critical)**:
   - **Python Version**: Select **Python 3.12** (Community Cloud may default to 3.9/3.10 which can cause TensorFlow wheel conflicts).
   - Memory limit: Community Cloud has a 2.7 GB RAM cap. The model is lightweight (~7.7 MB FP32), and headless OpenCV is used to stay well below the ceiling.

---

## 3. Fallback: Deploying to Hugging Face Spaces

If Community Cloud memory limits kill the TensorFlow process:
1. Create a new Space on [Hugging Face](https://huggingface.co/spaces) selecting the **Streamlit** SDK.
2. In the Space repository settings:
   - Place `requirements.txt` in the root (copy from `app/requirements.txt`).
   - Place an `app.py` in the root that imports or runs `app/dashboard.py`.
3. Standard free CPU tier provides 16 GB RAM, which easily runs TensorFlow inference and Grad-CAM without memory pressure.

---

## 4. Local Testing

Run locally in your terminal from the repo root:
```bash
streamlit run app/dashboard.py
```
Open `http://localhost:8501` in your browser.
