# Course Extension: Safe, Explained Leaf-Disease Assistant

This document describes the additive extension built on top of the finished Plant Disease Detection
project. Nothing in notebooks 01-04, the existing scripts, the README or the existing tests was
modified. The extension reuses the trained FP32 model (`best_model.keras`) without retraining.

## Problem
A leaf-disease classifier that is 92.8% accurate on clean lab-style photos is not yet useful to a
farmer. It still has to decide what to do with blurry or noisy photos, uncertain predictions and
non-expert users who need to know what to do next.

## What the extension adds
| Piece | Where | Purpose |
|---|---|---|
| Input-quality gate | `scripts/inference_utils.py` | Rejects blurry, noisy, dark, bright or tiny photos before prediction |
| Confidence threshold | notebook 05, `inference_utils.decide` | Says "uncertain" instead of forcing a label; threshold chosen on the validation split only |
| Advisory table | `advisory/advisory_table.csv` | Plain-language cause, action, prevention and when to get help for all 15 classes |
| Grad-CAM overlay | reuses `scripts/gradcam.py` | Shows where the model looked |
| Safeguard analysis | `notebooks/05_...` | Calibration, corruption sweep, gate evaluation |
| End-to-end demo | `notebooks/06_...`, `app/app.py` | Full photo-to-advice flow, as notebook and Streamlit app |

Pipeline for one photo: decode (TensorFlow, same as training) -> quality gate -> FP32 model ->
confidence threshold -> advisory lookup -> Grad-CAM.

## How to run
From the repo root, in the existing environment:

1. `pip install -r app/requirements-app.txt`
2. Run notebook `05_Safeguards_Confidence_and_Quality_Gate.ipynb` top to bottom. It writes
   `advisory/gate_config.json` and `advisory/safeguards_report.json`.
3. Run notebook `06_Advisory_and_End_to_End_Demo.ipynb`.
4. Launch the app: `streamlit run app/app.py`

Tests: `pytest` (whole suite) or `pytest tests/test_project_state.py tests/test_inference_utils.py
tests/test_notebooks_valid.py` for the extension only.

## Results
Run notebook 05, then record here the values from `advisory/safeguards_report.json`: chosen
confidence threshold, validation and test ECE, share of clean test images answered and accuracy on
those answers, and the gate's rejection rate on clean versus degraded photos. Do not quote numbers
that are not in that file.

## Design decisions and honest limits
- **FP32 only.** The INT8 TFLite model scores 60.9% and is not used. Quantization-aware training is
  future work.
- **Corruption severity.** The committed robustness result (about 7.9%) uses the library-default
  `GaussNoise`, whose strength depends on the albumentations version. Notebook 05, section 4,
  measures how noisy that corruption actually is and compares it with a controlled sweep, so the
  result can be reported together with its severity.
- **No "not a leaf" class.** Photos of other crops or non-leaf objects can still receive a
  confident label if they pass the gate. The app states this limit.
- **Lab-style training data and a measured field gap.** The model was trained on plain-background
  photos. An initial check on in-the-wild PlantDoc photos (1,098 images mapped to 12 of the model's
  classes, evaluated in a sandbox) gave about 19% accuracy (95% CI 16.5-21.1%) against 92.8% on the
  lab test split. The confidence score did not separate right from wrong answers on those photos, and
  the quality gate rejected only about 5% of them. Notebook 07 must reproduce this on your machine
  before any figure is quoted.
- **Advisory guidance is general and unreviewed.** All 15 entries are marked with a `source_status`
  of `url_found_content_unreviewed` or `general_guidance_unreviewed`. Before submission, check each
  entry against its cited source and set the status to `reviewed`. The table deliberately gives no
  chemical names or doses; fungicide choice and timing are referred to local extension services.
- **Decision support, not diagnosis.** The app and the notebooks say so on every result.
- **Severity is not estimated.** The model has no severity signal, so none is shown.

## Future work
Quantization-aware training for an on-device model, an out-of-distribution or "not a leaf" class,
evaluation on field-condition photos, and crop-specific guidance reviewed by an agronomist.
