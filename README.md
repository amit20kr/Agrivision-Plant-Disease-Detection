# AgriVision CBM: Concept Bottleneck Model for Plant Disease Detection

AgriVision CBM is an open-source machine learning pipeline implementing a two-headed Residual Concept Bottleneck Model (MobileNetV3Small backbone, 4-unit concept head, and 15-unit disease head; model file size: 7.34 MB) trained on the PlantVillage laboratory dataset across Potato, Tomato, and Pepper crops.

### What it is and is not
- **What it is:** A research prototype and laboratory-trained classifier evaluating concept-supervised transfer learning, corrupted test set degradation, edge quantization tradeoffs, and in-the-wild generalization.
- **What it is NOT:** It is **NOT** a field-ready agricultural diagnostic tool. In real-world field evaluations, model accuracy drops to 18.9%, and confidence scores do not reliably separate target crops from other plant species.

---

## Results Snapshot

Every metric below is traceable directly to committed artifacts (`eval_report/eval_test.json`, `eval_report/eval_test_corrupt.json`, `eval_tflite.txt`, or `advisory/field_gap_report.json`):

| Evaluation Track | Metric | Value | Reference File / Source |
|---|---|---|---|
| **Clean Lab Test (FP32)** | Accuracy (n=442) | **92.8%** (410/442) | `eval_report/eval_test.json` |
| **Corrupted Lab Test** | Accuracy under blur/noise | **7.9%** (35/442) | `eval_report/eval_test_corrupt.json` |
| **Edge Quantization (INT8)** | Post-training TFLite accuracy | **60.9%** (−31.9 pp) | `eval_tflite.txt` |
| **Field Reality (PlantDoc)** | Overall accuracy (frozen model) | **18.9%** | `advisory/field_gap_report.json` |
| **Field Bootstrap 95% CI** | Empirical confidence interval | **[16.8%, 21.2%]** | `advisory/field_gap_report.json` |
| **Field Class Mapping** | Exact vs Approximate mapping | **21.8% vs 10.0%** | `advisory/field_gap_report.json` |
| **Field Calibration** | Expected Calibration Error (ECE) | **0.455** | `advisory/field_gap_report.json` |
| **Quality Gate Rejection** | Rejection rate on field photos | **6.8%** | `advisory/field_gap_report.json` |
| **OOD Other-Crop Leaves** | Confidently answered (conf ≥ 0.80) | **24.4%** | `advisory/field_gap_report.json` |
| **Feature Probe Baseline** | Linear probe 5-fold grouped CV | **35.0%** | `advisory/field_gap_report.json` |

> **Execution Note on Notebooks 01–04:** Results for 01–04 are in `eval_report/*.json` and `eval_*.txt`; notebook outputs are pending data download.

### Notebook 07 Section 24 Claims Table (Verbatim)

```
                           Metric          Value                         Source
    Lab Test Accuracy (Committed)          92.8%     eval_report/eval_test.json
    Lab Pipeline Control Accuracy          96.7%              nb07 control cell
    Field Accuracy (Frozen Model)          18.9% advisory/field_gap_report.json
  Field Accuracy 95% Bootstrap CI [16.8%, 21.2%] advisory/field_gap_report.json
          Exact vs Approx Mapping 21.8% vs 10.0% advisory/field_gap_report.json
                        Field ECE          0.455 advisory/field_gap_report.json
      Quality Gate Rejection Rate           6.8% advisory/field_gap_report.json
Other-Crop Leaf Confident Answers          24.4% advisory/field_gap_report.json
Linear Probe CV (Frozen Features)          35.0% advisory/field_gap_report.json
```

- A linear probe trained on PlantDoc reaches about 35%, so the frozen features carry some field signal.
- Confidence does not separate other-crop leaves from target crops (AUROC about 0.56): other plants still get an answer.

---

## Notebook Index & Execution Order

All project workflows are arranged sequentially in `notebooks/`:

1. **`01_EDA_and_Data_Pipeline.ipynb`**: Dataset scan, stratified 70/15/15 split generation (`manifest.csv`), and near-duplicate leakage detection.
2. **`02_Model_Training.ipynb`**: Training recipe for the two-headed Residual Concept Bottleneck Model. (Status: training recipe; training log was not preserved and this notebook is intentionally not re-executed so `best_model.keras` is never overwritten).
3. **`03_Evaluation_Robustness_and_Quantization.ipynb`**: Clean test evaluation (92.8%), synthetic corruption benchmark (7.9%), and post-training INT8 TFLite export (60.9%).
4. **`04_GradCAM_Explainability.ipynb`**: Class-discriminative gradient-weighted Grad-CAM visualization at layer `activation_17`.
5. **`05_Safeguards_Confidence_and_Quality_Gate.ipynb`**: Input quality gate screening (blur, brightness, noise), confidence filtering, and calibration diagnostics.
6. **`06_Advisory_and_End_to_End_Demo.ipynb`**: Agronomic advisory table mapping, treatment lookups, and end-to-end interactive diagnostic demonstration.
7. **`07_Field_Reality_Check.ipynb`**: Evaluation on in-the-wild Cropped-PlantDoc data, out-of-distribution negative crop screening, linear probe baseline, and gap report generation.

**Execution Order:** `01` → `02` (recipe only, skip execution) → `03` → `04` → `05` → `06` → `07`.
*(Note: Results for 01–04 are in eval_report/*.json and eval_*.txt; notebook outputs are pending data download.)*

---

## Safeguards & Advisory (Notebooks 05 & 06)

Notebook 05 introduces an operational quality gate and confidence safeguard layer before predictions reach downstream users:
- **Quality Gate (`advisory/gate_config.json`):** Evaluates input images against configured thresholds: minimum dimension (`min_side`: 100 px), minimum sharpness (`min_sharpness`: 44.75), maximum Gaussian noise sigma (`max_noise_sigma`: 19.09), minimum brightness (`min_brightness`: 23.47), and maximum brightness (`max_brightness`: 174.55).
- **Confidence Filtering & Rejection:** Rejects low-confidence inferences to reduce unvetted erroneous recommendations.
- **Agronomic Advisory Table (`advisory/advisory_table.csv`):** Maps conditions to structured guidance containing: `crop`, `condition`, `is_healthy`, `category`, `cause`, `what_to_do_now`, `prevention`, `when_to_seek_help`, `source_name`, `source_url`, and `source_status`. Provides general cultural management steps and safety disclaimers (no specific chemical dosages, brand names, or severity estimates).

> **Status Notice:** Calibrated on a partial copy of the data (cloud-sync gap); re-calibration on the full data is pending.

---

## Streamlit Dashboard

The interactive web dashboard (`app/dashboard.py`) provides an operational interface with four dedicated views:
- **Single Leaf Diagnosis & Explainability:** Supports direct image upload or preset sample selection; computes top-3 predictions with confidence scores and margin to rank-2; evaluates input quality against the quality gate; renders interactive Grad-CAM heatmap overlays with adjustable blend opacity and a radio selector to switch the explained class; and displays the corresponding advisory guidance card.
- **Batch Processing:** Allows multi-image upload with real-time inference and downloadable CSV summary including quality gate flags and top predictions.
- **Lab Performance:** Displays committed test set benchmarks, confusion matrix analysis, and corruption robustness comparisons.
- **Field Reality:** Displays in-the-wild PlantDoc benchmark metrics, out-of-distribution rejection statistics, and the 6 analytical figures.

### Running Locally
```powershell
$env:KERAS_HOME = "$pwd\cache\keras"
streamlit run app/dashboard.py
```

### Cloud Deployment
- **Live Deployment URL:** <ADD AFTER DEPLOY>
- Step-by-step instructions for Streamlit Community Cloud and Hugging Face Spaces are documented in [DEPLOY.md](DEPLOY.md).

---

## Project History

- **Version 1 (v1-baseline):** Initial course baseline repository including dataset split (`manifest.csv`), trained model (`best_model.keras`), clean/corrupted test evaluation, INT8 quantization, and notebooks 01–04. Link to earlier repository: <OLD REPO URL>.
- **Version 2 (v2):** Maintains the identical model weights (7.34 MB), dataset split, and baseline numbers, while introducing operational quality gating, agronomic advisory lookup, in-the-wild field reality evaluation (PlantDoc), the interactive Streamlit dashboard, and an automated pytest suite. v2 adds rigorous empirical validation, not new performance claims.

---

## Known Limitations & Future Work

1. **Field Adaptation:** Retrain or fine-tune backbone representations jointly on PlantVillage and in-the-wild field datasets (e.g. PlantDoc) to close the field gap.
2. **Confidence & OOD Calibration:** Confidence signals currently fail to detect other-crop leaves (AUROC ~0.56); dedicated out-of-distribution heads or Mahalanobis distance filters are needed.
3. **Edge Optimization:** Post-training INT8 quantization incurs a 31.9 pp degradation; quantization-aware training (QAT) or float16 TFLite should be explored.
4. **Notebook 05 Re-baseline:** Re-calibrate quality gate thresholds on the fully hydrated dataset.
5. **Training Logging:** Ensure future training passes persist complete epoch history logs alongside model checkpoints.

---

## Repository Structure

```
plant_disease_detection/
├── .streamlit/
│   └── config.toml                  # Streamlit configuration
├── advisory/
│   ├── advisory_table.csv           # Agronomic guidance lookup
│   ├── gate_config.json             # Input quality gate parameters
│   ├── safeguards_report.json       # Safeguard diagnostic outputs
│   ├── plantdoc_class_map.csv       # PlantDoc to PlantVillage class mapping
│   ├── field_gap_report.json        # PlantDoc field evaluation results
│   └── field_figures/               # 6 analysis figures from notebook 07
├── app/
│   ├── dashboard.py                 # Multi-tab Streamlit dashboard
│   ├── requirements.txt             # Pinned dashboard deployment requirements
│   ├── samples/                     # 6 preset evaluation images
│   └── demo_samples/                # Preset evaluation images
├── notebooks/
│   ├── 01_EDA_and_Data_Pipeline.ipynb
│   ├── 02_Model_Training.ipynb
│   ├── 03_Evaluation_Robustness_and_Quantization.ipynb
│   ├── 04_GradCAM_Explainability.ipynb
│   ├── 05_Safeguards_Confidence_and_Quality_Gate.ipynb
│   ├── 06_Advisory_and_End_to_End_Demo.ipynb
│   └── 07_Field_Reality_Check.ipynb
├── scripts/
│   ├── data_pipeline.py             # Stratified manifest generation
│   ├── train.py                     # Standalone CBM training recipe
│   ├── evaluate.py                  # Evaluation on clean, corrupted, or TFLite data
│   ├── quantize.py                  # INT8 post-training quantization
│   ├── gradcam.py                   # Grad-CAM heatmap generation (activation_17)
│   ├── check_duplicates.py          # Perceptual hash near-duplicate check
│   ├── dashboard_utils.py           # Dashboard utilities and Grad-CAM blending
│   ├── field_analysis.py            # PlantDoc evaluation, OOD AUROC, linear probe
│   ├── field_evaluation.py          # Field metrics support
│   ├── make_demo_samples.py         # Sample preset preparation
│   └── requirements.txt             # Development environment dependencies
├── tests/                           # Automated pytest suite
├── eval_report/
│   ├── eval_test.json               # Committed clean test results (FP32)
│   └── eval_test_corrupt.json       # Committed corrupted test results
├── eval_tflite.txt                  # INT8 evaluation summary
├── best_model.keras                 # Frozen trained model (7.34 MB)
├── manifest.csv                     # Stratified dataset split specification
├── duplicate_report.csv             # Near-duplicate scan results
├── DEPLOY.md                        # Deployment runbook
├── ANTIGRAVITY_NB07_DASHBOARD.md    # Notebook 07 & dashboard runbook
└── README.md
```

### Running Tests
The automated test suite covers pipeline mechanics, data integrity, model loading, dashboard utilities, and field analysis:
```powershell
$env:KERAS_HOME = "$pwd\cache\keras"
pytest tests/ -q
```

---

## License

MIT, LICENSE file to be added.
