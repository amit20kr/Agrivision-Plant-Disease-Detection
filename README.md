# AgriVision CBM: Explainable Edge AI for Plant Disease Detection

A **Residual Concept Bottleneck Model** (CBM) that identifies 15 plant disease/healthy classes across Potato, Tomato, and Pepper — deployed offline via INT8 TFLite, explained via gradient-weighted Grad-CAM.

---

## The Problem

Agricultural AI models fail in the field for three consistent reasons:
1. **Background bias** — models memorise PlantVillage's controlled-lab backgrounds, not leaf pathology.
2. **Black-box outputs** — farmers cannot act on a diagnosis they cannot understand.
3. **Cloud dependency** — heavy server-based models are useless without rural internet.

---

## Architecture: 3-Track Design

### Track A — Robustness via Adversarial Augmentation

Training applies aggressive data augmentation (`CoarseDropout`, `HorizontalFlip`, `VerticalFlip`, `Rotate`, `RandomBrightnessContrast`) to force the model to learn lesion texture/shape rather than background. Robustness is measured on a **separately corrupted held-out test set** (MotionBlur + GaussNoise + brightness/contrast shift) that the model has never seen in any form.

| Metric | Value |
|---|---|
| **Clean test accuracy** (n=442) | **92.8%** |
| **Corrupted test accuracy** (Track A robustness proof) | **7.9%** |
| Accuracy drop under corruption | −84.9 pp |

> **Honest finding:** The corrupted-test accuracy (7.9%) is near random chance for a 15-class problem (6.7% chance). The training augmentation does **not** include the MotionBlur/GaussNoise family at this severity, so the model has not learned to be invariant to it. The Track A claim in the original project was built around the corrupted *test set pipeline* existing, but was never actually evaluated — this is the first run that closes that loop. If robust field performance is required, the training transform must be strengthened to include this corruption family.

### Track B — Explainable AI (Concept Bottleneck)

Instead of a single black-box output, the model predicts 4 intermediate agronomic symptom groups:

| Concept | Meaning |
|---|---|
| `is_healthy` | No visible pathology |
| `has_spots` | Bacterial/fungal spotting lesions |
| `is_blighted` | Blight-family disease (Early/Late) |
| `complex_pathogen` | Systemic/viral disease (Curl Virus, Mosaic, Mold) |

These concept predictions feed into the disease classification head (concatenated with backbone features). Gradient-weighted **Grad-CAM** (`scripts/gradcam.py`, verified layer: `activation_17`) generates class-discriminative heatmaps — they change with the predicted class, confirming they are actual gradient-based explanations and not a static activation average.

> **Honesty note:** The 4 concepts are a deterministic re-encoding of the 15 disease classes — every image of a given class gets an identical concept vector. This is a hierarchical/auxiliary-label training signal that improves supervision of the disease head, not a from-scratch Concept Bottleneck Model in the Koh et al. (2020) sense (where concepts are independently annotated and can vary within a class). The value is in the intermediate supervision and interpretable output layer, not in independently grounded concept annotations.

### Track C — Lightweight Edge Deployment

Post-training INT8 quantization via TensorFlow Lite for offline Android deployment.

| | FP32 (best_model.keras) | INT8 (edge_model.tflite) |
|---|---|---|
| **Size** | 7.34 MB | 0.79 MB |
| **Compression** | — | **9.3× smaller** |
| **Test accuracy** | 92.8% | 60.9% |
| **Accuracy delta** | — | **−31.9 pp** |

> **Honest finding:** INT8 quantization costs 31.9 accuracy points (92.8% → 60.9%). The compression ratio alone should not be the headline — the accuracy delta must be reported alongside it. If edge accuracy is a hard requirement, quantization-aware training (QAT) or a smaller FP32 baseline should be explored before shipping.

---

## Per-Class Performance (Clean Test Set, n=442)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Pepper__bell___Bacterial_spot | 1.000 | 0.867 | 0.929 |
| Pepper__bell___healthy | 0.938 | 1.000 | 0.968 |
| Potato___Early_blight | 1.000 | 1.000 | **1.000** |
| Potato___Late_blight | 1.000 | 0.933 | 0.966 |
| Potato___healthy | 1.000 | 1.000 | **1.000** |
| Tomato_Bacterial_spot | 0.893 | 0.833 | 0.862 |
| Tomato_Early_blight | 0.958 | 0.767 | **0.852** ⬇ |
| Tomato_Late_blight | 0.714 | 1.000 | **0.833** ⬇ |
| Tomato_Leaf_Mold | 0.906 | 0.967 | 0.935 |
| Tomato_Septoria_leaf_spot | 0.867 | 0.867 | 0.867 |
| Tomato_Spider_mites | 1.000 | 0.828 | 0.906 |
| Tomato__Target_Spot | 0.900 | 0.900 | 0.900 |
| Tomato__Tomato_YellowLeaf__Curl_Virus | 0.906 | 0.967 | 0.935 |
| Tomato__Tomato_mosaic_virus | 0.968 | 1.000 | 0.984 |
| Tomato_healthy | 1.000 | 1.000 | **1.000** |
| **Macro average** | **0.944** | **0.930** | **0.932** |

Weakest classes: `Tomato_Late_blight` (F1=0.833, low precision 0.714 — confused with Tomato_Early_blight) and `Tomato_Early_blight` (F1=0.852, low recall 0.767).

---

## Model Details

| Property | Value |
|---|---|
| Backbone | MobileNetV3Small (alpha=0.75, ImageNet pretrained) |
| Total parameters | ~2.5M |
| Training epochs | Up to 30, stopped early by `val_D_accuracy` (patience=3, `restore_best_weights=True`) |
| Best val accuracy | 93.7% (disease head) |
| Image size | 224×224 px |
| Batch size | 32 |
| Optimizer | Adam (lr=1e-4) |
| Regularization | L2(0.01) on both heads + Dropout(0.40) |
| Dataset | PlantVillage — 3 species, 15 classes, ≤200 images/class (undersampled) |
| Split | 70/15/15 stratified, reproducible seed=42, persisted in `manifest.csv` |

---

## Known Limitations

- **Corrupted-test collapse:** 7.9% accuracy under MotionBlur/GaussNoise — the model is not robust to this corruption family; the training augmentation must be expanded to include it before any field-deployment claim can be made.
- **INT8 accuracy gap:** 31.9 pp drop under quantization; QAT should be explored for any real edge deployment.
- **Dataset scale:** 2,951 images across 15 classes (after undersampling and one duplicate removed), lab-condition photos only — not a substitute for real field-photo validation.
- **Near-duplicate leakage:** one cross-split near-duplicate was detected (perceptual hash distance=2, `Tomato_Spider_mites` class) and removed before evaluation. See `duplicate_report.csv`.
- **Concept labels are class-derived:** see Track B note above.
- **No GPU on training machine:** trained on CPU (Windows, TF ≥ 2.11 does not support native Windows GPU). Results reflect CPU-trained weights.

---

## Repository Structure & How to Run

```
plant_disease_detection/
├── notebooks/
│   ├── 01_EDA_and_Data_Pipeline.ipynb       # raw scan → manifest → leakage check
│   ├── 02_Model_Training.ipynb              # tf.data + augmentation + EarlyStopping
│   ├── 03_Evaluation_Robustness_and_Quantization.ipynb  # clean/corrupted/INT8 eval
│   └── 04_GradCAM_Explainability.ipynb      # gradient-weighted Grad-CAM dashboard
├── scripts/
│   ├── data_pipeline.py    # single source of truth for the dataset split (manifest.csv)
│   ├── gradcam.py          # gradient-weighted Grad-CAM (layer: activation_17, verified)
│   ├── evaluate.py         # clean/corrupted/INT8 eval, JSON output
│   ├── check_duplicates.py # perceptual-hash near-duplicate leakage detector
│   ├── kaggle_download.py  # headless dataset download (no Colab files.upload())
│   ├── train.py            # standalone training script (mirrors Notebook 02)
│   └── requirements.txt
├── tests/                  # 24-test pytest suite covering all scripts
├── eval_report/
│   ├── eval_test.json         # clean test results (FP32)
│   └── eval_test_corrupt.json # corrupted test results (FP32)
├── manifest.csv            # reproducible train/val/test split (committed)
├── duplicate_report.csv    # perceptual-hash leakage report (1 pair found, resolved)
└── pytest.ini
```

### Setup

```bash
pip install -r scripts/requirements.txt
```

### Run the pretest suite first

Validates all pipeline mechanics against synthetic data before touching the real dataset:

```bash
pytest
# Expected: 24 passed, 7 skipped (skips become tests once real artifacts exist)
```

### Full pipeline

```bash
# 1. Download dataset (needs KAGGLE_USERNAME + KAGGLE_KEY env vars, or ~/.kaggle/kaggle.json)
python scripts/kaggle_download.py --dest data/raw

# 2. Build the manifest (single source of truth for all splits)
python scripts/data_pipeline.py --base_path data/raw/PlantVillage --out manifest.csv

# 3. Check for cross-split near-duplicate leakage
python scripts/check_duplicates.py --manifest manifest.csv --out duplicate_report.csv

# 4. Train with early stopping
python scripts/train.py

# 5. Evaluate: clean / corrupted / INT8 parity
python scripts/evaluate.py --model best_model.keras --manifest manifest.csv --split test
python scripts/evaluate.py --model best_model.keras --manifest manifest.csv --split test --corrupt
python scripts/quantize.py  # produces edge_model.tflite
python scripts/evaluate.py --model best_model.keras --manifest manifest.csv --split test --tflite edge_model.tflite
```

Or run the numbered notebooks (`notebooks/01` → `02` → `03` → `04`) in order for cell-by-cell output and visualizations.

### Model artifacts

`best_model.keras` (7.34 MB) and `edge_model.tflite` (0.79 MB) are stored locally in the project root. Attach to a GitHub Release or host via Git LFS to make them available to repo cloners without a full retrain.

---

## References

- Koh, P. W., et al. (2020). *Concept Bottleneck Models.* ICML. [arXiv:2007.04612](https://arxiv.org/abs/2007.04612)
- Howard, A., et al. (2019). *Searching for MobileNetV3.* ICCV.
- Selvaraju, R. R., et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV.
- PlantVillage Dataset: [Kaggle / emmarex/plantdisease](https://www.kaggle.com/datasets/emmarex/plantdisease)

---

## License

MIT — see `LICENSE` file (to be added).
