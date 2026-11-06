# Project Walkthrough — Plant Disease Detection: From Hackathon to Industry Grade

> **Who this is for:** You (Amit), to explain what was done and why — so you can answer any interview question about this project with confidence.

---

## The Starting Point: What Was Wrong

Before any of this work, the project had 8 serious problems that would fail a technical interview:

1. **The headline metric was a lie** — the README advertised 88.8% accuracy, but that was *training* accuracy. The model had never been evaluated on data it hadn't seen. Validation accuracy was 67% and *still falling*.
2. **The "robustness" test was built but never run** — the code to corrupt images with blur and noise existed, but `model.evaluate()` was never called on it. The Track A claim had zero evidence.
3. **Grad-CAM was mislabeled** — the code averaged raw feature map activations. That's *not* Grad-CAM. Real Grad-CAM uses *gradients* — the original code produced the same heatmap no matter what disease was predicted.
4. **INT8 model accuracy was never checked** — the project proudly claimed "9.4× compression" but never verified whether the compressed model still made accurate predictions.
5. **No one could reproduce the results** — no `requirements.txt`, no committed model files, and the inference notebook required a manual Colab file-upload dialog.
6. **The "Concept Bottleneck" claim was misleading** — the 4 "concepts" (healthy, spots, blight, complex pathogen) were just a 1-to-1 repackaging of the 15 class labels. Every Tomato Early Blight image got the exact same concept vector. That's not independently annotated concepts — it's just a fancy label.
7. **No early stopping** — training ran for exactly 12 epochs regardless of whether the model was still improving or had started overfitting. The saved model was the *last* epoch's weights, not the *best* epoch's weights.
8. **Potential data leakage** — PlantVillage is known to contain near-duplicate photos of the same leaf. A random train/test split could accidentally put nearly-identical images in both training and the test set, making accuracy look better than it is.

---

## Phase 0: Pre-Test Gate (Verify the Fix Machinery Before Using It)

**What we did:** Before touching the real dataset or running any GPU training, we ran the project's automated test suite (`pytest`) against synthetic (fake) data.

**Why:** If the scripts themselves had bugs, no amount of real data would fix that. The test suite was designed to catch exactly these bugs on tiny dummy data, without needing Kaggle credentials or a GPU.

**Command run:**
```
pip install -r scripts/requirements.txt
pytest
```

**What happened:**
- We hit one dependency version error: `opencv-python-headless==4.13.0.88` didn't exist. Fixed the `requirements.txt` to use `>=4.12` instead.
- After that fix: **24 tests passed, 7 skipped** (the 7 skips were intentional — they only run once real model files exist).

**Think of it like:** A car mechanic checking that all their wrenches are working before they touch your engine.

---

## Phase 1: Download the Real Dataset

**What we did:** Downloaded the PlantVillage dataset from Kaggle automatically — no manual upload dialogs, no browser.

**Why:** The original project used `google.colab.files.upload()` which pops up a manual "upload a file" dialog. This only works inside Google Colab, and only if a human is sitting there clicking. That's unusable for any automated or reproducible pipeline.

**The fix:** `scripts/kaggle_download.py` reads your Kaggle API credentials from an environment variable (`KAGGLE_USERNAME`, `KAGGLE_KEY`) and calls the Kaggle CLI silently.

**What happened:**
- Dataset downloaded and extracted to `data/raw/PlantVillage/`
- 16 class folders found (15 disease/healthy classes + 1 nested subfolder)

**Think of it like:** Teaching a robot to shop for groceries on its own instead of waiting for you to hand it each item.

---

## Phase 2: Build the Manifest (Single Source of Truth for Splits)

**What we did:** Scanned all 20,637 images across 15 classes, capped each class at 200 images (undersampling to prevent majority-class dominance), then split into 70% train / 15% val / 15% test using stratified sampling. Wrote the result to `manifest.csv`.

**Why this matters:** The original project computed `X_train`, `X_val`, `X_test` as in-memory Python lists inside the training notebook — they were never saved. The inference notebook had no idea which images were in the test set. If you ran the split twice with a different random state, you'd get different results and couldn't compare them. Now everything reads from one CSV file with a fixed `seed=42`.

**What happened:**
```
Found 20,637 raw images across 15 classes
Wrote manifest with 2,952 rows to manifest.csv
split
test      443
train    2066
val       443
```

**What "stratified" means:** If 5% of your images are `Potato___healthy`, then 5% of your train set, 5% of your val set, and 5% of your test set are also `Potato___healthy`. The class proportions are preserved in every split.

**Think of it like:** Dividing a deck of 15 different card types so that every player gets the same *proportion* of each type, not just random cards.

---

## Phase 3: Check for Data Leakage (Near-Duplicate Detection)

**What we did:** Ran a perceptual hash algorithm on every image. A "perceptual hash" is a short fingerprint of an image that stays similar even if the image is slightly compressed or recolored. If two images in *different* splits had hashes within distance 5 of each other, they were flagged as near-duplicates (potential leakage).

**Why:** PlantVillage is known to contain multiple photos of the same leaf from slightly different angles. If a near-duplicate of a training image ends up in the test set, the model gets a "free point" — it's essentially being tested on something it's already seen. That inflates the reported accuracy.

**What happened:**
```
[FLAG] 1 cross-split near-duplicate pairs found.
  -> 1 involve the test split
```

One pair was found: two `Tomato_Spider_mites` images with a hash distance of 2 (nearly identical). The training image was kept; the test image was dropped from the manifest. The manifest was rewritten.

**Verification:** Re-ran `pytest` — the leakage gate test now passed.

**Think of it like:** Before an exam, checking whether any practice questions accidentally ended up on the real test paper.

---

## Phase 4: Retrain with Early Stopping

**What we did:** Rebuilt the exact same model architecture (MobileNetV3Small backbone, 4-unit concept head, 15-unit disease head) but added two critical callbacks:
- `EarlyStopping(monitor='val_D_accuracy', mode='max', patience=3)` — stops training automatically when the validation accuracy hasn't improved for 3 epochs, and *restores the best weights*.
- `ModelCheckpoint('best_model.keras', save_best_only=True)` — saves the model at every epoch where validation accuracy improved.

**Why:** The original model trained for exactly 12 epochs. By epoch 12, the training accuracy was 88.8% but the validation accuracy had plateaued at 67% — a sign of overfitting. The best weights were somewhere earlier in training, but they were never saved. Now the `best_model.keras` file contains the weights from the epoch with the *highest validation accuracy*, not just the last epoch.

**Also fixed:** The INT8 calibration data. The original code passed augmented training images (with blur, dropout holes) to the quantization calibration step. That's wrong — the calibration images should look like what the model sees at inference time (clean, unaugmented). The new `quantize.py` script uses `augment_fn=None` for calibration.

**Training result:**
- Training ran for 30 epochs (with early stopping deciding when to quit)
- Best `val_D_accuracy`: **93.7%**
- Saved to `best_model.keras`

**Think of it like:** Instead of studying for exactly 12 hours and submitting whatever you know then, you study until you've peaked — and you bookmark the page where you felt sharpest.

---

## Phase 5: Evaluate (Clean Test Set)

**What we did:** Loaded `best_model.keras` and ran it on the 442 held-out test images it had *never seen during training*.

**Results:**
```
[Keras/FP32] accuracy on test (clean): 0.9276
```

**Per-class F1 scores (highlights):**

| Class | F1 | Note |
|---|---|---|
| Potato___Early_blight | 1.000 | Perfect |
| Potato___healthy | 1.000 | Perfect |
| Tomato_healthy | 1.000 | Perfect |
| Tomato__Tomato_mosaic_virus | 0.984 | Near-perfect |
| Tomato_Late_blight | 0.833 | Weakest — confused with Early blight |
| Tomato_Early_blight | 0.852 | Second weakest — recall only 76.7% |

**This is the number that belongs in the README** — 92.8% on unseen test data, not 88.8% on data the model trained on.

**Think of it like:** The final exam score, not the practice test score.

---

## Phase 6: Evaluate (Corrupted Test Set — The Robustness Proof)

**What we did:** Applied the same test images but corrupted them with MotionBlur (blur_limit=7), GaussNoise, and RandomBrightnessContrast before running them through the model. This simulates real-world field conditions: shaky cameras, poor lighting, dust on lenses.

**Result:**
```
[Keras/FP32] accuracy on test (corrupted): 0.0792
```

**What 7.9% means:** Random chance for 15 classes is 6.7% (1/15). The model scored barely above chance. Under corruption, it essentially fails completely.

**The honest finding:** The original project's Track A "robustness" claim was entirely unproven — the corruption pipeline was built but `model.evaluate()` was never called. This is the first time the number was ever computed. It shows the training augmentation (flip, rotate, dropout) does **not** make the model robust to MotionBlur/GaussNoise at this severity. To genuinely claim robustness, the training augmentation would need to include these corruptions too.

**This is exactly the kind of thing an interviewer would ask:** "How does your model perform under image degradation?" Now you have a real answer — and you're honest about what it means.

**Think of it like:** Training for a spelling bee using only easy words, then being surprised you struggle with hard words in the actual competition.

---

## Phase 7: INT8 Quantization + Accuracy Parity Check

**What we did:** Converted the 7.34 MB Keras FP32 model to a 0.79 MB INT8 TFLite model for Android edge deployment, then immediately evaluated the quantized model on the same test set to measure any accuracy loss.

**Results:**
```
FP32 accuracy:  92.8%
INT8 accuracy:  60.9%
Delta:         -31.9 percentage points
Compression:    9.3× smaller
```

**The honest finding:** The original project proudly advertised "9.4× size reduction" with no mention of the accuracy cost. A 31.9 pp accuracy drop is enormous — going from "9 out of 10 correct" to "6 out of 10 correct." That must be disclosed next to the compression ratio, not hidden. If the Android app needs reliable disease identification, quantization-aware training (QAT) should be explored.

**Think of it like:** Advertising that you shrunk a painting to 1/9th its size, without mentioning that most of the colors washed out.

---

## Phase 8: Hardcode the Grad-CAM Layer

**What we did:** Ran the model inspection to identify exactly which layer is the "last semantically meaningful convolutional feature map" — the one Grad-CAM should read gradients from. Then hardcoded that layer name in `scripts/gradcam.py`.

**What was found:** The last convolutional activation before the GlobalAveragePooling layer is `activation_17` (shape: 7×7×432). This is now hardcoded in `gradcam.py`:
```python
LAST_CONV_LAYER_NAME = "activation_17"
```

**Why this matters:** The original project used a fragile "scan backwards and find the first 4D-output layer" trick. MobileNetV3 has *many* 4D-shaped layers (depthwise convolutions, expansion blocks). The trick could silently pick the wrong one, especially after any model change. Hardcoding means:
1. You always get the right layer.
2. If the backbone changes, the assertion fails loudly instead of silently returning garbage.

**Why real Grad-CAM matters vs. activation average:** The original "Grad-CAM" just averaged all feature channels equally — the heatmap was identical regardless of which disease was predicted. Real Grad-CAM weights each channel by how much the *target class probability* changes when that channel activates. Different diseases → different heatmaps → actual explanations.

**Think of it like:** The difference between "this region of the leaf is always bright" (activation average) vs. "this region of the leaf specifically caused the model to say 'blight'" (Grad-CAM).

---

## Phase 9: Build 4 Production Notebooks

**What we did:** Created four Jupyter notebooks in `notebooks/`, each handling one stage of the pipeline, each importing from the tested `scripts/` modules instead of copying code inline.

| Notebook | Purpose |
|---|---|
| `01_EDA_and_Data_Pipeline.ipynb` | Download data → build manifest → visualize class balance → run leakage check |
| `02_Model_Training.ipynb` | Load manifest → build tf.data pipelines → define architecture → train with early stopping → plot curves |
| `03_Evaluation_Robustness_and_Quantization.ipynb` | Load eval JSONs or recompute → show F1 chart → corrupted-test bar chart → INT8 parity check |
| `04_GradCAM_Explainability.ipynb` | Load model → assert layer name is set → generate 3-panel dashboards → verify class-discriminativeness |

**Why import from scripts instead of copying code:** If you copy the same function into 4 notebooks and then fix a bug in one, the other 3 still have the bug. By importing from `scripts/`, there is one version of every function. The notebooks and the pytest suite both test *the same code*.

**Key design decisions:**
- NB01 uses `dp.build_dataframe()`, `dp.undersample()`, `dp.stratified_split()` from `data_pipeline.py` — no inline reimplementation.
- NB03 loads `eval_report/*.json` if the files exist (so you can re-run the notebook for documentation without needing a GPU), and recomputes from scratch only if they don't.
- NB04 asserts `LAST_CONV_LAYER_NAME` is not `None` before running — fails loudly if someone forgot to hardcode the layer.

**Think of it like:** Having one recipe in a cookbook and every chef in the kitchen following that same recipe, instead of each chef writing their own version from memory.

---

## Phase 10: Write the Final README with Real Numbers

**What we did:** Replaced the old README (which advertised 88.8% training accuracy and had no test numbers) with a new one that contains only real, verified numbers.

**Key changes:**
- **Headline metric changed** from 88.8% (train) to **92.8% (clean test, n=442)**.
- **Corrupted-test result added**: 7.9%, with an honest explanation of what it means.
- **INT8 delta added**: −31.9 pp, disclosed alongside the compression ratio.
- **Track A claim rewritten**: No longer says "robust" — says "near-random-chance under corruption, training augmentation must be extended to include this corruption family before a robustness claim can be made."
- **Concept bottleneck honesty added**: Explicitly says the concepts are class-derived labels, not independently annotated ground truth.
- **Known limitations section added**: Four paragraphs covering corrupted-test collapse, INT8 gap, dataset scale, and near-duplicate leakage.
- **Full per-class F1 table added**: Including the two weakest classes (Tomato_Late_blight F1=0.833, Tomato_Early_blight F1=0.852) with a note on the confusion pattern.

**Think of it like:** Replacing a marketing brochure with an engineer's honest technical report.

---

## Phase 11: Final Verification — All 31 Tests Pass

**Result:**
```
31 passed, 0 skipped, 0 failed
```

Previously there were 7 skipped tests (the real-artifact gates). Now all 7 run against the actual model and manifest:

| Real-artifact test | What it checks |
|---|---|
| `test_real_manifest_has_every_image_readable` | Every path in manifest.csv points to a file that actually exists and can be opened |
| `test_real_manifest_size_floor` | manifest.csv has at least 1,000 rows (catches an accidental empty manifest) |
| `test_real_manifest_no_cross_split_path_duplicates` | No image path appears in more than one split |
| `test_real_manifest_concept_vectors_parse` | Every concept_vec column value is a valid Python list of 4 floats |
| `test_real_no_unresolved_test_split_leakage` | duplicate_report.csv has zero rows involving the test split |
| `test_real_model_loads_and_has_two_outputs` | best_model.keras loads and has exactly 2 outputs (concept head + disease head) |
| `test_real_model_output_widths_match_expected_classes` | Disease head has 15 outputs, concept head has 4 outputs |

**Think of it like:** The project passed a full audit — not just unit tests on dummy data, but integration tests on the actual artifacts.

---

## Final State: What's in the Project Now

```
plant_disease_detection/
├── notebooks/
│   ├── 01_EDA_and_Data_Pipeline.ipynb       ← NEW
│   ├── 02_Model_Training.ipynb              ← NEW
│   ├── 03_Evaluation_Robustness_and_Quantization.ipynb  ← NEW
│   └── 04_GradCAM_Explainability.ipynb      ← NEW
├── scripts/
│   ├── data_pipeline.py     ← single source of truth for splits
│   ├── gradcam.py           ← real gradient-weighted Grad-CAM, layer=activation_17
│   ├── evaluate.py          ← clean/corrupted/INT8 eval, JSON output
│   ├── check_duplicates.py  ← perceptual-hash leakage detector
│   ├── kaggle_download.py   ← headless download (no Colab dialog)
│   ├── train.py             ← standalone training script with early stopping
│   ├── quantize.py          ← INT8 quantization (clean calibration data)
│   └── requirements.txt
├── tests/
│   ├── conftest.py          ← shared fixtures (synthetic dataset + dummy model)
│   ├── test_data_pipeline.py
│   ├── test_duplicates.py
│   ├── test_evaluate.py
│   ├── test_gradcam.py
│   ├── test_kaggle_download.py
│   ├── test_model_sanity.py
│   └── test_real_artifacts.py
├── eval_report/
│   ├── eval_test.json         ← FP32 clean test results (92.8%)
│   └── eval_test_corrupt.json ← FP32 corrupted test results (7.9%)
├── best_model.keras           ← trained model (best val epoch, 7.34 MB)
├── edge_model.tflite          ← INT8 quantized model (0.79 MB)
├── manifest.csv               ← reproducible train/val/test split
├── duplicate_report.csv       ← 0 unresolved leakage pairs (1 found + resolved)
├── README.md                  ← fully rewritten with real numbers
└── pytest.ini
```

---

## Numbers That Belong On Your Resume

| Metric | Value | Context |
|---|---|---|
| Clean test accuracy | **92.8%** | 442 held-out images, never seen during training |
| Best val accuracy | **93.7%** | Epoch with highest validation performance (EarlyStopping) |
| Macro F1 (clean test) | **0.932** | Across all 15 classes |
| Model size (FP32) | **7.34 MB** | Keras format |
| Model size (INT8) | **0.79 MB** | TFLite, 9.3× compression |
| INT8 accuracy | **60.9%** | Report alongside compression ratio |
| Corrupted test accuracy | **7.9%** | Be honest — training augmentation does not cover this |
| Test suite | **31/31 passing** | Including real-artifact integration tests |
| Leakage pairs found | **1** | Detected and resolved before reporting any test accuracy |

---

## What You Can Honestly Say In An Interview

**"What accuracy does your model achieve?"**
> "92.8% on a held-out test set of 442 images the model never saw during training. The best validation accuracy during training was 93.7%. I want to be clear that train accuracy was 88.8% — that's a different number that the original project mistakenly reported as the headline."

**"Is it robust to real-world image degradation?"**
> "Under MotionBlur, Gaussian noise, and brightness/contrast corruption, accuracy drops to 7.9% — near random chance. That was the honest finding when I actually ran the evaluation. The training augmentation I used (flips, rotations, dropout) didn't cover that corruption family. To genuinely claim robustness, I'd need to include MotionBlur and GaussNoise in training augmentation and retrain."

**"What's Grad-CAM showing?"**
> "Real gradient-weighted Grad-CAM — the gradient of the predicted disease class with respect to the final convolutional feature map (layer `activation_17`), globally average-pooled into per-channel importance weights. I verified it's class-discriminative: the heatmap changes when the target class changes. The original code averaged activations without any gradient, which is not Grad-CAM."

**"How does the edge model perform?"**
> "INT8 quantization gives 9.3× size reduction (7.34 MB → 0.79 MB), but accuracy drops from 92.8% to 60.9% — a 31.9 point gap. I report both numbers honestly. For a real deployment requiring reliable diagnosis, quantization-aware training would be the next step."

**"Are your concepts real CBM concepts?"**
> "No — they're a hierarchical re-encoding of the 15 class labels. Every image of a given disease gets the same concept vector. That's an auxiliary supervision signal that helps the disease head learn, not independently annotated ground truth in the Koh et al. sense. I document this explicitly rather than overclaiming."
