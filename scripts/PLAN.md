# Implementation Plan — Plant Disease CBM Resume Hardening

Status legend: [DONE] = built + functionally tested here on synthetic data.
[TODO-ANTIGRAVITY] = needs GPU, the real dataset, or your Kaggle credentials —
none of which this environment has network/hardware access to.

## Why these files exist
The original project's only reported metric was TRAINING accuracy (88.8%)
while validation accuracy plateaus at 67.0% — a large, undisclosed
generalization gap. The "robustness" test set (MotionBlur/GaussNoise) was
built but never evaluated. The "Grad-CAM" heatmap was actually an
unweighted activation average with no gradient term. None of this survives
a technical interview follow-up question. These scripts fix the mechanism;
Antigravity needs to run them against your real trained model to get real
numbers.

## Files delivered (in `scripts/`)
- `data_pipeline.py` [DONE] — single source of truth for the dataset build,
  200/class undersampling, and the 70/15/15 stratified split. Writes
  `manifest.csv` so training/eval/inference all read the exact same split
  instead of three independently-computed ones.
- `gradcam.py` [DONE] — real gradient-weighted Grad-CAM. Verified
  class-discriminative on a synthetic dummy model (different predicted
  class → measurably different heatmap; the original code could not do this).
- `evaluate.py` [DONE] — classification report, confusion matrix, clean-val
  accuracy, corrupted-test accuracy (the actual Track A evidence), and
  FP32-vs-INT8 accuracy delta after quantization. Verified end-to-end on a
  synthetic dummy CBM model + dummy `.tflite` export.
- `check_duplicates.py` [DONE] — perceptual-hash near-duplicate detector,
  flags cross-split leakage (PlantVillage is documented to contain
  near-duplicate images). Verified: correctly flags a planted near-duplicate
  pair, correctly ignores an unrelated image.
- `kaggle_download.py` [DONE] — headless dataset download, no
  `google.colab.files.upload()` prompt. Reads `~/.kaggle/kaggle.json` or
  `KAGGLE_USERNAME`/`KAGGLE_KEY` env vars. Credential-check logic tested;
  actual download untested here (no Kaggle network access in this sandbox —
  Antigravity's environment should have it).
- `requirements.txt`, `.gitignore` [DONE].

All five scripts import/run cleanly under TF 2.21 (tested in this sandbox).
None of them have touched your real PlantVillage data or your real trained
model — that requires GPU + the actual dataset, which only Antigravity's
environment can provide.

## Pretest suite (in `tests/`, run from repo root) [DONE]
A full pytest suite validating every script above against synthetic data and
a dummy model. 24 tests pass, 7 auto-skip until real artifacts exist. Run
from the repo root with `pytest` (needs `pytest.ini` at repo root, delivered).

- `tests/conftest.py` — shared fixtures: synthetic 3-class dataset, dummy
  two-headed CBM model (same topology as the real one, cheap to build/train).
- `tests/test_data_pipeline.py` — split ratios (~70/15/15), zero cross-split
  path duplicates, every class present in every split, label/concept map
  consistency, split reproducibility given the same seed, tf.data batch
  shapes/no-NaN check.
- `tests/test_model_sanity.py` — output shapes/ranges (sigmoid concepts in
  [0,1], softmax disease sums to 1), gradients actually reach both heads,
  disease head actually consumes the concept head's output (catches a
  dropped concat in a future refactor).
- `tests/test_gradcam.py` — regression test for the original bug: asserts
  the heatmap is class-discriminative (different predicted class -> visibly
  different heatmap) and that calling without a verified layer name fails
  loudly instead of silently guessing.
- `tests/test_evaluate.py` — evaluate_keras report structure, TFLite
  conversion + inference mechanics, and specifically that the disease-head
  output tensor is correctly identified by shape (not accidentally the
  concept head).
- `tests/test_duplicates.py` — verified against a planted near-duplicate
  pair (must flag) and a same-split duplicate (must NOT flag).
- `tests/test_kaggle_download.py` — credential-check logic: raises clearly
  with no credentials, passes with env vars or kaggle.json present.
- `tests/test_real_artifacts.py` — the same invariants re-run against your
  REAL `manifest.csv` / `duplicate_report.csv` / `best_model.keras` once
  they exist at the repo root. Auto-skips until each file appears, so this
  file is the actual GATE between pipeline stages (see step numbers below).
  Verified with both a positive case (valid fake-real manifest -> passes)
  and a negative control (manifest pointing at a missing file -> fails
  correctly, not silently green).

## README_DRAFT.md [DONE]
A full rewrite of the README with every flagged claim fixed: real
val/test/corrupted-test accuracy in place of train-only accuracy, correct
Grad-CAM language, the concept-label honesty note, INT8 accuracy-parity
row, a Known Limitations section, and updated repo structure/run
instructions covering `scripts/` and `tests/`. Every number that can't be
known without a real run is marked `[[TODO: ...]]` with the exact command
that produces it — do not fill these in by hand. Once Antigravity has real
numbers, replace `README.md` with this file's content and delete the
reviewer note at the top.

## Execution order for Antigravity

0. **Pretest gate**: `pip install -r scripts/requirements.txt && pytest` from
   the repo root. All 24 mechanism tests must pass before touching the real
   dataset — if any fail, the pipeline itself is broken and no amount of
   GPU time will fix that.
1. **Environment**: `pip install -r requirements.txt`
2. **Credentials**: set `KAGGLE_USERNAME`/`KAGGLE_KEY` (or drop
   `kaggle.json` in `~/.kaggle/`) — do NOT commit either to the repo.
3. **Data**: `python scripts/kaggle_download.py --dest data/raw`
4. **Manifest**: `python scripts/data_pipeline.py --base_path data/raw/plantvillage_dataset/color --out manifest.csv`
   (adjust `--base_path` to whatever the actual extracted folder name is —
   the original notebook already had to handle this being inconsistent).
   Then re-run `pytest` — `tests/test_real_artifacts.py`'s manifest tests
   will now execute instead of skipping. Fix any failure before proceeding.
5. **Leakage check**: `python scripts/check_duplicates.py --manifest manifest.csv --out duplicate_report.csv`
   — inspect the output before trusting any test-set accuracy number. Re-run
   `pytest` again; `test_real_no_unresolved_test_split_leakage` is the hard
   gate here.
6. **Retrain**: port `PlantDisease_Training.ipynb`'s architecture cell
   (MobileNetV3Small backbone, concept+disease heads — that part is fine,
   keep it) onto `data_pipeline.get_tf_dataset()` instead of the inline
   lists, and add:
   - `tf.keras.callbacks.EarlyStopping(monitor='val_D_accuracy', patience=3, restore_best_weights=True)`
   - `tf.keras.callbacks.ModelCheckpoint('best_model.keras', monitor='val_D_accuracy', save_best_only=True)`
   Train for more epochs than 12 with these callbacks active rather than a
   fixed epoch count — let early stopping decide. Re-run `pytest` once
   `best_model.keras` exists — `test_real_model_*` gates now execute.
7. **Evaluate — clean**: `python scripts/evaluate.py --model best_model.keras --manifest manifest.csv --split test`
8. **Evaluate — corrupted (Track A proof)**: `python scripts/evaluate.py --model best_model.keras --manifest manifest.csv --split test --corrupt`
9. **Quantize** (keep the existing notebook cell's INT8 conversion logic),
   then **evaluate — parity check**:
   `python scripts/evaluate.py --model best_model.keras --manifest manifest.csv --split test --tflite edge_model.tflite`
10. **Grad-CAM layer ID**: `python scripts/gradcam.py --inspect best_model.keras`,
    pick the correct final feature layer from the printed candidates, hardcode
    it as `LAST_CONV_LAYER_NAME` in `gradcam.py`. Regenerate the inference
    notebook's 3-panel dashboard using `gradcam.make_gradcam_heatmap()`
    instead of the raw activation average.
11. **README** — `README_DRAFT.md` (delivered, repo root) already has the
    corrected structure and every fixed claim; it just needs the
    `[[TODO: ...]]` markers replaced with the real numbers from steps 7–9
    and the classification report. Fill those in, delete the reviewer note
    at the top, then replace `README.md`'s content with it. Do not add new
    unverified claims while doing this.
12. **Artifact hosting**: commit `best_model.keras` and `edge_model.tflite`
    via Git LFS or attach them to a GitHub Release, and link them from the
    README — right now a cloned repo has no way to run inference without a
    full retrain.
13. **Duplicate remediation**: if step 5 flags meaningful cross-split
    leakage, either drop one image from each flagged pair before re-running
    the split, or re-split at the near-duplicate-cluster level. Re-run steps
    7-9 after.

## What I did NOT do and why (Phase 1)
- Did not download the real PlantVillage dataset or retrain — no network
  access to Kaggle from this sandbox, and no GPU here regardless.
- Did not edit your two `.ipynb` files directly — modifying notebook JSON
  blind (without being able to execute and inspect real cell outputs) risks
  producing a notebook that looks right but silently breaks a variable
  reference. Safer for Antigravity to integrate these tested `.py` modules
  into the notebooks where it can execute and see the output at each step.
- Did not touch your existing `README.md` — the current draft's numbers need
  to be replaced with real post-retrain numbers (step 11), not guessed at.

---

## Phase 2 — Notebook execution & the two accuracy-collapse findings (current)

Steps 0-13 above are all [DONE]. Real training, evaluation, quantization and
leakage-checking have happened — `manifest.csv` (2,951 rows), `best_model.keras`,
`edge_model.tflite`, `duplicate_report.csv`, and `eval_report/*.json` all exist
and are internally consistent with each other and with `README.md`. Two real,
non-cosmetic findings came out of that work and are already documented
honestly in `README.md`'s "Known Limitations" — they are not bugs in the
write-up, they are findings that need a decision:

1. **Corrupted-test collapse**: 92.8% (clean) -> 7.9% (MotionBlur+GaussNoise+
   brightness shift), i.e. barely above the 6.7% random-chance floor.
   `eval_report/eval_test_corrupt.json`'s confusion matrix shows predictions
   collapsing onto just 2 of 15 classes (`Tomato_Leaf_Mold`: 257/442,
   `Tomato_Septoria_leaf_spot`: 183/442) — the model isn't "somewhat
   fragile", it has learned to map noisy/blurry texture onto whichever
   classes are themselves texturally spotty, regardless of true label.
2. **INT8 quantization collapse**: 92.8% -> 60.9% (-31.9pp) from
   `scripts/quantize.py`'s post-training INT8 conversion. `evaluate_tflite()`
   in `evaluate.py` has been patched (this session) to also return a
   confusion matrix / per-class report / `predicted_class_counts`, matching
   `evaluate_keras()` — it previously only returned a bare accuracy number,
   which made this collapse impossible to root-cause without re-instrumenting
   the eval code first.

### Ordered next steps for Antigravity

14. **Re-run the patched TFLite eval** to get the class-collapse breakdown for
    finding #2, same shape as finding #1's:
    `python scripts/evaluate.py --model best_model.keras --manifest manifest.csv --split test --tflite edge_model.tflite`
    Inspect `eval_report/eval_test.json`'s new `tflite.predicted_class_counts`
    field. If it also collapses onto 1-2 classes -> likely a
    representative-dataset/calibration bug in `quantize.py` (try increasing
    the 100-sample calibration set, or calibrating with `shuffle=True` across
    more of `train`, since a skewed 100-sample draw could mis-calibrate
    per-channel activation ranges for a dual-head model). If it's spread
    across more classes but still much worse than FP32 -> more consistent
    with genuine INT8 precision loss on this architecture; consider
    quantization-aware training (QAT) instead of post-training quantization.
15. **Execute all 4 notebooks in place** so the repo shows real, reproducible
    cell outputs instead of empty output arrays (currently `01`-`04` in
    `notebooks/` have zero saved outputs on any code cell — this is what a
    recruiter or interviewer opening them on GitHub will see first):
    ```
    jupyter nbconvert --to notebook --execute --inplace notebooks/01_EDA_and_Data_Pipeline.ipynb
    jupyter nbconvert --to notebook --execute --inplace notebooks/02_Model_Training.ipynb
    jupyter nbconvert --to notebook --execute --inplace notebooks/03_Evaluation_Robustness_and_Quantization.ipynb
    jupyter nbconvert --to notebook --execute --inplace notebooks/04_GradCAM_Explainability.ipynb
    ```
    Run strictly in that order (02 reads what 01 wrote, 03 reads what 02
    wrote, etc.). If 02 attempts a real retrain rather than loading the
    existing `best_model.keras`, gate/short-circuit that cell first (it
    should reuse the already-trained, already-evaluated model — retraining
    inside the notebook risks producing a *different* model than the one
    `README.md`'s numbers describe). Confirm notebook 03's printed
    `[FINDING]` lines match the numbers already in `README.md` before
    treating the notebooks as final; if they diverge, the manifest/model on
    disk and the notebook's own re-derivation have drifted and that must be
    resolved before shipping.
16. **Full pytest run** against real artifacts: `pytest` from repo root,
    expect **31 passed** (24 always-run + 7 `test_real_artifacts.py` gates,
    all of which should now execute instead of skip since `manifest.csv`,
    `duplicate_report.csv`, and `best_model.keras` all exist). Add one more
    gate test here: an assertion that `abs(tflite_accuracy - keras_accuracy)
    < 0.05` (or whatever threshold is decided after step 14) in
    `test_real_artifacts.py`, so a future re-quantization silently regressing
    to another class-collapse is a test failure, not a manual eval-report read.
17. **Two legacy root-level notebooks archived** (this session): moved
    `PlantDisease_Training.ipynb` and `PlantDisease_Inference.ipynb` into
    `archive/` (Colab-only auth, un-persisted split, unweighted Grad-CAM —
    see `archive/README.md`). Do not restore them to the repo root or link
    them from the README/resume; they represent the pre-remediation state.

