# ANTIGRAVITY PLAN: Close the Field Gap, Keep the Product Honest

Audience: Antigravity (autonomous executor). Owner: Amit. Style: imperative, measurable, no code here.
Written 2026-11-05 from sandbox measurements. Every number below must be REPRODUCED on the owner's
machine before it is quoted anywhere outside this file.

## 0. Why this plan changed
A sandbox run of the frozen FP32 model on PlantDoc (in-the-wild web photos, CC BY 4.0,
`github.com/pratikkayal/PlantDoc-Dataset`) showed the real-world problem is domain shift, not photo quality.

| Measurement (1,098 mapped target images, 12 classes) | Value |
|---|---|
| Lab test accuracy (committed report) | 92.8% |
| PlantDoc accuracy, frozen model | 18.9%, 95% CI 16.5-21.1% |
| Exact-mapping classes vs approximate-mapping classes | 21.8% vs 10.0% (conclusion survives mapping doubt) |
| ECE on PlantDoc | 0.455 (mean confidence 0.64 vs accuracy 0.19) |
| Wrong answers with confidence >= 0.9 | 168 of 891 |
| Selective accuracy at confidence >= 0.95 | 28.1% at 13.9% coverage (confidence cannot fix this) |
| Quality gate (lab-style defaults) rejection on field images | 4.7%; accuracy on accepted 19.1% (gate does not catch domain shift) |
| Other-crop leaves (314 images) answered at confidence >= 0.8 | 27.1% |
| AUROC target-vs-other-crop from confidence / margin / entropy | 0.54 / 0.52 / 0.56 (useless) |
| Linear probe on frozen 436-d features, grouped 5-fold CV | 37.7% vs 18.9% frozen (features carry field signal) |
| De-duplicated PlantDoc test (92 images) | frozen 19.6%, linear probe 41.3% |
| PlantDoc test images with a near-duplicate in PlantDoc train | 10 of 102 |

Consequence: temperature scaling, threshold tuning and a confidence-based non-leaf guard cannot fix this.
Only adapting the model to field images can move the accuracy number. The original model stays untouched.

## 1. Hard rules
* Additive only. Protected: existing `scripts/*.py`, notebooks 01-04, `README_DRAFT.md`, `PLAN.md`, the 7 original
  test files, `best_model.keras`, `edge_model.tflite`, `manifest.csv`. The only allowed edit to an existing file is
  APPENDING lines to `.gitignore`.
* `best_model.keras` remains the default model and the resume-safe artifact. Any adapted model is a NEW file in
  `models_field/` and is never saved over the original.
* Never tune anything on a split that is later reported. Group near-duplicate images (perceptual hash, Hamming
  distance <= 6) so no cluster spans two splits.
* No fabricated numbers. Every figure in docs comes from a saved JSON or notebook output.
* Preprocessing contract: TensorFlow decode, 224x224, raw 0-255 floats, no division by 255.
* Every new function gets a unit test. Never weaken an existing test. Commit per phase on branch `course-extension`.
* If a phase gate fails, stop and report. Do not improvise around it.

## 2. Phase 0 - Baseline and execution (must-do, about 1-2 h)
* Pause OneDrive sync. Run `git status`. The repo has no commits: make the first commit and tag `v1-baseline`.
* Install `app/requirements-app.txt`. Run the full `pytest`; record pass/skip/fail counts.
* Run `tests/test_notebooks_valid.py`; if it fails, fix only notebooks 05 and 06.
* Execute notebooks 05 and 06 on full data. Report `advisory/safeguards_report.json` and, from notebook 05 section 4,
  the installed albumentations version plus the estimated noise level of the `evaluate.py`-style corruption.
* Launch the app and test: clean leaf, blurry photo, screenshot, non-leaf photo. Confirm the scope warning is visible.
* Gate: suite green, both notebooks run, app launches.

## 3. Phase 1 - Reproduce the field gap (must-do, about 1-2 h)
* Create notebook `07_Field_Reality_Check` (stepwise cells, comments, print per cell, heavy pandas).
* Clone Cropped-PlantDoc outside the repo (about 1.9 GB; never commit it). Use `advisory/plantdoc_class_map.csv`:
  `target` rows are evaluated, `negative_other_crop` rows are used for the non-target test, `excluded_*` rows are skipped.
* Compute perceptual hashes, cluster near-duplicates, and report train/test leakage.
* Report: overall and per-class accuracy with bootstrap 95% CI, exact vs approximate mapping, top confusions,
  ECE, coverage-vs-accuracy, gate rejection, other-crop confident-answer rate, AUROC of confidence signals.
* Gate: reproduces roughly 19% (within +/-4 points). If not, find out why before continuing and report it.

## 4. Phase 2 - Field adaptation as a separate model (the only step that can move the number)
* Ladder, stop at the first rung that meets the target, and record every rung's result:
  1. Dense head trained on frozen features (sandbox baseline: about 38%).
  2. Fine-tune the last backbone block plus both heads on a MIX of lab train (replay, to limit forgetting) and
     PlantDoc train, with class-balanced sampling, strong augmentation (random resized crop, colour jitter, blur,
     noise, rotation, random backgrounds if cheap), low learning rate, early stopping on grouped PlantDoc validation.
* Split discipline: PlantDoc train -> grouped train/val; PlantDoc test (de-duplicated) is touched once at the end;
  lab test is touched once to check regression.
* Classes: PlantDoc covers 12 of 15. Potato healthy, Tomato Target Spot and Tomato Spider mites stay lab-only; report
  them separately and say so in the UI.
* Target (a goal, not a promise): field accuracy at least 55% on the de-duplicated PlantDoc test with lab accuracy
  dropping no more than 3 points. If missed, publish the honest negative result and keep the app scope-limited.
* Save the model to `models_field/best_model_field.keras` and its metrics to `models_field/field_report.json`.
  Append `models_field/*.keras` to `.gitignore`.
* Gate: new unit tests pass; lab and field numbers are both in the report.

## 5. Phase 3 - Make the app safe for what it can and cannot do
* The app defaults to the original model. It offers the field model only if the file exists AND `field_report.json`
  shows it met the Phase 2 target. State in the UI which model is active and what it was validated on.
* Re-fit a separate FIELD operating point (threshold, gate) on PlantDoc validation only; keep the lab one unchanged.
  Evaluate temperature scaling and adopt it only if ECE improves on validation, lab test and PlantDoc test.
* Other-crop guard: confidence signals fail (AUROC about 0.54). Test feature-space distance (Mahalanobis or kNN on the
  436-d features) with lab validation + PlantDoc target as in-distribution and PlantDoc other-crop leaves as
  out-of-distribution. Adopt only at AUROC >= 0.90; otherwise keep it as a documented limitation.
* Gate: each adopted piece has a test; the app falls back safely if any config file is missing.

## 6. Phase 4 - Usability extras (cut first if exams press)
* Multi-photo consensus (up to 5 leaves of one plant, average probabilities, show agreement, say "uncertain" on
  disagreement). This directly compensates for unreliable single-photo answers.
* Hindi toggle via a separate `advisory/advisory_table_hi.csv`, every row `translation_unreviewed` until Amit reviews it.
* Opt-in feedback log (local CSV: prediction, confidence, status, timestamp; no images stored).
* Cut order: feedback log, Hindi, multi-photo.

## 7. Phase 5 - Package and document
* Update `COURSE_EXTENSION.md` with real numbers only. Add 3-5 app screenshots (healthy, diseased, rejected, uncertain).
* One-page viva sheet: why FP32 not INT8; how the threshold was chosen; what the gate does and does not catch; the
  measured field gap; what adaptation achieved or failed to achieve.
* Draft resume bullets using only numbers from saved reports. Keep ONE project entry plus one extension line.

## 8. Edge cases Antigravity must handle
* PlantDoc is web-scraped and label-noisy (several leaves, hands, watermarks). Treat it as a stress test, never as ground truth.
* PlantDoc "Tomato leaf" and "Bell_pepper leaf" are mapped to the healthy classes only approximately. Report exact and approximate separately.
* Fine-tuning can erase lab performance. Always keep lab replay and re-measure lab test.
* The advisory table is still unreviewed. Do not mark any row reviewed; add no dosages or product names.
