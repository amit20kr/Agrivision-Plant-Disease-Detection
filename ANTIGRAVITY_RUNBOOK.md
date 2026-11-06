# ANTIGRAVITY RUNBOOK: Execute `ANTIGRAVITY_PLAN.md` With Minimal Tokens

Read this file and `ANTIGRAVITY_PLAN.md` exactly once, at the start. Do not re-read them; record state in `PROGRESS.md`.
The plan says WHAT to do. This runbook says HOW to do it cheaply and reliably.

## 1. Session start (budget: under 10 tool calls)
* Create `PROGRESS.md` (max 40 lines): current phase, last gate result, files created, next action, open questions.
  Update it at the end of every phase. After any context reset, read only this file and continue.
* Verify the environment once, in a single command: Python version, TensorFlow version, albumentations version,
  GPU availability, free disk space, current git branch. Record the result in `PROGRESS.md`. Never re-check.
* Run everything from the repo root. Use `pathlib`. Manifest paths are repo-relative with mixed slashes.
* Create a gitignored `cache/` folder for reusable intermediates (see section 4).

## 2. Do not read what is already summarised here
Never open the files below to learn their API. Use this index; open one only if a test fails because of it.

* `scripts/inference_utils.py` (new, tested). Public functions:
  * `to_uint8_rgb(path | bytes | upload | PIL | ndarray)`: decodes with TensorFlow (matches training).
  * `resize_for_model(rgb_uint8)`: 224x224 float32, raw 0-255.
  * `GateConfig` with `.load()` and `.save()`; `image_quality_metrics`; `evaluate_gate(metrics, min_side, cfg)`; `quality_gate`.
  * `corrupt_image(rgb224, noise_sigma, blur_len, brightness_delta, seed)`: explicit severities.
  * `decide(probs, threshold)`; `coverage_accuracy_table(frame)`; `pick_threshold(table, target, min_coverage)`;
    `expected_calibration_error(conf, correct)`.
  * `predict_frame(model, df, cfg, corrupt, batch_size)`: df needs `File_Path`, `Class_Name`, `label_idx`; `corrupt` is a
    dict or a callable on a uint8 224x224 image. Returns pred, confidence, margin, entropy, correct, quality metrics, gate flag.
  * `load_advisory`, `validate_advisory`, `lookup_advisory`, `gradcam_overlay`, `analyze_image(model, image, advisory, cfg, threshold)`.
* Existing modules: `data_pipeline` has `SELECTED_CLASSES` (15), `LABEL_MAP`, `CONCEPT_MAP`, `IMG_SIZE`, `_decode_resize(path)`.
  `gradcam.make_gradcam_heatmap(model, x, layer, pred_index)` returns `(heatmap, pred_index)`; layer name `activation_17`.
* Model: input 224x224x3 on a raw 0-255 scale; outputs `[concepts(4), diseases(15)]`. The 436-d vector feeding the
  disease Dense layer `D` is the 432-d pooled feature concatenated with the 4 concept outputs. Layer `C` makes the concepts.
* Manifest columns: `Class_Name`, `File_Name`, `File_Path`, `split` (train/val/test), `label_idx`, `concept_vec`.
  Splits are 2066 / 443 / 442.
* Known traps: PIL and TensorFlow decode JPEGs differently (always use TensorFlow); the README spells a class
  `Tomato_Spider_mites` but the real class is `Tomato_Spider_mites_Two_spotted_spider_mite`; albumentations
  `GaussNoise` defaults changed between 1.x and 2.x.

## 3. Output discipline (the biggest token saver)
* Print at most 20 rows of any table; use `head`, `describe`, or aggregates. Never print a full DataFrame.
* Run long jobs headless, send output to a log file, and read only the last 15 lines. Never stream training or
  inference progress into the conversation.
* Silence noise: TensorFlow log level 3, `verbose=0` or `verbose=2` for training, warnings filtered.
* Use `pytest -q --tb=short`. While debugging, run only the failing test file with `-x`. Run the full suite only at phase gates.
* Use `git diff --stat` and `git status --short`. Never diff notebook JSON.
* Do not open images, except one Grad-CAM grid per phase when the plan requires visual checking.
* Do not print or paste code back to the user. State what changed in one line per file.
* No web searches except a one-time license check of a dataset.

## 4. Compute once, reuse everywhere
* Cache these under `cache/` and reuse them: PlantDoc image index, perceptual-hash clusters, frozen-model predictions
  and 436-d features for PlantDoc and for the lab splits (`.npy` or `.csv`), and the fine-tuning split assignments.
* Key each cache file by model file name and modification time, so a changed model invalidates it automatically.
* Clone PlantDoc once, shallow, outside the repo. Never copy it into the repo and never commit it.
* If only a head is trained, train it on cached features and skip image decoding entirely.

## 5. Build order that avoids rework
* Put logic in tested functions under `scripts/` first. Notebook cells only call those functions and print results.
* Validate each function on a tiny subset (about 5% of data, or 2 images per class), then run the full data once.
* Execute each notebook once, headless, after its logic is validated. Do not iterate by re-running whole notebooks.
* Notebooks keep text outputs only. Save figures to files; do not embed large images.
* Keep the notebook style the owner asked for: stepwise cells, comments, a print per cell, and wide pandas use.

## 6. Phase 2 fine-tuning discipline
* Check for a GPU first (from section 1). With no GPU, do rung 1 (cached-feature head) fully, then run rung 2 on the last
  backbone block only, with a fixed epoch cap, and report the measured seconds per epoch before committing to more.
* Run training as a background process with checkpoints. Log one line per epoch. Poll the log every few minutes; never
  watch it live.
* Pre-register in `PROGRESS.md`, before training: the success targets from the plan, the split names, the epoch cap,
  and the early-stopping rule. Do not change them after seeing results.
* Touch the PlantDoc test split once and the lab test split once. Everything else uses grouped validation.
* Stop after 2 failed attempts at the same step. Report the cause and the numbers; do not keep trying variations.
* A missed target is a valid result. Record it honestly and keep the original model as the default.

## 7. Git and file hygiene
* One commit per phase gate, on branch `course-extension`, with a one-line message.
* Never add: `data/`, `cache/`, PlantDoc, `*.keras`, `*.tflite`. Append the needed ignore lines to `.gitignore` only.
* Never touch the protected files listed in the plan. Check `git status --short` before every commit.
* Pause OneDrive sync during git operations.

## 8. Reporting format (after each phase, at most 15 lines)
* Phase and gate result (pass or fail).
* Numbers, copied from the saved JSON or notebook output, with the file they came from.
* Files created or changed (names only).
* Anything unexpected, and the single next action.
Do not restate the plan, and do not apologise or narrate.

## 9. Budget guide (share of total effort)
* Phase 0: 15%. Phase 1: 20%. Phase 2: 40%. Phase 3: 15%. Phases 4 and 5: 10%.
* If effort runs short, cut in the plan's order: feedback log, then Hindi, then multi-photo. Never cut Phase 0 or 1.
* The owner has exams. After Phase 1, stop and ask whether to continue into Phase 2.

## 10. Ask the owner instead of guessing
* An ambiguous class mapping, a license question, a disk or GPU limit, a missed target, any request to touch a protected file,
  or any plan to quote a number that is not in a saved report.
