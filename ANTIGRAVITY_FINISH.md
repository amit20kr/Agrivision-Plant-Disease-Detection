# ANTIGRAVITY HANDOFF: final close-out (NO retraining)

Working root: C:\Users\amit0\OneDrive\Desktop\plant_disease_detection (the only authoritative copy).
Touch only: this repo, C:\Users\amit0\PlantDoc-Dataset, a temp folder for the fresh clone. No drive-wide searches, no clipboard, no recent-items.
Bound every command with a timeout. Do NOT use `attrib /S` on data/. Set KERAS_HOME as in PROGRESS.md before any TensorFlow run.

## Scope
IN:  verify data, execute notebooks 01/03/04, note on notebook 02, README rewrite, fresh-clone test, tags, optional fp16 check.
OUT: fine-tuning, temperature scaling, OOD guard, notebook 05 re-baseline, notebook 02 execution, any edit to
     best_model.keras, manifest.csv, duplicate_report.csv, eval_report/*, advisory/*. Reason: cost in time and tokens, and each one
     changes committed artifacts or the dashboard. List them as future work in the README, nothing more.

## Step 0: preconditions (owner has set "Always keep on this device" on the repo folder)
- Open AND decode every file in manifest.csv (batches, per-batch timeout). Gate: 2,951 of 2,951.
- If fewer: print counts by split and class, STOP steps 2 and 5-data parts, still do Steps 1, 3, 4 (README) and report.
- Record SHA256 of best_model.keras, manifest.csv, duplicate_report.csv and eval_report/*.json. Copy them to cache/backup_final/ (gitignored).

## Step 1: dry read of the notebooks before running
- grep notebooks 01, 03, 04 for writes (to_csv, json.dump, .save(, open(..., "w"), tflite). List every path they write.
- If one writes a committed artifact, run it anyway but expect to revert (Step 2 guard). Never let a run replace a committed file.

## Step 2: execute notebooks 01, 03, 04 headless, in place, cwd = notebooks/
- Timeout 3600 s each. Keep text outputs; if an executed notebook exceeds 3 MB, strip image outputs except the first two.
- Gates (all required, otherwise `git checkout -- <changed committed files>` and report, do not commit):
  a. Notebook 03 prints n = 442 and clean accuracy 0.9276 (410/442) and corrupted accuracy 0.0792.
  b. `git diff --stat -- manifest.csv duplicate_report.csv eval_report advisory best_model.keras` is EMPTY and the SHA256s from Step 0 match.
  c. Scan every cell's outputs for errors. The process exit code is NOT evidence (PowerShell reports stderr as exit 1).
  d. Notebook 01 manifest summary: 2,951 rows, same split counts as the committed manifest.
- Notebook 04: Grad-CAM layer must be activation_17; confirm heatmaps differ between two predicted classes (the notebook already tests this).

## Step 3: notebook 02 (not executed)
- Insert one markdown cell at the top: "Status: training recipe. The committed model was trained earlier on CPU with scripts/train.py;
  the epoch log was not preserved and this notebook was intentionally not re-executed so best_model.keras is never overwritten."
- Do not run it. Do not edit other cells.

## Step 4: README rewrite (single pass, keep every existing honest finding)
Required content, in this order:
1. One-paragraph summary + "What it is and is not": lab-trained classifier, NOT a field diagnostic tool.
2. Results snapshot table. Lab 92.8% (410/442), corrupted 7.9%, INT8 60.9%, field (PlantDoc) 18.9% [16.8, 21.2], exact vs approximate
   mapping 21.8% / 10.0%, field ECE 0.455, other-crop confident answers 24.4%, quality-gate rejection 6.8%, linear probe 35.0%.
   Every number must be read from eval_report/*.json, advisory/field_gap_report.json or an executed notebook output. Paste the notebook 07
   section 24 claims table verbatim.
3. Notebook index 01-07 (one line each) and run order. Say notebook 02 is a recipe, not re-run.
4. Safeguards and advisory (notebooks 05-06): describe the mechanism only. State: "Calibrated on a partial copy of the data
   (cloud-sync gap); re-calibration on the full data is pending." Do NOT quote threshold/ECE numbers from safeguards_report.json.
5. Dashboard: features, run command, live URL placeholder `<ADD AFTER DEPLOY>`, link to DEPLOY.md.
6. Project history: v1 = course baseline (model, split, evaluation, notebooks 01-04); v2 = same model, split and numbers, plus notebooks
   05-07 and the dashboard. v2 adds validation, not new claims. Link the earlier repository (owner will supply the URL: leave `<OLD REPO URL>`).
7. Known limitations and future work: field adaptation (fine-tune on PlantVillage + PlantDoc), calibration, other-crop guard,
   float16 or QAT for edge, re-calibration of notebook 05 on the full data, training log preserved next time.
8. Fix stale text: real test count from the final pytest run, repo tree, license line (leave "MIT, LICENSE file to be added" if no file exists).
Forbidden: any claim about WHY the field gap exists (backgrounds, lighting, classifier head), "learns lesion texture not background",
or any number not traceable to a file.

## Step 5: fresh-clone verification
- `git clone` the committed repo into a temp folder (no data/, no cache/). Run `pytest -q` there with KERAS_HOME set.
  Gate: all pass except tests that skip because data/ is absent. Any other failure: report, do not "fix" tests silently.
- From the clone: `streamlit run app/dashboard.py`, confirm HTTP 200 and that a sample photo yields a prediction.
- `git ls-files`: best_model.keras present; nothing under data/, cache/; no tracked file over 50 MB (list the 5 largest).
- Tags: `git tag v1-baseline <the initial baseline commit>` and `git tag v2 HEAD`.

## Step 6: OPTIONAL, one attempt, 20 minutes max, only if Steps 0-5 passed
float16 TFLite check for the README's edge row.
- First read scripts/evaluate.py. It wrote to eval_report/eval_test.json in the past. If it cannot write elsewhere, SKIP this step.
- Else: new file scripts/quantize_fp16.py (TFLiteConverter, Optimize.DEFAULT, supported_types float16) writing to cache/, evaluate to a path
  outside eval_report/. Report accuracy and size in the README as "single run, artifact not committed". Do not replace the INT8 row; add a row.

## Step 7: commits
Small commits on course-extension: executed notebooks; README; (optional) quantize_fp16.py. No push. The owner pushes and deploys per DEPLOY.md.

## Report (max 15 lines)
Gates 0, 2a-d, 5; table of notebooks 01/03/04 (cells run, cell errors, size); final pytest counts; README numbers cross-checked against files
(list any mismatch); Step 6 result or "skipped: reason"; anything unexpected.
