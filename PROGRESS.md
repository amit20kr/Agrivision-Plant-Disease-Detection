WORKING ROOT = C:\Users\amit0\OneDrive\Desktop\plant_disease_detection. No other copy is authoritative.

# ANTIGRAVITY PROGRESS

## Path Audit & Single Source of Truth
* Authoritative root: `C:\Users\amit0\OneDrive\Desktop\plant_disease_detection` (branch `course-extension`, HEAD `69aced20a903d8174da4f1f82be1fc09e9167393`, 40,878 files).
* Duplicate check: `C:\Users\amit0\Desktop\plant_disease_detection` was inspected (separate non-junction folder, branch `master` with no commits, 866 files, latest file from 25 Sep 2026, 0 files modified since 8 Nov).
* Retired other copy: Safely renamed to `C:\Users\amit0\Desktop\plant_disease_detection__OLD_DO_NOT_USE` as instructed.
* Bundle status: `nb07_dashboard_bundle.zip` generated and placed in `Downloads/` and repo root.

## Bundle & Dashboard Implementation (Complete)
* Files implemented and validated:
  - `ANTIGRAVITY_NB07_DASHBOARD.md`
  - `DEPLOY.md`
  - `.streamlit/config.toml`
  - `app/dashboard.py` (Multi-tab Streamlit dashboard: Classify & explain, Batch, Lab performance, Field reality)
  - `app/requirements.txt`
  - `notebooks/07_Field_Reality_Check.ipynb` (28 stepwise cells)
  - `scripts/field_analysis.py` (exact 436-d feature extractor, imagehash near-duplicate clustering, bootstrap CI, AUROC, linear probe)
  - `scripts/dashboard_utils.py` (model loader, tolerant report parser, upload validation, Grad-CAM blending, batch prediction)
  - `scripts/make_demo_samples.py` (extracts 6 sample demo images into `app/demo_samples/`)
  - 5 new test modules in `tests/`
* Test suite: **97/97 passed** (80 original + 17 new tests), 0 failures.

## Advisory Artifacts & Rebaseline Status
* **Status**: REVERTED to committed baseline via `git checkout -- advisory/`.
* Provisional runs in `cache/rebaseline_nb05/` and `advisory/*.provisional` were calibrated on partial data (396 val, 409 test) due to 277 offline OneDrive stubs.
* **MANDATE**: "calibrated on partial data; do not quote". All docs and dashboard quote only the committed **92.8%** from `eval_report/eval_test.json`.
* Final notebook 05 rebaseline runs only AFTER 100% hydration (2,951/2,951 files readable) and owner approval.

## Environment & Workarounds
* Python: 3.12.9 | TensorFlow: 2.21.0 | Albumentations: 2.0.8 | GPU: None (CPU only) | Free Disk: 116.05 GB
* KERAS_HOME workaround: `$env:KERAS_HOME = "$pwd\cache\keras"` before running `pytest` or Python commands.
* PlantDoc dataset: Full checkout at `C:\Users\amit0\PlantDoc-Dataset` (2,570 images, 0 failures, all class-map folders verified). Gate PASS.

## Next Action
* Run Notebook 07 (`notebooks/07_Field_Reality_Check.ipynb`) to evaluate against the real PlantDoc dataset and generate `advisory/field_gap_report.json` and figures.
