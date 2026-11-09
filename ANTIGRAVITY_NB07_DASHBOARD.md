# Antigravity Runbook: Notebook 07 & Streamlit Dashboard

## Overview
This runbook guides executing Notebook 07 (`07_Field_Reality_Check.ipynb`) and verifying the Streamlit dashboard (`app/dashboard.py`).

## 1. Environment & Setup
* Working Root: `C:\Users\amit0\OneDrive\Desktop\plant_disease_detection`
* Model: `best_model.keras`
* PlantDoc Directory: `C:\Users\amit0\PlantDoc-Dataset`
* Environment variable: `$env:KERAS_HOME = "$pwd\cache\keras"`

## 2. Stop Conditions & Gates
1. **Pipeline Control Gate (Lab Photos)**:
   - Evaluates $\ge 90$ lab test images across 15 classes.
   - Accuracy must be $\ge 85\%$ (expected ~92.8%). If below 85%, stop immediately.
2. **Reproduction Gate (Field Reality)**:
   - PlantDoc target accuracy must land within $18.9\% \pm 4.0\%$ ($14.9\% - 22.9\%$).
   - If outside this window, stop and diagnose mapping or directory path before proceeding.
3. **Artifact Generation Gate**:
   - `advisory/field_gap_report.json` must be written in strict JSON format without bare `NaN`s.
   - Six figures must be generated in `advisory/field_figures/`.

## 3. Known Traps
* **Keras config permission**: Always ensure `$env:KERAS_HOME` points to `cache/keras` so TensorFlow does not attempt to access `~/.keras/keras.json`.
* **PlantDoc location**: Use `C:\Users\amit0\PlantDoc-Dataset` directly; avoid re-cloning.
* **Feature dimensionality**: Layer `D` input is exactly 436-d (`[GAP(432), C(4)]`).
* **Strict JSON**: Python floats with `NaN` must be serialized as `null` or omitted so standard JSON parsers and Streamlit don't raise `ValueError` / `TypeError`.
* **Committed baseline accuracy**: Always quote **92.8%** from `eval_report/eval_test.json`, never 93.9%.

## 4. Running Tests
Run the test suite:
```powershell
$env:KERAS_HOME = "$pwd\cache\keras"; pytest tests/ -q
```

## 5. Launching Dashboard
```powershell
streamlit run app/dashboard.py
```
