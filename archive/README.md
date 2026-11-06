# Archive

Original Colab-era hackathon notebooks, kept for history only. Superseded entirely by `../notebooks/01`-`04`.

- `PlantDisease_Training_ORIGINAL_HACKATHON.ipynb` — used `google.colab.files.upload()` for auth (breaks outside Colab), computed train_test split as in-memory lists only (not persisted, not reproducible), never ran the corrupted/robustness eval, never quantized, "FOR JUDGING PANEL" framing.
- `PlantDisease_Inference_ORIGINAL_HACKATHON.ipynb` — original Grad-CAM implementation averaged raw activations with no gradient weighting (not class-discriminative); fixed in `../scripts/gradcam.py` and `../notebooks/04`.

Do not link these from the README or a resume. They exist so the "before" state of the remediation is auditable if asked in an interview.
