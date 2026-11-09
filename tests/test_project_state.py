"""
State-pinning pretests for the REAL project artifacts.

Purpose: establish the actual, verified state of the repo (manifest, trained
model, Grad-CAM, committed eval reports) BEFORE any course-extension code is
added, so any later drift or breakage is detectable.

Unlike the synthetic-data tests, these run against the real manifest.csv,
best_model.keras, eval_report/*.json and (where present on disk) real test
images. Every test auto-skips if its prerequisite artifact is missing, so
`pytest` is safe to run on any machine.

Run from the repo root:  pytest tests/test_project_state.py
"""
import ast
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import data_pipeline as dp  # noqa: E402

MANIFEST = os.path.join(ROOT, "manifest.csv")
MODEL = os.path.join(ROOT, "best_model.keras")
TFLITE = os.path.join(ROOT, "edge_model.tflite")
EVAL_CLEAN = os.path.join(ROOT, "eval_report", "eval_test.json")
EVAL_CORRUPT = os.path.join(ROOT, "eval_report", "eval_test_corrupt.json")

# ---- values the README currently claims; update deliberately, never silently ----
EXPECTED_TOTAL = 2951
EXPECTED_SPLITS = {"train": 2066, "val": 443, "test": 442}
EXPECTED_CLEAN_ACC = 0.9276
EXPECTED_CORRUPT_ACC = 0.0792
EXPECTED_INT8_ACC = 0.6086
ACC_TOL = 0.002
MAX_PER_CLASS = dp.MAX_IMAGES_PER_CLASS

needs_manifest = pytest.mark.skipif(not os.path.exists(MANIFEST), reason="manifest.csv missing")
needs_model = pytest.mark.skipif(not os.path.exists(MODEL), reason="best_model.keras missing")
needs_clean_report = pytest.mark.skipif(not os.path.exists(EVAL_CLEAN), reason="eval_test.json missing")
needs_corrupt_report = pytest.mark.skipif(not os.path.exists(EVAL_CORRUPT), reason="eval_test_corrupt.json missing")


def _fix_path(p):
    """Manifest paths are repo-root-relative with mixed slashes (Windows-built).
    Normalise so the same manifest resolves on any OS and from any cwd."""
    return os.path.join(ROOT, p.replace("\\", "/"))


@pytest.fixture(scope="module")
def manifest():
    df = pd.read_csv(MANIFEST)
    df["concept_vec"] = df["concept_vec"].apply(ast.literal_eval)
    return df


@pytest.fixture(scope="module")
def model():
    import tensorflow as tf
    return tf.keras.models.load_model(MODEL)


@pytest.fixture(scope="module")
def sample_images(manifest):
    """One test-split image per class, only those actually present on disk and readable."""
    test = manifest[manifest["split"] == "test"]
    out = []
    seen = set()
    for _, r in test.iterrows():
        cls = r["Class_Name"]
        if cls in seen:
            continue
        p = _fix_path(r["File_Path"])
        try:
            with open(p, "rb") as f:
                if len(f.read(1)) == 1:
                    out.append((p, int(r["label_idx"]), cls))
                    seen.add(cls)
        except OSError:
            continue
    if len(out) < 3:
        pytest.skip("fewer than 3 real test images present on disk")
    return out


def _load(path):
    import tensorflow as tf
    x = tf.cast(dp._decode_resize(path), tf.float32)  # exact training-time preprocessing, 0-255 scale
    return tf.expand_dims(x, 0)


# --------------------------------------------------------------------------
# 1. Scripts import cleanly
# --------------------------------------------------------------------------
def test_core_scripts_import():
    import evaluate  # noqa: F401
    import gradcam  # noqa: F401
    assert len(dp.SELECTED_CLASSES) == 15
    assert set(dp.CONCEPT_MAP) == set(dp.SELECTED_CLASSES) == set(dp.LABEL_MAP)


# --------------------------------------------------------------------------
# 2. Manifest = single source of truth
# --------------------------------------------------------------------------
@needs_manifest
def test_manifest_columns_and_totals(manifest):
    for col in ["Class_Name", "File_Name", "File_Path", "split", "label_idx", "concept_vec"]:
        assert col in manifest.columns, f"missing column {col}"
    assert len(manifest) == EXPECTED_TOTAL
    assert manifest["split"].value_counts().to_dict() == EXPECTED_SPLITS


@needs_manifest
def test_manifest_classes_match_pipeline(manifest):
    assert set(manifest["Class_Name"]) == set(dp.SELECTED_CLASSES)
    for split in ["train", "val", "test"]:
        assert set(manifest[manifest["split"] == split]["Class_Name"]) == set(dp.SELECTED_CLASSES), \
            f"split '{split}' is missing classes"


@needs_manifest
def test_manifest_label_and_concept_consistency(manifest):
    assert (manifest["Class_Name"].map(dp.LABEL_MAP) == manifest["label_idx"]).all()
    expected = manifest["Class_Name"].map(lambda c: dp.CONCEPT_MAP[c])
    assert all(a == b for a, b in zip(manifest["concept_vec"], expected))


@needs_manifest
def test_manifest_cap_and_no_duplicate_paths(manifest):
    assert manifest.groupby("Class_Name").size().max() <= MAX_PER_CLASS
    assert not manifest.duplicated("File_Path").any()


@needs_manifest
def test_manifest_split_is_stratified_close_to_70_15_15(manifest):
    frac = manifest["split"].value_counts(normalize=True)
    assert abs(frac["train"] - 0.70) < 0.02
    assert abs(frac["val"] - 0.15) < 0.02
    assert abs(frac["test"] - 0.15) < 0.02


# --------------------------------------------------------------------------
# 3. Committed evaluation reports == what the README claims
# --------------------------------------------------------------------------
@needs_clean_report
def test_clean_report_matches_readme():
    r = json.load(open(EVAL_CLEAN))
    assert r["split"] == "test" and r["corrupted"] is False
    assert r["keras"]["n_samples"] == EXPECTED_SPLITS["test"]
    assert abs(r["keras"]["accuracy"] - EXPECTED_CLEAN_ACC) < ACC_TOL
    if "tflite" in r:
        assert abs(r["tflite"]["accuracy"] - EXPECTED_INT8_ACC) < ACC_TOL


@needs_clean_report
def test_clean_report_internal_consistency():
    r = json.load(open(EVAL_CLEAN))["keras"]
    cm = np.array(r["confusion_matrix"])
    assert cm.shape == (15, 15)
    assert cm.sum() == r["n_samples"]
    assert abs(np.trace(cm) / cm.sum() - r["accuracy"]) < 1e-6
    macro_f1 = r["report"]["macro avg"]["f1-score"]
    assert 0.92 < macro_f1 < 0.94, f"macro F1 {macro_f1} no longer matches README (0.932)"


@needs_corrupt_report
def test_corrupt_report_documents_the_robustness_gap():
    """Pins the KNOWN weakness. If this ever fails because accuracy improved,
    update the README + EXPECTED_CORRUPT_ACC deliberately."""
    r = json.load(open(EVAL_CORRUPT))
    assert r["corrupted"] is True
    assert abs(r["keras"]["accuracy"] - EXPECTED_CORRUPT_ACC) < ACC_TOL
    assert r["keras"]["accuracy"] < 0.2, "corrupted accuracy is near chance; README must keep saying so"


@needs_clean_report
def test_int8_model_is_not_deployable_state():
    """Guards against claiming the TFLite model is deployable while the
    committed report shows a >5pp accuracy loss."""
    if not os.path.exists(TFLITE):
        pytest.skip(f"TFLite model not found at {TFLITE}")
    r = json.load(open(EVAL_CLEAN))
    if "tflite" not in r:
        pytest.skip("no tflite section in eval_test.json")
    assert r["tflite"]["accuracy"] - r["keras"]["accuracy"] < -0.05
    assert os.path.exists(TFLITE)


# --------------------------------------------------------------------------
# 4. Trained model contract (what Grad-CAM and any demo rely on)
# --------------------------------------------------------------------------
@needs_model
def test_model_io_contract(model):
    assert model.input_shape == (None, dp.IMG_SIZE, dp.IMG_SIZE, 3)
    assert [o.shape[-1] for o in model.outputs] == [4, 15], \
        "output order must be [concepts(4), disease(15)]; gradcam uses disease_head_index=1"


@needs_model
def test_gradcam_layer_is_valid(model):
    from gradcam import LAST_CONV_LAYER_NAME
    layer = model.get_layer(LAST_CONV_LAYER_NAME)
    assert len(layer.output.shape) == 4
    assert layer.output.shape[1] == layer.output.shape[2] == 7


# --------------------------------------------------------------------------
# 5. Real-image behaviour: correct class + working Grad-CAM
# --------------------------------------------------------------------------
@needs_model
def test_predictions_valid_and_mostly_correct_on_real_samples(model, sample_images):
    hits = 0
    for path, label, cls in sample_images:
        concepts, disease = model.predict(_load(path), verbose=0)
        assert np.isclose(disease.sum(), 1.0, atol=1e-3)
        assert ((concepts >= 0) & (concepts <= 1)).all()
        assert np.isfinite(disease).all()
        hits += int(np.argmax(disease[0]) == label)
    assert hits / len(sample_images) >= 0.8, f"only {hits}/{len(sample_images)} sample leaves classified correctly"


@needs_model
def test_gradcam_heatmap_properties(model, sample_images):
    from gradcam import LAST_CONV_LAYER_NAME, make_gradcam_heatmap
    for path, label, cls in sample_images:
        x = _load(path)
        out = make_gradcam_heatmap(model, x, LAST_CONV_LAYER_NAME)
        assert isinstance(out, tuple) and len(out) == 2, "make_gradcam_heatmap returns (heatmap, pred_index)"
        hm, idx = out
        assert hm.shape == (7, 7)
        assert np.isfinite(hm).all()
        assert hm.min() >= 0.0 and hm.max() <= 1.0 + 1e-6
        assert hm.max() > 0.99, "heatmap must be normalised to a max of ~1"
        assert hm.std() > 0.05, "heatmap is flat; no spatial signal"
        _, disease = model.predict(x, verbose=0)
        assert idx == int(np.argmax(disease[0])), "default pred_index must equal the model's top-1 class"


@needs_model
def test_gradcam_is_class_discriminative_and_deterministic(model, sample_images):
    from gradcam import LAST_CONV_LAYER_NAME, make_gradcam_heatmap
    path, _, _ = sample_images[0]
    x = _load(path)
    _, disease = model.predict(x, verbose=0)
    order = np.argsort(disease[0])[::-1]
    hm_a, _ = make_gradcam_heatmap(model, x, LAST_CONV_LAYER_NAME, pred_index=int(order[0]))
    hm_b, _ = make_gradcam_heatmap(model, x, LAST_CONV_LAYER_NAME, pred_index=int(order[1]))
    hm_a2, _ = make_gradcam_heatmap(model, x, LAST_CONV_LAYER_NAME, pred_index=int(order[0]))
    assert np.abs(hm_a - hm_b).mean() > 0.05, "heatmap does not change with the target class (activation-average bug)"
    assert np.allclose(hm_a, hm_a2, atol=1e-5), "Grad-CAM is not deterministic for the same input/class"


@needs_model
def test_preprocessing_contract_is_raw_0_255(model, sample_images):
    """The model must be fed 0-255 floats (no /255). Feeding 0-1 should
    visibly degrade confidence; this pins the contract the demo must follow."""
    import tensorflow as tf
    path, label, _ = sample_images[0]
    x = _load(path)
    assert float(tf.reduce_max(x)) > 50.0
    _, d_ok = model.predict(x, verbose=0)
    _, d_bad = model.predict(x / 255.0, verbose=0)
    assert d_ok[0][label] > d_bad[0][label], \
        "0-255 input should give higher true-class probability than 0-1 input"
