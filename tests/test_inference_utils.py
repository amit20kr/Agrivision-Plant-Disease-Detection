"""
Unit tests for scripts/inference_utils.py and advisory/advisory_table.csv.

Pure-logic tests run anywhere (no model, no dataset). Real-model tests auto-skip
if best_model.keras or the sample images are missing.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import data_pipeline as dp  # noqa: E402
import inference_utils as iu  # noqa: E402

MODEL = os.path.join(ROOT, "best_model.keras")
MANIFEST = os.path.join(ROOT, "manifest.csv")


def _photo_like(size=224, seed=0):
    """Smooth, leaf-ish synthetic image: gentle texture, mid brightness, little noise."""
    import cv2
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1, (size, size, 3)).astype(np.float32)
    base = cv2.GaussianBlur(base, (0, 0), 3.0)
    base = (base - base.min()) / (base.max() - base.min())
    img = 70 + 120 * base
    yy, xx = np.mgrid[0:size, 0:size]
    img += 15 * np.sin(xx / 6.0)[..., None]
    return np.clip(img, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------- advisory table
def test_advisory_covers_all_15_classes_and_is_valid():
    df = iu.load_advisory()
    problems = iu.validate_advisory(df)
    assert problems == [], problems
    assert set(df.index) == set(dp.SELECTED_CLASSES)


def test_advisory_does_not_contain_dosages():
    """Guidance must stay general: no chemical rates or product doses."""
    df = iu.load_advisory()
    text = " ".join(df[["what_to_do_now", "prevention", "when_to_seek_help"]].astype(str).sum(axis=1)).lower()
    for token in [" ml ", "ml/", " g/", "grams per", "tablespoon", "tbsp", "ppm", "%"]:
        assert token not in text, f"advisory text contains a dosage-like token: {token!r}"


def test_advisory_category_not_derived_from_concept_vector():
    """Leaf Mold shares the 'Complex Pathogen' concept with the viruses but is fungal."""
    df = iu.load_advisory()
    assert df.loc["Tomato_Leaf_Mold", "category"] == "fungal"
    assert df.loc["Tomato__Tomato_mosaic_virus", "category"] == "viral"
    assert df.loc["Tomato_Spider_mites_Two_spotted_spider_mite", "category"] == "pest"
    assert df.loc["Tomato_Late_blight", "category"] == "oomycete"


def test_lookup_unknown_class_falls_back_without_error():
    df = iu.load_advisory()
    adv = iu.lookup_advisory(df, "Maize___Rust")
    assert adv["category"] == "unknown"
    assert "extension" in adv["what_to_do_now"].lower()


def test_validate_advisory_detects_missing_class():
    df = iu.load_advisory().drop(index="Tomato_healthy")
    assert any("Tomato_healthy" in p for p in iu.validate_advisory(df))


# ----------------------------------------------------------------- preprocessing contract
def test_resize_matches_pipeline_decode_resize(tmp_path):
    from PIL import Image
    arr = (np.random.default_rng(1).random((300, 260, 3)) * 255).astype(np.uint8)
    p = tmp_path / "x.jpg"
    Image.fromarray(arr).save(p, quality=95)
    ours = iu.resize_for_model(iu.to_uint8_rgb(str(p)))
    ref = dp._decode_resize(str(p)).numpy()
    assert ours.shape == (224, 224, 3) and ours.dtype == np.float32
    assert np.allclose(ours, ref, atol=1e-3), "demo preprocessing diverged from the training pipeline"
    assert ours.max() > 50, "must stay on the raw 0-255 scale"


def test_to_uint8_rgb_handles_gray_and_rgba():
    g = np.full((40, 50), 128, np.uint8)
    assert iu.to_uint8_rgb(g).shape == (40, 50, 3)
    rgba = np.zeros((40, 50, 4), np.uint8)
    assert iu.to_uint8_rgb(rgba).shape == (40, 50, 3)


# ----------------------------------------------------------------- quality gate
def test_metrics_respond_to_noise_blur_and_brightness():
    clean = _photo_like()
    m0 = iu.image_quality_metrics(clean)
    noisy = iu.corrupt_image(clean, noise_sigma=40)
    blurred = iu.corrupt_image(clean, blur_len=15)
    dark = iu.corrupt_image(clean, brightness_delta=-100)
    assert iu.image_quality_metrics(noisy)["noise_sigma"] > m0["noise_sigma"] + 15
    assert iu.image_quality_metrics(blurred)["sharpness"] < m0["sharpness"]
    assert iu.image_quality_metrics(dark)["brightness"] < m0["brightness"] - 50


def test_gate_logic_rejects_each_failure_mode():
    cfg = iu.GateConfig()
    good = {"sharpness": 1500.0, "noise_sigma": 6.0, "brightness": 120.0}
    assert iu.evaluate_gate(good, 256, cfg) == (True, [])
    assert not iu.evaluate_gate(dict(good, sharpness=10.0), 256, cfg)[0]
    assert not iu.evaluate_gate(dict(good, noise_sigma=40.0), 256, cfg)[0]
    assert not iu.evaluate_gate(dict(good, brightness=10.0), 256, cfg)[0]
    assert not iu.evaluate_gate(dict(good, brightness=250.0), 256, cfg)[0]
    assert not iu.evaluate_gate(good, 50, cfg)[0]
    passed, reasons = iu.evaluate_gate(dict(sharpness=1.0, noise_sigma=99.0, brightness=1.0), 10, cfg)
    assert not passed and len(reasons) == 4  # 'too dark' and 'too bright' are mutually exclusive


def test_gate_rejects_blank_tiny_and_heavy_noise_images():
    cfg = iu.GateConfig()
    blank = np.full((300, 300, 3), 128, np.uint8)
    assert not iu.quality_gate(blank, cfg)["passed"]
    tiny = np.zeros((40, 40, 3), np.uint8)
    assert not iu.quality_gate(tiny, cfg)["passed"]
    assert not iu.quality_gate(iu.corrupt_image(_photo_like(), noise_sigma=45), cfg)["passed"]


def test_gate_config_roundtrip(tmp_path):
    p = str(tmp_path / "gate.json")
    iu.GateConfig(min_sharpness=222.0, max_noise_sigma=9.0).save(p)
    loaded = iu.GateConfig.load(p)
    assert loaded.min_sharpness == 222.0 and loaded.max_noise_sigma == 9.0
    assert iu.GateConfig.load(str(tmp_path / "missing.json")) == iu.GateConfig()


def test_corrupt_image_zero_severity_is_identity_and_deterministic():
    img = _photo_like()
    assert np.array_equal(iu.corrupt_image(img), img)
    a = iu.corrupt_image(img, noise_sigma=20, seed=3)
    b = iu.corrupt_image(img, noise_sigma=20, seed=3)
    assert np.array_equal(a, b) and not np.array_equal(a, img)
    assert a.dtype == np.uint8


def test_bytes_and_upload_decode_identically_to_path(tmp_path):
    import io
    from PIL import Image
    arr = (np.random.default_rng(2).random((120, 140, 3)) * 255).astype(np.uint8)
    p = tmp_path / "y.jpg"
    Image.fromarray(arr).save(p, quality=90)
    raw = p.read_bytes()
    from_path = iu.to_uint8_rgb(str(p))
    assert np.array_equal(from_path, iu.to_uint8_rgb(raw))
    buf = io.BytesIO(raw)
    assert np.array_equal(from_path, iu.to_uint8_rgb(buf))
    assert buf.tell() == 0, "upload stream must be rewound after reading"


# ----------------------------------------------------------------- thresholding and calibration
def test_decide_threshold_and_top3():
    probs = np.zeros(15)
    probs[3], probs[5], probs[7] = 0.6, 0.3, 0.1
    hi = iu.decide(probs, 0.5)
    lo = iu.decide(probs, 0.7)
    assert hi["status"] == "ok" and hi["pred_idx"] == 3 and hi["label"] == dp.SELECTED_CLASSES[3]
    assert lo["status"] == "uncertain"
    assert [c for c, _ in hi["top3"]] == [dp.SELECTED_CLASSES[i] for i in (3, 5, 7)]


def test_coverage_accuracy_table_and_pick_threshold():
    frame = pd.DataFrame({"confidence": [0.95, 0.9, 0.6, 0.5], "correct": [True, True, False, False]})
    t = iu.coverage_accuracy_table(frame, thresholds=[0.4, 0.7, 0.99])
    assert t["n_accepted"].tolist() == [4, 2, 0]
    assert t["coverage"].tolist() == [1.0, 0.5, 0.0]
    assert t.loc[0, "selective_accuracy"] == 0.5 and t.loc[1, "selective_accuracy"] == 1.0
    assert np.isnan(t.loc[2, "selective_accuracy"])
    assert iu.pick_threshold(t, target_accuracy=0.95, min_coverage=0.5) == 0.7
    assert iu.pick_threshold(t, target_accuracy=0.95, min_coverage=0.9) is None


def test_expected_calibration_error_behaviour():
    assert iu.expected_calibration_error([1.0, 1.0], [True, True]) == pytest.approx(0.0)
    assert iu.expected_calibration_error([0.9] * 10, [True] * 5 + [False] * 5) == pytest.approx(0.4, abs=1e-6)


# ----------------------------------------------------------------- real model + real images
def _real_sample_paths(n=6):
    if not (os.path.exists(MODEL) and os.path.exists(MANIFEST)):
        return []
    df = pd.read_csv(MANIFEST)
    df = df[df["split"] == "test"].groupby("Class_Name").head(1)
    out = []
    for _, r in df.iterrows():
        p = os.path.join(ROOT, r["File_Path"].replace("\\", "/"))
        if os.path.exists(p):
            out.append((p, r["Class_Name"], int(r["label_idx"])))
    return out[:n]


@pytest.fixture(scope="module")
def real_model():
    if not os.path.exists(MODEL):
        pytest.skip("best_model.keras missing")
    import tensorflow as tf
    return tf.keras.models.load_model(MODEL)


@pytest.fixture(scope="module")
def real_samples():
    s = _real_sample_paths()
    if len(s) < 2:
        pytest.skip("real sample images not present on disk")
    return s


def test_analyze_image_real_leaves_end_to_end(real_model, real_samples):
    adv = iu.load_advisory()
    hits = 0
    for path, cls, _ in real_samples:
        res = iu.analyze_image(real_model, path, advisory=adv, cfg=iu.GateConfig(), threshold=0.5)
        assert res["status"] in {"ok", "uncertain", "rejected_quality"}
        if res["status"] == "ok":
            hits += int(res["label"] == cls)
            assert res["advisory"]["class_name"] == res["label"]
            assert res["heatmap"].shape == (7, 7)
            assert res["overlay"].shape == (224, 224, 3) and res["overlay"].dtype == np.uint8
            assert res["disclaimer"]
    assert hits >= max(1, len(real_samples) - 1), "pipeline mislabels most real sample leaves"


def test_analyze_image_rejects_heavily_corrupted_input(real_model, real_samples):
    path, _, _ = real_samples[0]
    noisy = iu.corrupt_image(iu.to_uint8_rgb(path), noise_sigma=50)
    res = iu.analyze_image(real_model, noisy, cfg=iu.GateConfig(), threshold=0.5)
    assert res["status"] == "rejected_quality"
    assert "retake" in res["message"].lower()


def test_analyze_image_high_threshold_gives_uncertain(real_model, real_samples):
    path, _, _ = real_samples[0]
    res = iu.analyze_image(real_model, path, cfg=iu.GateConfig(), threshold=1.01, with_gradcam=False)
    assert res["status"] == "uncertain"
    assert res["advisory"]["category"] == "unknown"
    assert len(res["top3"]) == 3


def test_predict_frame_columns_and_accuracy(real_model, real_samples):
    rows = []
    for path, cls, idx in real_samples:
        rows.append({"File_Path": path, "Class_Name": cls, "label_idx": idx})
    out = iu.predict_frame(real_model, pd.DataFrame(rows), cfg=iu.GateConfig(), batch_size=4)
    for col in ["pred_idx", "confidence", "margin", "entropy", "correct", "sharpness",
                "noise_sigma", "brightness", "gate_passed"]:
        assert col in out.columns
    assert len(out) == len(real_samples)
    assert out["confidence"].between(0, 1).all()
    assert out["correct"].mean() >= 0.8


def test_predict_frame_accepts_callable_corruption(real_model, real_samples):
    rows = [{"File_Path": p, "Class_Name": c, "label_idx": i} for p, c, i in real_samples[:2]]
    clean = iu.predict_frame(real_model, pd.DataFrame(rows), cfg=iu.GateConfig())
    noisy = iu.predict_frame(real_model, pd.DataFrame(rows), cfg=iu.GateConfig(),
                             corrupt=lambda im: iu.corrupt_image(im, noise_sigma=45, seed=1))
    assert (noisy["noise_sigma"] > clean["noise_sigma"] + 10).all()
    assert not noisy["gate_passed"].any()
