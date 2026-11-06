"""
Shared fixtures for the pretest suite.

Two kinds of tests live in this suite:
  1. Mechanism tests (most of them) -- run against a tiny SYNTHETIC dataset
     and a tiny DUMMY model. These don't need your real data or a trained
     model; they exist to catch a broken pipeline/script BEFORE you spend
     GPU time and Kaggle bandwidth on the real thing.
  2. Real-artifact tests (test_manifest_integrity.py, test_real_model.py) --
     run against your ACTUAL manifest.csv / trained model IF they exist at
     the repo root. They auto-skip if those files aren't there yet, so this
     suite is safe to run at any stage of the project.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest
from PIL import Image

SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts")
sys.path.insert(0, os.path.abspath(SCRIPTS_DIR))

import data_pipeline as dp  # noqa: E402

# A small, fast subset of classes/concepts used by every synthetic-data test,
# so we're not scanning the real 15-class map for a 12-image fixture.
FIXTURE_CLASSES = ["Tomato_healthy", "Tomato_Early_blight", "Potato___healthy"]
FIXTURE_LABEL_MAP = {k: i for i, k in enumerate(FIXTURE_CLASSES)}
FIXTURE_CONCEPT_MAP = {
    "Tomato_healthy": [1.0, 0.0, 0.0, 0.0],
    "Tomato_Early_blight": [0.0, 0.0, 1.0, 0.0],
    "Potato___healthy": [1.0, 0.0, 0.0, 0.0],
}


@pytest.fixture
def patch_dp_classes(monkeypatch):
    """Point data_pipeline's global class/label/concept maps at the small
    fixture set for the duration of a test, instead of the real 15 classes."""
    monkeypatch.setattr(dp, "SELECTED_CLASSES", FIXTURE_CLASSES)
    monkeypatch.setattr(dp, "LABEL_MAP", FIXTURE_LABEL_MAP)
    monkeypatch.setattr(dp, "CONCEPT_MAP", FIXTURE_CONCEPT_MAP)
    yield


@pytest.fixture
def synthetic_dataset_dir(tmp_path):
    """Builds a tiny fake PlantVillage-shaped folder tree: 3 classes x 20
    images each, random noise images (fast, no real download needed)."""
    base = tmp_path / "raw_dataset"
    rng = np.random.default_rng(42)
    for cls in FIXTURE_CLASSES:
        cls_dir = base / cls
        cls_dir.mkdir(parents=True)
        for i in range(20):
            arr = rng.integers(0, 255, size=(48, 48, 3), dtype=np.uint8)
            Image.fromarray(arr).save(cls_dir / f"img_{i}.jpg")
    return str(base)


@pytest.fixture
def manifest_df(patch_dp_classes, synthetic_dataset_dir):
    """Runs the real data_pipeline build->undersample->split against the
    synthetic dataset and returns the resulting manifest dataframe."""
    df = dp.build_dataframe(synthetic_dataset_dir)
    df = dp.undersample(df, cap=200)
    df = dp.stratified_split(df)
    df["label_idx"] = df["Class_Name"].map(dp.LABEL_MAP)
    df["concept_vec"] = df["Class_Name"].map(lambda c: dp.CONCEPT_MAP[c])
    return df


@pytest.fixture
def manifest_csv(manifest_df, tmp_path):
    path = str(tmp_path / "manifest.csv")
    out = manifest_df.copy()
    out.to_csv(path, index=False)
    return path


def build_dummy_cbm(n_classes=3, n_concepts=4, img_size=224):
    """A tiny stand-in for the real MobileNetV3-backed CBM: same topology
    (conv backbone -> GAP -> concept head -> concat -> disease head), cheap
    enough to build/train/convert in a CPU-only sandbox in milliseconds."""
    import tensorflow as tf
    from tensorflow.keras.layers import Concatenate, Conv2D, Dense, GlobalAveragePooling2D, Input
    from tensorflow.keras.models import Model

    inp = Input(shape=(img_size, img_size, 3), name="input")
    x = Conv2D(8, 3, activation="relu", name="last_conv")(inp)
    gap = GlobalAveragePooling2D()(x)
    c_out = Dense(n_concepts, activation="sigmoid", name="C")(gap)
    combined = Concatenate()([gap, c_out])
    d_out = Dense(n_classes, activation="softmax", name="D")(combined)
    return Model(inp, [c_out, d_out])


@pytest.fixture
def dummy_model():
    return build_dummy_cbm()
