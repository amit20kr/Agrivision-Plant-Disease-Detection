"""
Same invariants as test_data_pipeline.py, but run against the REAL
manifest.csv / trained model at the repo root once Antigravity has produced
them (data_pipeline.py step 4, training step 6). Auto-skips if those files
don't exist yet, so `pytest tests/` is always safe to run at any project
stage without editing this file.

Run this as a GATE:
  - after step 4 (manifest.csv exists)      -> real data integrity
  - after step 5 (duplicate_report.csv)     -> real leakage check
  - after step 6 (best_model.keras exists)  -> real model sanity
Don't proceed to the next step if any of these fail.
"""
import ast
import os

import pandas as pd
import pytest
import tensorflow as tf

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
MANIFEST_PATH = os.path.join(REPO_ROOT, "manifest.csv")
DUP_REPORT_PATH = os.path.join(REPO_ROOT, "duplicate_report.csv")
MODEL_PATH = os.path.join(REPO_ROOT, "best_model.keras")

# Set this to your real dataset size for a sanity floor once you know it
# (the hackathon dataset was 2952 images after undersampling).
MIN_EXPECTED_TOTAL_IMAGES = 500


needs_manifest = pytest.mark.skipif(
    not os.path.exists(MANIFEST_PATH), reason="manifest.csv not generated yet (run data_pipeline.py first)"
)
needs_dup_report = pytest.mark.skipif(
    not os.path.exists(DUP_REPORT_PATH), reason="duplicate_report.csv not generated yet (run check_duplicates.py first)"
)
needs_model = pytest.mark.skipif(
    not os.path.exists(MODEL_PATH), reason="best_model.keras not trained yet"
)


@needs_manifest
def test_real_manifest_has_every_image_readable():
    df = pd.read_csv(MANIFEST_PATH)
    missing = [p for p in df["File_Path"] if not os.path.exists(p)]
    assert not missing, f"{len(missing)} manifest rows point at files that don't exist, e.g. {missing[:5]}"


@needs_manifest
def test_real_manifest_size_floor():
    df = pd.read_csv(MANIFEST_PATH)
    assert len(df) >= MIN_EXPECTED_TOTAL_IMAGES, (
        f"Manifest only has {len(df)} rows -- did the dataset download/extraction actually complete?"
    )


@needs_manifest
def test_real_manifest_no_cross_split_path_duplicates():
    df = pd.read_csv(MANIFEST_PATH)
    dupes = df[df.duplicated("File_Path", keep=False)]
    assert dupes.empty, f"{len(dupes)} rows share a File_Path across splits"


@needs_manifest
def test_real_manifest_concept_vectors_parse():
    df = pd.read_csv(MANIFEST_PATH)
    for v in df["concept_vec"]:
        parsed = ast.literal_eval(v)
        assert len(parsed) == 4
        assert all(x in (0.0, 1.0) for x in parsed)


@needs_dup_report
def test_real_no_unresolved_test_split_leakage():
    """This is the hard gate before trusting ANY test-set accuracy number.
    If this fails, resolve it (PLAN.md step 13) and re-run before reporting
    results in the README."""
    dup_df = pd.read_csv(DUP_REPORT_PATH)
    if dup_df.empty:
        return
    test_leaks = dup_df[dup_df["split_a"].eq("test") | dup_df["split_b"].eq("test")]
    assert test_leaks.empty, (
        f"{len(test_leaks)} near-duplicate pairs touch the test split. "
        "Resolve before reporting test accuracy in the README (PLAN.md step 13)."
    )


@needs_model
def test_real_model_loads_and_has_two_outputs():
    model = tf.keras.models.load_model(MODEL_PATH)
    assert len(model.outputs) == 2, "Expected [concept_output, disease_output]"


@needs_model
def test_real_model_output_widths_match_expected_classes():
    import sys
    sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
    import data_pipeline as dp

    model = tf.keras.models.load_model(MODEL_PATH)
    widths = [o.shape[-1] for o in model.outputs]
    assert 4 in widths, "No 4-wide concept head found in model outputs"
    assert len(dp.SELECTED_CLASSES) in widths, (
        f"No {len(dp.SELECTED_CLASSES)}-wide disease head found in model outputs -- "
        "does the trained model match the current SELECTED_CLASSES list?"
    )
