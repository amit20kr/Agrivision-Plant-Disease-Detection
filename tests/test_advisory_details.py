"""
Task 1: Validate advisory_details.csv content integrity.
Checks all 15 classes have non-empty, substantive symptoms / organic / chemical text with no N/A.
"""
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DETAILS_PATH = REPO_ROOT / "advisory" / "advisory_details.csv"
TABLE_PATH = REPO_ROOT / "advisory" / "advisory_table.csv"

SELECTED_CLASSES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy",
]

HEALTHY_CLASSES = {c for c in SELECTED_CLASSES if "healthy" in c.lower()}
DOSE_PATTERN = re.compile(
    r"\d+(\.\d+)?\s*(ml|l|g|kg|oz|lb|ppm|%|tbsp|tsp)\b", re.IGNORECASE
)
BAD_VALUES = {"n/a", "nan", "tbd", "none", "na", ""}


@pytest.fixture(scope="module")
def details_df():
    assert DETAILS_PATH.exists(), f"advisory_details.csv not found at {DETAILS_PATH}"
    df = pd.read_csv(str(DETAILS_PATH))
    # No duplicate columns
    assert len(df.columns) == len(set(df.columns)), "Duplicate columns in advisory_details.csv"
    return df


@pytest.fixture(scope="module")
def table_df():
    assert TABLE_PATH.exists(), f"advisory_table.csv not found at {TABLE_PATH}"
    return pd.read_csv(str(TABLE_PATH))


def test_details_has_exactly_15_rows(details_df):
    assert len(details_df) == 15, f"Expected 15 rows, got {len(details_df)}"


def test_details_class_names_match_selected_classes(details_df):
    names = set(details_df["class_name"].tolist())
    expected = set(SELECTED_CLASSES)
    missing = expected - names
    extra = names - expected
    assert not missing, f"Missing classes in advisory_details.csv: {missing}"
    assert not extra, f"Unexpected classes in advisory_details.csv: {extra}"


def test_details_no_duplicate_class_names(details_df):
    dupes = details_df["class_name"][details_df["class_name"].duplicated()].tolist()
    assert not dupes, f"Duplicate class names: {dupes}"


def test_details_class_names_match_advisory_table(details_df, table_df):
    detail_names = set(details_df["class_name"].tolist())
    table_names = set(table_df["class_name"].tolist())
    assert detail_names == table_names, (
        f"Mismatch: only in details={detail_names - table_names}, "
        f"only in table={table_names - detail_names}"
    )


def _check_column(df, col):
    errors = []
    for _, row in df.iterrows():
        name = row["class_name"]
        val = str(row.get(col, "")).strip()
        if val.lower() in BAD_VALUES:
            errors.append(f"[{name}] {col!r} is bad value: {val!r}")
        if len(val) < 40:
            errors.append(f"[{name}] {col!r} too short ({len(val)} chars): {val!r}")
        if DOSE_PATTERN.search(val):
            errors.append(f"[{name}] {col!r} contains a dose pattern: {val!r}")
    return errors


def test_symptoms_non_empty_no_na_no_doses(details_df):
    errors = _check_column(details_df, "symptoms")
    assert not errors, "\n".join(errors)


def test_organic_treatment_non_empty_no_na_no_doses(details_df):
    errors = _check_column(details_df, "organic_treatment")
    assert not errors, "\n".join(errors)


def test_chemical_treatment_non_empty_no_na_no_doses(details_df):
    errors = _check_column(details_df, "chemical_treatment")
    assert not errors, "\n".join(errors)


def test_healthy_chemical_starts_with_none_needed(details_df):
    errors = []
    for _, row in details_df.iterrows():
        name = row["class_name"]
        if name in HEALTHY_CLASSES:
            chem = str(row.get("chemical_treatment", "")).strip()
            if not chem.lower().startswith("none needed"):
                errors.append(f"[{name}] chemical_treatment should start with 'None needed', got: {chem[:60]!r}")
    assert not errors, "\n".join(errors)


def test_disease_chemical_contains_label(details_df):
    errors = []
    for _, row in details_df.iterrows():
        name = row["class_name"]
        if name not in HEALTHY_CLASSES:
            chem = str(row.get("chemical_treatment", "")).strip().lower()
            if "label" not in chem:
                errors.append(f"[{name}] chemical_treatment missing 'label': {chem[:80]!r}")
    assert not errors, "\n".join(errors)
