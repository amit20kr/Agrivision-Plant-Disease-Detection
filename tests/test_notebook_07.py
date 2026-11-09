"""
Static validation tests for notebook 07 (Field Reality Check).
"""
import json
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
NB07_PATH = REPO_ROOT / "notebooks" / "07_Field_Reality_Check.ipynb"
FIELD_REPORT_PATH = REPO_ROOT / "advisory" / "field_gap_report.json"

# Every headline field that the dashboard displays must be non-null
_REQUIRED_PATHS = [
    ("pipeline_control", "lab_control_accuracy"),
    ("pipeline_control", "status"),
    ("field_target_metrics", "overall_accuracy"),
    ("field_target_metrics", "ci_95"),
    ("field_target_metrics", "ece"),
    ("field_target_metrics", "gate_rejection_rate"),
    ("out_of_distribution_negatives", "auroc_confidence"),
    ("out_of_distribution_negatives", "auroc_margin"),
    ("feature_adaptation_baseline", "frozen_linear_probe_cv_acc"),
    ("reproduction_gate", "status"),
]


def test_notebook_07_file_exists():
    assert NB07_PATH.exists(), f"Notebook 07 not found at {NB07_PATH}"


def test_notebook_07_valid_json():
    content = NB07_PATH.read_text(encoding="utf-8")
    data = json.loads(content)
    assert "cells" in data
    assert len(data["cells"]) >= 20


def test_notebook_07_cell_structure_and_comments():
    data = json.loads(NB07_PATH.read_text(encoding="utf-8"))
    code_cells = [c for c in data["cells"] if c.get("cell_type") == "code"]
    assert len(code_cells) >= 15

    # Check for expected imports and checks in notebook
    full_text = " ".join("".join(c.get("source", [])) for c in code_cells)
    assert "field_analysis" in full_text
    assert "best_model.keras" in full_text
    assert "advisory/field_gap_report.json" in full_text or "field_gap_report.json" in full_text
    assert "0.189" in full_text or "reproduction" in full_text.lower()


def test_field_gap_report_headline_fields_non_null():
    """Every headline field the dashboard displays must be present and non-null."""
    assert FIELD_REPORT_PATH.exists(), "advisory/field_gap_report.json missing — run notebook 07 first"
    report = json.loads(FIELD_REPORT_PATH.read_text(encoding="utf-8"))
    nulls = []
    for section, key in _REQUIRED_PATHS:
        val = report.get(section, {}).get(key)
        if val is None:
            nulls.append(f"{section}.{key}")
    assert not nulls, f"Null headline fields in field_gap_report.json: {nulls}"

