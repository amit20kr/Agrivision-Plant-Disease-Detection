"""
Static validation tests for notebook 07 (Field Reality Check).
"""
import json
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
NB07_PATH = REPO_ROOT / "notebooks" / "07_Field_Reality_Check.ipynb"


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
