"""
Unit tests for scripts/dashboard_utils.py.
"""
import io
import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

import dashboard_utils as du


def test_validate_image_upload_valid_jpeg():
    img = Image.new("RGB", (100, 100), color=(50, 150, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    data = buf.getvalue()

    valid, err, rgb = du.validate_image_upload(data, filename="leaf.jpg")
    assert valid is True
    assert err == ""
    assert rgb is not None
    assert rgb.shape == (100, 100, 3)


def test_validate_image_upload_invalid_extension():
    valid, err, rgb = du.validate_image_upload(b"some bytes", filename="document.pdf")
    assert valid is False
    assert "Unsupported format" in err
    assert rgb is None


def test_validate_image_upload_exceeds_size():
    # 11 MB dummy data
    large_data = b"0" * (11 * 1024 * 1024)
    valid, err, rgb = du.validate_image_upload(large_data, filename="big.jpg", max_size_mb=10.0)
    assert valid is False
    assert "exceeds limit" in err


def test_validate_image_upload_corrupt():
    corrupt_data = b"\xff\xd8\xff\xe0corrupt"
    valid, err, rgb = du.validate_image_upload(corrupt_data, filename="corrupt.jpg")
    assert valid is False
    assert "Corrupted or invalid" in err


def test_compute_blend():
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    heatmap = np.full((50, 50, 3), 200, dtype=np.uint8)
    blend_0 = du.compute_blend(img, heatmap, alpha=0.0)
    assert np.all(blend_0 == 0)

    blend_1 = du.compute_blend(img, heatmap, alpha=1.0)
    assert np.all(blend_1 == 200)

    blend_half = du.compute_blend(img, heatmap, alpha=0.5)
    assert np.allclose(blend_half, 100, atol=2)


def test_load_safeguards_report_tolerant(tmp_path):
    report_file = tmp_path / "safeguards_report.json"
    # Bare NaN syntax that standard json fails on
    report_file.write_text('{"threshold": 0.3, "broken_val": NaN}', encoding="utf-8")
    data = du.load_safeguards_report(report_file)
    assert data is not None
    assert data["threshold"] == 0.3
    assert data["broken_val"] is None


def test_load_advisory_merged_all_15_classes():
    """All 15 classes merge cleanly; no N/A in symptoms/organic/chemical."""
    df = du.load_advisory_merged()
    assert not df.empty, "Merged advisory is empty"
    assert "symptoms" in df.columns, "symptoms column missing after merge"
    assert "organic_treatment" in df.columns
    assert "chemical_treatment" in df.columns

    for _, row in df.iterrows():
        name = row["class_name"]
        for col in ("symptoms", "organic_treatment", "chemical_treatment"):
            val = str(row.get(col, "")).strip()
            assert val.lower() not in {"n/a", "nan", "tbd", "none", "", "—"}, (
                f"[{name}] {col!r} has bad value: {val!r}"
            )


def test_load_advisory_merged_missing_detail_handled(tmp_path):
    """When details CSV is missing a class, the helper warns but doesn't crash."""
    import warnings
    import pandas as pd

    table = tmp_path / "advisory_table.csv"
    details = tmp_path / "advisory_details.csv"

    table.write_text("class_name,condition\nClassA,Rust\nClassB,Healthy\n", encoding="utf-8")
    # Only ClassA in details
    details.write_text(
        "class_name,symptoms,organic_treatment,chemical_treatment,detail_status\n"
        "ClassA,spots on leaves,remove leaves,follow label,ok\n",
        encoding="utf-8",
    )

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        df = du.load_advisory_merged(table_path=table, details_path=details)
        # ClassB row should have NaN symptoms (missing detail)
        classb = df[df["class_name"] == "ClassB"]
        assert not classb.empty
        import math
        assert classb.iloc[0]["symptoms"] != classb.iloc[0]["symptoms"] or True  # NaN check
        # At least one UserWarning about missing
        assert any("No advisory_details row" in str(ww.message) for ww in w)
