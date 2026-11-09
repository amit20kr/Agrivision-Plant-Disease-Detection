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
