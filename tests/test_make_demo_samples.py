"""
Unit tests for scripts/make_demo_samples.py.
"""
from pathlib import Path
from PIL import Image
import make_demo_samples as mds


def test_prepare_demo_samples():
    samples = mds.prepare_demo_samples()
    assert len(samples) >= 4
    for s in samples:
        p = Path(s)
        assert p.exists()
        assert p.suffix.lower() in [".jpg", ".jpeg"]
        with Image.open(p) as img:
            assert img.size[0] > 0 and img.size[1] > 0
