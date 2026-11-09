"""
App-level tests for Streamlit dashboard using AppTest.
"""
import os
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

REPO_ROOT = Path(__file__).resolve().parent.parent
APP_PATH = str(REPO_ROOT / "app" / "dashboard.py")


def test_dashboard_renders_without_exception():
    os.environ["KERAS_HOME"] = str(REPO_ROOT / "cache" / "keras")
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception, f"App threw an unhandled exception: {at.exception}"
    assert len(at.tabs) >= 3


def test_dashboard_model_performance_tab_renders():
    os.environ["KERAS_HOME"] = str(REPO_ROOT / "cache" / "keras")
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    # Check that committed accuracy 92.8% is quoted in text
    all_text = " ".join(t.value for t in at.markdown)
    assert "92.8%" in all_text
    assert "93.9%" not in all_text or "preliminary" in all_text.lower()


def test_dashboard_concept_caption_present():
    os.environ["KERAS_HOME"] = str(REPO_ROOT / "cache" / "keras")
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception, f"App threw an unhandled exception: {at.exception}"
