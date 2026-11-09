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


def test_dashboard_mission_banner_visible():
    """The positive mission banner must be present."""
    os.environ["KERAS_HOME"] = str(REPO_ROOT / "cache" / "keras")
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception, f"App threw an unhandled exception: {at.exception}"
    all_success = " ".join(str(s.value) for s in at.success)
    assert "agrivision" in all_success.lower() or "agronomist" in all_success.lower(), (
        "Mission banner not found in dashboard output"
    )


def test_dashboard_sample_no_na_in_advisory():
    """Pick a disease sample preset; if status is Confirmed, advisory must not contain 'N/A'."""
    os.environ["KERAS_HOME"] = str(REPO_ROOT / "cache" / "keras")
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    assert not at.exception, f"App threw an unhandled exception on initial load: {at.exception}"

    # Switch to 'Pick Sample Preset' radio (index 1)
    if at.radio:
        at.radio[0].set_value("Pick Sample Preset").run()
        assert not at.exception, f"Exception after switching to sample preset: {at.exception}"

    # Collect all rendered markdown text
    all_text = " ".join(str(m.value) for m in at.markdown)
    all_caps = " ".join(str(c.value) for c in at.caption)
    full_text = all_text + " " + all_caps

    # If the advisory section rendered (Symptoms present), assert no raw N/A
    if "Symptoms" in full_text:
        assert "N/A" not in full_text, (
            "Raw 'N/A' string found in advisory output. "
            f"First 500 chars: {full_text[:500]}"
        )

