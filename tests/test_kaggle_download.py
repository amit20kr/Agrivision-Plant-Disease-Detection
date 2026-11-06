"""
Verifies the headless credential check fails loudly with a clear message
when no credentials exist (instead of the original notebook's interactive
files.upload() prompt, which just hangs/breaks outside Colab), and passes
silently when env-var credentials are present.
"""
import pytest

import kaggle_download as kd


def test_raises_clear_error_with_no_credentials(monkeypatch, tmp_path):
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)  # no kaggle.json here

    with pytest.raises(RuntimeError, match="No Kaggle credentials found"):
        kd.ensure_kaggle_credentials()


def test_passes_with_env_var_credentials(monkeypatch, tmp_path):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.setenv("KAGGLE_USERNAME", "someuser")
    monkeypatch.setenv("KAGGLE_KEY", "somekey")
    kd.ensure_kaggle_credentials()  # must not raise


def test_passes_with_kaggle_json_present(monkeypatch, tmp_path):
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    kaggle_dir = tmp_path / ".kaggle"
    kaggle_dir.mkdir()
    (kaggle_dir / "kaggle.json").write_text('{"username":"x","key":"y"}')
    kd.ensure_kaggle_credentials()  # must not raise
