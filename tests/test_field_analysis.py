"""
Unit tests for scripts/field_analysis.py.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from PIL import Image

import field_analysis as fa


def test_compute_field_metrics_synthetic():
    # 10 samples: 2 correct, 8 wrong (20% accuracy)
    df = pd.DataFrame({
        "correct": [True, True] + [False] * 8,
        "quality": ["exact"] * 5 + ["approximate"] * 5,
        "confidence": [0.95, 0.92, 0.91, 0.85, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
        "gate_passed": [True] * 9 + [False],
        "model_class": ["Tomato_healthy"] * 5 + ["Potato___Early_blight"] * 5,
    })
    # Test both keyword names
    m1 = fa.compute_field_metrics(df, n_bootstrap=100)
    m2 = fa.compute_field_metrics(df, n_boot=100)
    assert m1["overall_accuracy"] == pytest.approx(0.20)
    assert m2["overall_accuracy"] == pytest.approx(0.20)
    assert len(m1["ci_95"]) == 2
    assert m1["ci_95"][0] <= m1["overall_accuracy"] <= m1["ci_95"][1]
    assert m1["n_samples"] == 10
    assert m1["confident_wrong_ge_09"] == 1
    assert m1["gate_rejection_rate"] == pytest.approx(0.10)


def test_compute_negative_crop_metrics_synthetic():
    target_df = pd.DataFrame({
        "confidence": [0.9, 0.8, 0.7],
        "margin": [0.8, 0.6, 0.5],
        "entropy": [0.2, 0.5, 0.8],
    })
    neg_df = pd.DataFrame({
        "confidence": [0.85, 0.4, 0.3],
        "margin": [0.7, 0.2, 0.1],
        "entropy": [0.4, 1.2, 1.5],
    })
    metrics = fa.compute_negative_crop_metrics(neg_df, target_df)
    assert metrics["n_other_crop"] == 3
    assert metrics["other_crop_answered_at_08"] == pytest.approx(1 / 3)
    assert 0.0 <= metrics["auroc_confidence"] <= 1.0
    assert 0.0 <= metrics["auroc_margin"] <= 1.0
    assert 0.0 <= metrics["auroc_entropy"] <= 1.0


def test_linear_probe_cv_synthetic():
    rng = np.random.default_rng(42)
    # 20 samples, 2 classes, 436 features (exact dimension!)
    features = rng.normal(size=(20, 436))
    labels = np.array([0] * 10 + [1] * 10)
    features[:, 0] += labels * 3.0
    groups = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4] * 2)

    score = fa.linear_probe_cv(features, labels, groups, n_splits=3)
    assert 0.5 <= score <= 1.0


def test_save_field_gap_report_strict_json(tmp_path):
    out_file = tmp_path / "field_gap_report.json"
    target_m = {"overall_accuracy": 0.189, "ci_95": [0.165, 0.211], "ece": 0.45}
    neg_m = {"other_crop_answered_at_08": 0.27, "auroc_confidence": 0.54}
    fa.save_field_gap_report(
        target_metrics=target_m,
        neg_metrics=neg_m,
        linear_probe_acc=0.38,
        leakage_count=10,
        lab_control_acc=0.92,
        out_path=out_file,
    )
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "NaN" not in content
    data = json.loads(content)
    assert data["reproduction_gate"]["status"] == "PASS"


def test_generate_field_figures_synthetic(tmp_path):
    target_df = pd.DataFrame({
        "correct": [True] * 4 + [False] * 16,
        "quality": ["exact"] * 10 + ["approximate"] * 10,
        "confidence": np.linspace(0.2, 0.99, 20),
        "margin": np.linspace(0.1, 0.8, 20),
        "entropy": np.linspace(0.1, 1.5, 20),
        "model_class": ["Tomato_healthy"] * 10 + ["Potato___Early_blight"] * 10,
    })
    neg_df = pd.DataFrame({
        "confidence": np.linspace(0.1, 0.85, 10),
        "margin": np.linspace(0.05, 0.5, 10),
        "entropy": np.linspace(0.5, 2.0, 10),
    })
    figs = fa.generate_field_figures(target_df, neg_df, out_dir=tmp_path)
    assert len(figs) == 6
    for f in figs:
        assert Path(f).exists()
