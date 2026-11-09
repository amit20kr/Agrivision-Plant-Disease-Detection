"""
Unit tests for scripts/field_evaluation.py.
"""
import numpy as np
import pandas as pd
import pytest

import field_evaluation as fe


def test_compute_field_metrics_synthetic():
    # 10 samples: 2 correct, 8 wrong (20% accuracy)
    df = pd.DataFrame({
        "correct": [True, True] + [False] * 8,
        "quality": ["exact"] * 5 + ["approximate"] * 5,
        "confidence": [0.95, 0.92, 0.91, 0.85, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
        "gate_passed": [True] * 9 + [False],
        "model_class": ["Tomato_healthy"] * 5 + ["Potato___Early_blight"] * 5,
    })
    metrics = fe.compute_field_metrics(df, n_bootstrap=100)
    assert metrics["overall_accuracy"] == pytest.approx(0.20)
    assert len(metrics["ci_95"]) == 2
    assert metrics["ci_95"][0] <= metrics["overall_accuracy"] <= metrics["ci_95"][1]
    assert metrics["n_samples"] == 10
    assert metrics["confident_wrong_ge_09"] == 1  # 0.91 is wrong
    assert metrics["gate_rejection_rate"] == pytest.approx(0.10)


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
    metrics = fe.compute_negative_crop_metrics(neg_df, target_df)
    assert metrics["n_other_crop"] == 3
    # 1 out of 3 has conf >= 0.8
    assert metrics["other_crop_answered_at_08"] == pytest.approx(1 / 3)
    assert 0.0 <= metrics["auroc_confidence"] <= 1.0
    assert 0.0 <= metrics["auroc_margin"] <= 1.0
    assert 0.0 <= metrics["auroc_entropy"] <= 1.0


def test_linear_probe_cv_synthetic():
    rng = np.random.default_rng(42)
    # 20 samples, 2 classes, 436 features
    features = rng.normal(size=(20, 436))
    labels = np.array([0] * 10 + [1] * 10)
    # Correlate first feature with label so probe can learn
    features[:, 0] += labels * 3.0
    groups = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4] * 2)

    score = fe.linear_probe_cv(features, labels, groups, n_splits=3)
    assert 0.5 <= score <= 1.0
