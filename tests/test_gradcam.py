"""
Regression test for the original bug: the inference notebook's heatmap
averaged raw activations with no gradient weighting, so it was NOT
class-discriminative -- it would render identically no matter which disease
the model predicted. This test fails loudly if that regresses.
"""
import numpy as np
import pytest

import gradcam


def test_heatmap_shape_and_range(dummy_model):
    img = np.random.rand(1, 224, 224, 3).astype(np.float32)
    heatmap, pred_idx = gradcam.make_gradcam_heatmap(dummy_model, img, "last_conv")

    assert heatmap.ndim == 2
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0 + 1e-6
    assert 0 <= pred_idx < 3


def test_heatmap_is_class_discriminative(dummy_model):
    """The core fix: forcing pred_index to different classes must produce
    measurably different heatmaps. If this ever returns ~identical heatmaps
    again, the gradient weighting has been broken/reverted."""
    img = np.random.rand(1, 224, 224, 3).astype(np.float32)
    heatmap_0, _ = gradcam.make_gradcam_heatmap(dummy_model, img, "last_conv", pred_index=0)
    heatmap_1, _ = gradcam.make_gradcam_heatmap(dummy_model, img, "last_conv", pred_index=1)

    diff = np.abs(heatmap_0 - heatmap_1).sum()
    assert diff > 1e-6, "Heatmaps for different classes are identical -- not class-discriminative"


def test_missing_layer_name_raises_immediately():
    """Calling without a verified layer name must fail fast and loud, not
    silently fall back to a guessed layer (that was the original bug's root
    cause -- a fragile 'scan backwards' auto-detect)."""
    with pytest.raises(ValueError):
        gradcam.make_gradcam_heatmap(None, None, last_conv_layer_name=None)


def test_find_candidate_conv_layers_lists_the_real_layer(dummy_model):
    candidates = gradcam.find_candidate_conv_layers(dummy_model)
    names = [name for name, _ in candidates]
    assert "last_conv" in names
