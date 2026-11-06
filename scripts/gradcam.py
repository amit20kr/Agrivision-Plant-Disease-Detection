"""
True gradient-weighted Grad-CAM for the CBM (concept, disease) two-headed model.

Replaces the inference notebook's heatmap, which averaged raw feature-map
activations (np.mean(features, axis=-1)) with NO gradient weighting. That is
a plain activation map, not Grad-CAM: it is not class-discriminative and
would look identical regardless of which disease the model predicted.

This module computes the actual Grad-CAM: gradients of the target class
logit w.r.t. the last conv feature map, globally average-pooled into
per-channel importance weights, then a weighted sum + ReLU.

Usage:
    from gradcam import make_gradcam_heatmap, LAST_CONV_LAYER_NAME
    heatmap = make_gradcam_heatmap(model, img_batch, LAST_CONV_LAYER_NAME, pred_index=disease_idx)
"""
import numpy as np
import tensorflow as tf

# Hardcode this once you've confirmed it for your trained model via
# `python gradcam.py --inspect path/to/model.keras`. Do NOT re-run the
# "scan backwards for a 4D-output layer" trick at inference time in
# production -- it is fragile (MobileNetV3 has many 4D-shaped depthwise/
# expansion blocks; the "first one found" is not guaranteed to be the
# semantically final feature map) and silently picks a different layer
# if the backbone or Keras version changes.
LAST_CONV_LAYER_NAME = "activation_17"  # verified via `python gradcam.py --inspect best_model.keras`
# see gradcam_inspect.txt (repo root) -- "activation_17" is the final activation
# before global pooling in this trained MobileNetV3Small backbone (alpha=0.75).


def find_candidate_conv_layers(model):
    """List all layers with 4D output, for manual inspection -- NOT for silent
    runtime use. Print this once per model version and hardcode the winner."""
    candidates = []
    for layer in model.layers:
        try:
            shape = layer.output.shape
        except Exception:
            continue
        if len(shape) == 4:
            candidates.append((layer.name, tuple(shape)))
    return candidates


def make_gradcam_heatmap(model, img_batch, last_conv_layer_name, pred_index=None, disease_head_index=1):
    """
    model: the two-headed CBM Keras model, outputs = [concepts, disease]
    img_batch: np.ndarray or tf.Tensor, shape (1, H, W, 3), already preprocessed
               exactly as at training time
    last_conv_layer_name: str, REQUIRED, must be an explicit, verified layer name
    pred_index: int or None. If None, uses the model's own top-1 disease prediction.
    disease_head_index: index of the disease-logit output in model.outputs
                         (1 for this CBM: outputs=[c_out, d_out])

    Returns: heatmap as np.ndarray, shape (H, W), values in [0, 1]
    """
    if not last_conv_layer_name:
        raise ValueError(
            "last_conv_layer_name is required. Run find_candidate_conv_layers(model), "
            "pick the correct feature layer once, and hardcode it -- do not "
            "auto-detect this on every inference call."
        )

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output] + list(model.outputs),
    )

    img_batch = tf.cast(img_batch, tf.float32)

    with tf.GradientTape() as tape:
        outputs = grad_model(img_batch)
        conv_output = outputs[0]
        preds = outputs[1:]
        disease_preds = preds[disease_head_index]

        if pred_index is None:
            pred_index = tf.argmax(disease_preds[0])
        class_channel = disease_preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_output)
    if grads is None:
        raise RuntimeError(
            f"No gradient flowed from the disease head to layer '{last_conv_layer_name}'. "
            "Confirm this layer actually feeds into the disease output path."
        )

    # Global-average-pool the gradients -> one importance weight per channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    heatmap = heatmap / (max_val + 1e-8)

    return heatmap.numpy(), int(pred_index)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--inspect":
        model_path = sys.argv[2]
        m = tf.keras.models.load_model(model_path)
        for name, shape in find_candidate_conv_layers(m):
            print(f"{name:30s} {shape}")
        print("\nPick the LAST semantically-final entry above (usually the final "
              "activation before global pooling) and hardcode it as "
              "LAST_CONV_LAYER_NAME in this file.")
