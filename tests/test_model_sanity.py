"""
Architecture-level sanity checks. These build a fresh (untrained) model and
check its wiring is correct -- catches a broken model definition in seconds
instead of discovering it after a 40-minute training run produces garbage.
"""
import numpy as np
import tensorflow as tf


def test_output_shapes_and_ranges(dummy_model):
    img = np.random.rand(2, 224, 224, 3).astype(np.float32)
    c_out, d_out = dummy_model.predict(img, verbose=0)

    assert c_out.shape == (2, 4)
    assert d_out.shape == (2, 3)

    # sigmoid concept head -> bounded [0, 1]
    assert (c_out >= 0).all() and (c_out <= 1).all()
    # softmax disease head -> each row sums to 1
    np.testing.assert_allclose(d_out.sum(axis=1), np.ones(2), atol=1e-5)


def test_gradients_flow_to_both_heads(dummy_model):
    """If either head has zero gradient, that head will never learn --
    training will run to completion and silently produce a useless output
    for that head. Cheaper to catch this on an untrained model now."""
    img = tf.random.uniform((2, 224, 224, 3))
    with tf.GradientTape() as tape:
        c_out, d_out = dummy_model(img, training=True)
        loss = tf.reduce_mean(c_out) + tf.reduce_mean(d_out[:, 0])
    grads = tape.gradient(loss, dummy_model.trainable_variables)

    none_grads = [v.name for v, g in zip(dummy_model.trainable_variables, grads) if g is None]
    assert not none_grads, f"No gradient reached these variables: {none_grads}"

    all_zero = [v.name for v, g in zip(dummy_model.trainable_variables, grads)
                if g is not None and tf.reduce_all(tf.equal(g, 0))]
    assert not all_zero, f"Gradient is exactly zero for: {all_zero}"


def test_disease_head_actually_uses_concept_head_output(dummy_model):
    """This CBM's design deliberately concatenates the concept output into the
    disease head's input (combined_features = [gap, c_out]). Verify that
    connection actually exists in the graph -- if a future refactor drops the
    concat, the model silently degrades to two independent heads."""
    d_layer = dummy_model.get_layer("D")
    input_dim = d_layer.input.shape[-1]
    gap_dim = dummy_model.get_layer(index=2).output.shape[-1]  # GAP layer
    assert input_dim > gap_dim, (
        "Disease head's input width doesn't look larger than the GAP output alone -- "
        "the concept->disease concatenation may have been dropped."
    )
