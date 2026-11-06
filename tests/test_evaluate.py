"""
Verifies evaluate.py's mechanics against a trainable-in-seconds dummy model,
so a broken evaluation script is caught before you rely on it to score your
real model. Accuracy numbers here are meaningless (random data) -- only the
mechanics (shapes, report structure, TFLite conversion) are under test.
"""
import numpy as np
import tensorflow as tf

import evaluate as ev


def _make_synthetic_ds(n=12, n_classes=3):
    def gen():
        for _ in range(n):
            img = np.random.rand(224, 224, 3).astype(np.float32)
            concept = np.random.rand(4).astype(np.float32)
            label = np.random.randint(0, n_classes)
            yield img, (concept, label)

    return tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
            (
                tf.TensorSpec(shape=(4,), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
            ),
        ),
    ).batch(4)


def test_evaluate_keras_report_structure(dummy_model):
    ds = _make_synthetic_ds(n=12, n_classes=3)
    results = ev.evaluate_keras(dummy_model, ds, class_names=["A", "B", "C"])

    assert results["n_samples"] == 12
    assert 0.0 <= results["accuracy"] <= 1.0
    assert set(["A", "B", "C"]).issubset(results["report"].keys())
    cm = np.array(results["confusion_matrix"])
    assert cm.shape == (3, 3)
    assert cm.sum() == 12


def test_evaluate_tflite_matches_sample_count(dummy_model, tmp_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(dummy_model)
    tflite_model = converter.convert()
    tflite_path = str(tmp_path / "dummy.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    ds = _make_synthetic_ds(n=8, n_classes=3)
    results = ev.evaluate_tflite(tflite_path, ds, class_names=["A", "B", "C"])

    assert results["n_samples"] == 8
    assert 0.0 <= results["accuracy"] <= 1.0


def test_tflite_disease_head_is_correctly_identified(dummy_model, tmp_path):
    """This model has two outputs (concepts: 4-dim, disease: 3-dim). The
    TFLite eval path picks the output tensor by matching len(class_names) --
    verify it actually grabs the 3-wide disease head, not the 4-wide concept
    head, or every 'accuracy' number downstream is meaningless."""
    converter = tf.lite.TFLiteConverter.from_keras_model(dummy_model)
    tflite_model = converter.convert()
    tflite_path = str(tmp_path / "dummy.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    try:
        from ai_edge_litert.interpreter import Interpreter
        interpreter = Interpreter(model_path=tflite_path)
    except ImportError:
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    shapes = [od["shape"][-1] for od in interpreter.get_output_details()]
    assert 3 in shapes, f"Expected a 3-wide disease head in outputs, got shapes {shapes}"
