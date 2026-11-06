"""
Fills the biggest gap in the original project: neither notebook ever reports
validation/test accuracy, per-class metrics, or an evaluation of the
"robustness" test set that was specifically built (with MotionBlur +
GaussNoise) to prove the project's Track A claim. The README's only
performance number is TRAINING accuracy (88.8%), while val accuracy plateaus
around 67% -- a large, undisclosed generalization gap.

This script is the single place that produces the numbers that belong in the
README: per-class precision/recall/F1, a confusion matrix, clean-val
accuracy, corrupted-test accuracy (the actual "robustness" evidence), and a
FP32-vs-INT8 accuracy delta after quantization (to confirm the TFLite
conversion didn't silently degrade accuracy).

Usage:
    python evaluate.py --model cbm_model.keras --manifest manifest.csv --split test \
        --tflite edge_model.tflite --out_dir eval_report/
"""
import argparse
import ast
import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

import data_pipeline as dp

CLASS_NAMES = dp.SELECTED_CLASSES
CONCEPT_NAMES = ["Is Healthy", "Has Spots", "Is Blighted", "Complex Pathogen"]


def load_manifest_split(manifest_csv, split):
    df = pd.read_csv(manifest_csv)
    df = df[df["split"] == split].copy()
    df["concept_vec"] = df["concept_vec"].apply(ast.literal_eval)
    return df


def evaluate_keras(model, ds, class_names, y_true_override=None):
    """Runs full-dataset inference and returns predictions + sklearn report."""
    y_true, y_pred = [], []
    for imgs, (concepts, labels) in ds:
        _, disease_pred = model.predict(imgs, verbose=0)
        y_pred.extend(np.argmax(disease_pred, axis=1).tolist())
        y_true.extend(labels.numpy().tolist())

    report = classification_report(
        y_true, y_pred, labels=list(range(len(class_names))), target_names=class_names, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    accuracy = float(np.mean(np.array(y_true) == np.array(y_pred)))
    return {"accuracy": accuracy, "report": report, "confusion_matrix": cm.tolist(),
            "n_samples": len(y_true)}


def evaluate_tflite(tflite_path, ds, class_names):
    """Same protocol as evaluate_keras but through the INT8 interpreter, so the
    reported quantized accuracy is directly comparable to the FP32 number."""
    try:
        # tf.lite.Interpreter is deprecated as of TF 2.20+ in favor of LiteRT.
        # Prefer the non-deprecated path if available rather than leaving a
        # silent future break for whatever TF version Antigravity runs this with.
        from ai_edge_litert.interpreter import Interpreter
        interpreter = Interpreter(model_path=tflite_path)
    except ImportError:
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    disease_output_index = None
    for od in output_details:
        if od["shape"][-1] == len(class_names):
            disease_output_index = od["index"]
    if disease_output_index is None:
        raise ValueError("Could not identify the disease-head output tensor by shape; "
                          "inspect interpreter.get_output_details() manually.")

    y_true, y_pred = [], []
    for imgs, (concepts, labels) in ds:
        for i in range(imgs.shape[0]):
            single_img = tf.expand_dims(imgs[i], 0)
            interpreter.set_tensor(input_details[0]["index"], single_img)
            interpreter.invoke()
            disease_pred = interpreter.get_tensor(disease_output_index)
            y_pred.append(int(np.argmax(disease_pred[0])))
            y_true.append(int(labels[i].numpy()))

    accuracy = float(np.mean(np.array(y_true) == np.array(y_pred)))
    # Diagnostic parity with evaluate_keras(): a bare accuracy number cannot
    # distinguish "uniformly worse everywhere" from "predictions collapsed
    # onto 1-2 classes" (the failure mode this project's corrupted-test eval
    # actually found). Report both so a large FP32-vs-INT8 delta is
    # diagnosable from eval_report/*.json without re-running anything.
    report = classification_report(
        y_true, y_pred, labels=list(range(len(class_names))), target_names=class_names, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    pred_class_counts = {class_names[c]: int((np.array(y_pred) == c).sum())
                          for c in range(len(class_names)) if (np.array(y_pred) == c).sum() > 0}
    n_classes_predicted = len(pred_class_counts)
    print(f"[DIAGNOSTIC] TFLite predictions used {n_classes_predicted}/{len(class_names)} classes: "
          f"{pred_class_counts}")
    return {"accuracy": accuracy, "n_samples": len(y_true), "report": report,
            "confusion_matrix": cm.tolist(), "n_classes_predicted": n_classes_predicted,
            "predicted_class_counts": pred_class_counts}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="path to cbm_model.keras")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--tflite", default=None, help="optional path to edge_model.tflite for parity check")
    parser.add_argument("--out_dir", default="eval_report")
    parser.add_argument("--corrupt", action="store_true",
                         help="apply the same MotionBlur/GaussNoise corruption used to build "
                              "the 'robustness' test set -- REQUIRED to actually substantiate "
                              "the Track A claim, since this transform is otherwise never run "
                              "through model.evaluate anywhere in the project")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = load_manifest_split(args.manifest, args.split)
    print(f"Loaded {len(df)} rows for split='{args.split}'"
          f"{' (corrupted)' if args.corrupt else ''}")

    augment_fn = None
    if args.corrupt:
        import albumentations as A
        import cv2
        transform = A.Compose([
            A.MotionBlur(blur_limit=7, p=1.0),
            A.GaussNoise(p=1.0),
            A.RandomBrightnessContrast(brightness_limit=-0.3, contrast_limit=0.3, p=1.0),
        ])

        def augment_fn(img):
            def _apply(img_np):
                return transform(image=img_np.astype(np.uint8))["image"]
            out = tf.numpy_function(_apply, [tf.cast(img, tf.uint8)], tf.uint8)
            out.set_shape([dp.IMG_SIZE, dp.IMG_SIZE, 3])
            return out

    ds = dp.get_tf_dataset(df, args.split, batch_size=32, shuffle=False, augment_fn=augment_fn)

    model = tf.keras.models.load_model(args.model)
    keras_results = evaluate_keras(model, ds, CLASS_NAMES)
    print(f"[Keras/FP32] accuracy on {args.split}"
          f"{' (corrupted)' if args.corrupt else ' (clean)'}: {keras_results['accuracy']:.4f}")

    results = {"split": args.split, "corrupted": args.corrupt, "keras": keras_results}

    if args.tflite:
        ds_tflite = dp.get_tf_dataset(df, args.split, batch_size=32, shuffle=False, augment_fn=augment_fn)
        tflite_results = evaluate_tflite(args.tflite, ds_tflite, CLASS_NAMES)
        print(f"[TFLite/INT8] accuracy: {tflite_results['accuracy']:.4f} "
              f"(delta vs FP32: {tflite_results['accuracy'] - keras_results['accuracy']:+.4f})")
        delta = tflite_results['accuracy'] - keras_results['accuracy']
        if abs(delta) > 0.05:
            print(f"[FINDING] INT8 quantization moved accuracy by {delta:+.4f} (>5pp) -- "
                  f"predictions used only {tflite_results['n_classes_predicted']}/{len(CLASS_NAMES)} "
                  f"classes. This is a quantization-calibration failure, not an acceptable edge-deployment "
                  f"trade-off; see predicted_class_counts in the JSON report before claiming this model is "
                  f"deployable.")
        results["tflite"] = tflite_results

    out_path = os.path.join(args.out_dir, f"eval_{args.split}{'_corrupt' if args.corrupt else ''}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote full report to {out_path}")


if __name__ == "__main__":
    main()
