"""
Field analysis utilities for PlantDoc dataset (Course Extension Phase 1).

Provides:
- PlantDoc indexing and class mapping
- Perceptual hash clustering and train/test leakage detection
- 436-dimensional feature extraction and prediction
- Accuracy with bootstrap 95% CI (supports n_bootstrap and n_boot)
- Exact vs approximate mapping evaluation
- Calibration (ECE) and quality gate analysis
- Out-of-distribution (other-crop) AUROC analysis
- Grouped linear probe cross-validation
- Strict JSON report saving and visualization generation
"""
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import imagehash
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold
import tensorflow as tf

import data_pipeline as dp
import inference_utils as iu


def index_plantdoc(plantdoc_root: Union[str, Path], class_map_path: Union[str, Path]) -> pd.DataFrame:
    """Scan Cropped-PlantDoc directory and map to project classes according to class_map_path."""
    mapping = pd.read_csv(class_map_path)
    map_dict = {r["plantdoc_folder"]: r for _, r in mapping.iterrows()}

    rows = []
    root = Path(plantdoc_root)
    for split in ["train", "test"]:
        split_dir = root / split
        if not split_dir.exists():
            continue
        for folder in sorted(os.listdir(split_dir)):
            folder_path = split_dir / folder
            if not folder_path.is_dir() or folder not in map_dict:
                continue
            meta = map_dict[folder]
            role = meta["role"]
            model_class = meta.get("model_class", "")
            quality = meta.get("quality", "n/a")
            label_idx = dp.LABEL_MAP.get(model_class, -1) if pd.notna(model_class) and model_class else -1

            for fname in sorted(os.listdir(folder_path)):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                full_path = str((folder_path / fname).resolve())
                rows.append({
                    "file_path": full_path,
                    "file_name": fname,
                    "split": split,
                    "plantdoc_folder": folder,
                    "model_class": model_class if pd.notna(model_class) else "",
                    "label_idx": int(label_idx),
                    "quality": quality,
                    "role": role,
                })
    return pd.DataFrame(rows)


def compute_image_hashes(df: pd.DataFrame, max_hamming_dist: int = 6) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute perceptual hashes for all images and cluster near-duplicates.
    Returns (df_with_clusters, leakage_df)."""
    df = df.copy()
    hashes = []
    for p in df["file_path"]:
        try:
            with Image.open(p) as img:
                h = imagehash.phash(img)
                hashes.append(str(h))
        except Exception:
            hashes.append("")
    df["phash"] = hashes

    valid_mask = df["phash"] != ""
    valid_df = df[valid_mask].copy()
    valid_hashes = [imagehash.hex_to_hash(h) for h in valid_df["phash"]]
    n = len(valid_hashes)

    parent = list(range(n))

    def find(i: int) -> int:
        path = []
        while parent[i] != i:
            path.append(i)
            i = parent[i]
        for node in path:
            parent[node] = i
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            if valid_hashes[i] - valid_hashes[j] <= max_hamming_dist:
                union(i, j)

    cluster_ids = [find(i) for i in range(n)]
    valid_df["cluster_id"] = cluster_ids
    df["cluster_id"] = -1
    df.loc[valid_df.index, "cluster_id"] = valid_df["cluster_id"]

    target_df = df[df["role"] == "target"]
    leakage_clusters = []
    for cid, grp in target_df.groupby("cluster_id"):
        if cid == -1:
            continue
        splits = set(grp["split"])
        if "train" in splits and "test" in splits:
            leakage_clusters.append(cid)

    leakage_df = df[df["cluster_id"].isin(leakage_clusters)].copy()
    return df, leakage_df


def extract_features_and_predictions(
    model: tf.keras.Model,
    df: pd.DataFrame,
    cfg: Optional[iu.GateConfig] = None,
    batch_size: int = 32,
    cache_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Run model on df images, collect quality gate metrics, predictions, and exact 436-d features."""
    if cache_path:
        csv_file = cache_path + ".csv"
        npy_file = cache_path + "_feats.npy"
        if os.path.exists(csv_file) and os.path.exists(npy_file):
            out_df = pd.read_csv(csv_file)
            feats = np.load(npy_file)
            return out_df, feats

    cfg = cfg or iu.GateConfig.load()
    d_layer = model.get_layer("D")

    # D's input tensor is already the exact 436-d concatenation of GAP(432) and C(4)
    feat_extractor = tf.keras.Model(
        inputs=model.inputs,
        outputs=[d_layer.input, d_layer.output]
    )

    rows = []
    all_feats = []
    batch_imgs = []
    batch_meta = []

    def flush_batch() -> None:
        if not batch_imgs:
            return
        x_tensor = tf.constant(np.stack(batch_imgs, axis=0), dtype=tf.float32)
        feats_out, d_out = feat_extractor(x_tensor, training=False)
        feats_np = feats_out.numpy()
        d_np = d_out.numpy()
        all_feats.append(feats_np)

        for i, meta in enumerate(batch_meta):
            probs = d_np[i]
            order = np.argsort(probs)[::-1]
            p_top = float(probs[order[0]])
            p_second = float(probs[order[1]])
            entropy = float(-(probs * np.log(probs + 1e-12)).sum())
            meta.update({
                "pred_idx": int(order[0]),
                "pred_class": dp.SELECTED_CLASSES[int(order[0])],
                "confidence": p_top,
                "margin": p_top - p_second,
                "entropy": entropy,
            })
            if meta.get("label_idx", -1) >= 0:
                meta["correct"] = bool(meta["pred_idx"] == meta["label_idx"])
            else:
                meta["correct"] = False
            rows.append(meta)

        batch_imgs.clear()
        batch_meta.clear()

    for _, r in df.iterrows():
        try:
            rgb = iu.to_uint8_rgb(r["file_path"])
        except Exception:
            continue
        min_side = int(min(rgb.shape[:2]))
        x = iu.resize_for_model(rgb)
        q = iu.image_quality_metrics(iu._as_uint8(x))
        passed, _ = iu.evaluate_gate(q, min_side, cfg)

        meta = {**r.to_dict(), "gate_passed": passed, **q}
        batch_imgs.append(x)
        batch_meta.append(meta)

        if len(batch_imgs) >= batch_size:
            flush_batch()
    flush_batch()

    out_df = pd.DataFrame(rows)
    feats = np.concatenate(all_feats, axis=0) if all_feats else np.empty((0, 436), dtype=np.float32)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        out_df.to_csv(cache_path + ".csv", index=False)
        np.save(cache_path + "_feats.npy", feats)

    return out_df, feats


def compute_field_metrics(
    target_df: pd.DataFrame,
    n_bootstrap: Optional[int] = None,
    n_boot: Optional[int] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute overall accuracy, bootstrap 95% CI, exact vs approx accuracy, and ECE.
    Accepts either n_bootstrap or n_boot for convenience."""
    n_trials = n_bootstrap if n_bootstrap is not None else (n_boot if n_boot is not None else 1000)

    if len(target_df) == 0:
        return {
            "n_samples": 0, "overall_accuracy": 0.0, "ci_95": [0.0, 0.0],
            "exact_accuracy": 0.0, "approx_accuracy": 0.0, "ece": 0.0,
        }

    corrects = target_df["correct"].values.astype(float)
    acc = float(np.mean(corrects))

    rng = np.random.default_rng(seed)
    boot_accs = [
        float(np.mean(rng.choice(corrects, size=len(corrects), replace=True)))
        for _ in range(n_trials)
    ]
    ci_low = float(np.percentile(boot_accs, 2.5))
    ci_high = float(np.percentile(boot_accs, 97.5))

    exact_df = target_df[target_df["quality"] == "exact"]
    approx_df = target_df[target_df["quality"] == "approximate"]
    exact_acc = float(exact_df["correct"].mean()) if len(exact_df) > 0 else 0.0
    approx_acc = float(approx_df["correct"].mean()) if len(approx_df) > 0 else 0.0

    ece = iu.expected_calibration_error(target_df["confidence"], target_df["correct"])
    gate_rejection_rate = float((~target_df["gate_passed"]).mean())
    accepted_df = target_df[target_df["gate_passed"]]
    acc_on_accepted = float(accepted_df["correct"].mean()) if len(accepted_df) > 0 else 0.0

    conf_95 = target_df[target_df["confidence"] >= 0.95]
    coverage_95 = float(len(conf_95) / len(target_df))
    acc_95 = float(conf_95["correct"].mean()) if len(conf_95) > 0 else 0.0

    wrong_df = target_df[~target_df["correct"]]
    confident_wrong = int((wrong_df["confidence"] >= 0.9).sum())

    per_class = target_df.groupby("model_class").agg(
        n=("correct", "count"),
        accuracy=("correct", "mean"),
        quality=("quality", "first")
    ).round(4).to_dict(orient="index")

    return {
        "n_samples": len(target_df),
        "overall_accuracy": acc,
        "ci_95": [ci_low, ci_high],
        "exact_accuracy": exact_acc,
        "n_exact": len(exact_df),
        "approx_accuracy": approx_acc,
        "n_approx": len(approx_df),
        "ece": float(ece),
        "gate_rejection_rate": gate_rejection_rate,
        "acc_on_accepted": acc_on_accepted,
        "coverage_at_conf_95": coverage_95,
        "acc_at_conf_95": acc_95,
        "confident_wrong_ge_09": confident_wrong,
        "total_wrong": len(wrong_df),
        "per_class": per_class,
    }


def compute_negative_crop_metrics(neg_df: pd.DataFrame, target_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute out-of-distribution other-crop confident answer rate and AUROC."""
    if len(neg_df) == 0 or len(target_df) == 0:
        return {
            "n_other_crop": len(neg_df), "other_crop_answered_at_08": 0.0,
            "auroc_confidence": 0.5, "auroc_margin": 0.5, "auroc_entropy": 0.5,
        }

    rate_ge_08 = float((neg_df["confidence"] >= 0.8).mean())

    y_true = np.concatenate([np.ones(len(target_df)), np.zeros(len(neg_df))])
    confs = np.concatenate([target_df["confidence"].values, neg_df["confidence"].values])
    margins = np.concatenate([target_df["margin"].values, neg_df["margin"].values])
    entropies = np.concatenate([-target_df["entropy"].values, -neg_df["entropy"].values])

    auroc_conf = float(roc_auc_score(y_true, confs))
    auroc_margin = float(roc_auc_score(y_true, margins))
    auroc_entropy = float(roc_auc_score(y_true, entropies))

    return {
        "n_other_crop": len(neg_df),
        "other_crop_answered_at_08": rate_ge_08,
        "auroc_confidence": auroc_conf,
        "auroc_margin": auroc_margin,
        "auroc_entropy": auroc_entropy,
    }


def linear_probe_cv(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
) -> float:
    """Evaluate a linear probe on frozen features using grouped cross-validation."""
    unique_groups = len(np.unique(groups))
    actual_splits = min(n_splits, unique_groups)
    if actual_splits < 2:
        clf = LogisticRegression(max_iter=500, solver="lbfgs")
        clf.fit(features, labels)
        return float(clf.score(features, labels))

    gkf = GroupKFold(n_splits=actual_splits)
    scores = []
    for train_idx, val_idx in gkf.split(features, labels, groups=groups):
        clf = LogisticRegression(max_iter=500, solver="lbfgs")
        clf.fit(features[train_idx], labels[train_idx])
        acc = clf.score(features[val_idx], labels[val_idx])
        scores.append(acc)
    return float(np.mean(scores))


def _clean_for_json(data: Any) -> Any:
    """Recursively convert float NaNs or numpy types to JSON-safe primitives."""
    if isinstance(data, dict):
        return {k: _clean_for_json(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [_clean_for_json(v) for v in data]
    if isinstance(data, float):
        return None if math.isnan(data) or math.isinf(data) else round(data, 6)
    if isinstance(data, (np.floating, np.integer)):
        val = data.item()
        return None if isinstance(val, float) and (math.isnan(val) or math.isinf(val)) else val
    if isinstance(data, np.ndarray):
        return _clean_for_json(data.tolist())
    return data


def save_field_gap_report(
    target_metrics: Dict[str, Any],
    neg_metrics: Dict[str, Any],
    linear_probe_acc: float,
    leakage_count: int,
    lab_control_acc: float,
    out_path: Union[str, Path] = "advisory/field_gap_report.json",
) -> Dict[str, Any]:
    """Assemble and save the final field gap report in strict JSON format."""
    report = {
        "pipeline_control": {
            "lab_control_accuracy": lab_control_acc,
            "status": "PASS" if lab_control_acc >= 0.85 else "FAIL",
        },
        "leakage": {
            "near_duplicate_train_test_leakage_clusters": leakage_count,
        },
        "field_target_metrics": target_metrics,
        "out_of_distribution_negatives": neg_metrics,
        "feature_adaptation_baseline": {
            "frozen_linear_probe_cv_acc": linear_probe_acc,
        },
        "reproduction_gate": {
            "expected_accuracy": 0.189,
            "tolerance": 0.040,
            "reproduced_accuracy": target_metrics["overall_accuracy"],
            "status": "PASS" if abs(target_metrics["overall_accuracy"] - 0.189) <= 0.040 else "FAIL",
        },
    }
    clean_report = _clean_for_json(report)
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(clean_report, f, indent=2)
    return clean_report


def generate_field_figures(
    target_df: pd.DataFrame,
    neg_df: pd.DataFrame,
    out_dir: Union[str, Path] = "advisory/field_figures",
) -> List[str]:
    """Generate and save six publication-ready figures."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    saved_files = []

    # 1. Overall Accuracy with Bootstrap CI
    fig, ax = plt.subplots(figsize=(6, 4))
    acc = target_df["correct"].mean() if len(target_df) else 0.0
    ax.bar(["Frozen Model (Field)"], [acc * 100], color="#2b5c8f", width=0.4)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 50)
    ax.set_title("Field Accuracy on PlantDoc (Target Classes)")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    f1 = out_path / "01_overall_accuracy.png"
    plt.tight_layout()
    plt.savefig(f1, dpi=150)
    plt.close()
    saved_files.append(str(f1))

    # 2. Exact vs Approximate Mapping Accuracy
    fig, ax = plt.subplots(figsize=(6, 4))
    exact_acc = target_df[target_df["quality"] == "exact"]["correct"].mean() * 100 if len(target_df) else 0.0
    approx_acc = target_df[target_df["quality"] == "approximate"]["correct"].mean() * 100 if len(target_df) else 0.0
    ax.bar(["Exact Mapping", "Approximate Mapping"], [exact_acc, approx_acc], color=["#1f77b4", "#aec7e8"], width=0.5)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Field Accuracy by Class Mapping Rigor")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    f2 = out_path / "02_exact_vs_approx.png"
    plt.tight_layout()
    plt.savefig(f2, dpi=150)
    plt.close()
    saved_files.append(str(f2))

    # 3. Per-class Accuracy
    fig, ax = plt.subplots(figsize=(8, 6))
    if len(target_df):
        per_class = target_df.groupby("model_class")["correct"].mean().sort_values() * 100
        ax.barh(per_class.index, per_class.values, color="#3b7a57")
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Per-Class Accuracy on PlantDoc")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    f3 = out_path / "03_per_class_accuracy.png"
    plt.tight_layout()
    plt.savefig(f3, dpi=150)
    plt.close()
    saved_files.append(str(f3))

    # 4. Confidence Distribution (Correct vs Wrong)
    fig, ax = plt.subplots(figsize=(7, 4))
    if len(target_df):
        ax.hist(target_df[target_df["correct"]]["confidence"], bins=20, alpha=0.6, label="Correct", color="green")
        ax.hist(target_df[~target_df["correct"]]["confidence"], bins=20, alpha=0.6, label="Wrong", color="red")
        ax.legend()
    ax.set_xlabel("Model Confidence")
    ax.set_ylabel("Count")
    ax.set_title("Confidence Distribution on Field Data")
    ax.grid(alpha=0.3)
    f4 = out_path / "04_confidence_distribution.png"
    plt.tight_layout()
    plt.savefig(f4, dpi=150)
    plt.close()
    saved_files.append(str(f4))

    # 5. Reliability / Calibration Curve
    fig, ax = plt.subplots(figsize=(6, 5))
    if len(target_df):
        bins = np.linspace(0, 1, 11)
        binned = pd.cut(target_df["confidence"], bins=bins)
        rel = target_df.groupby(binned, observed=True).agg(
            mean_conf=("confidence", "mean"),
            acc=("correct", "mean")
        ).dropna()
        ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
        ax.plot(rel["mean_conf"], rel["acc"], marker="o", color="#d62728", label="Frozen Model")
        ax.legend()
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Empirical Accuracy")
    ax.set_title("Calibration Curve (PlantDoc Target)")
    ax.grid(alpha=0.3)
    f5 = out_path / "05_calibration_curve.png"
    plt.tight_layout()
    plt.savefig(f5, dpi=150)
    plt.close()
    saved_files.append(str(f5))

    # 6. ROC Curves for OOD Detection
    fig, ax = plt.subplots(figsize=(6, 5))
    if len(target_df) and len(neg_df):
        y_true = np.concatenate([np.ones(len(target_df)), np.zeros(len(neg_df))])
        for metric_name, scores, color in [
            ("Confidence", np.concatenate([target_df["confidence"], neg_df["confidence"]]), "blue"),
            ("Margin", np.concatenate([target_df["margin"], neg_df["margin"]]), "green"),
            ("-Entropy", np.concatenate([-target_df["entropy"], -neg_df["entropy"]]), "orange"),
        ]:
            fpr, tpr, _ = roc_curve(y_true, scores)
            auc = roc_auc_score(y_true, scores)
            ax.plot(fpr, tpr, color=color, label=f"{metric_name} (AUC={auc:.2f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.legend()
    ax.set_xlabel("False Positive Rate (Other Crop as Target)")
    ax.set_ylabel("True Positive Rate (Target as Target)")
    ax.set_title("OOD Leaf Separation ROC Curves")
    ax.grid(alpha=0.3)
    f6 = out_path / "06_ood_roc_curves.png"
    plt.tight_layout()
    plt.savefig(f6, dpi=150)
    plt.close()
    saved_files.append(str(f6))

    return saved_files
