"""
Field evaluation utilities for PlantDoc dataset (course extension Phase 1).

Indexes Cropped-PlantDoc, detects perceptual hash duplicates and train/test leakage,
runs the frozen CBM model, extracts 436-d features, and computes field metrics:
accuracy with bootstrap CI, exact vs approximate mapping, ECE, quality gate rejection,
out-of-distribution other-crop confident rate, AUROC, and grouped linear probe CV.
"""
import hashlib
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
import tensorflow as tf

import data_pipeline as dp
import inference_utils as iu


def index_plantdoc(plantdoc_root: str, class_map_path: str) -> pd.DataFrame:
    """Scan Cropped-PlantDoc directory and map to project classes according to class_map_path."""
    mapping = pd.read_csv(class_map_path)
    map_dict = {r["plantdoc_folder"]: r for _, r in mapping.iterrows()}

    rows = []
    root = Path(plantdoc_root)
    for split in ["train", "test"]:
        split_dir = root / split
        if not split_dir.exists():
            continue
        for folder in os.listdir(split_dir):
            folder_path = split_dir / folder
            if not folder_path.is_dir() or folder not in map_dict:
                continue
            meta = map_dict[folder]
            role = meta["role"]
            model_class = meta.get("model_class", "")
            quality = meta.get("quality", "n/a")
            label_idx = dp.LABEL_MAP.get(model_class, -1) if pd.notna(model_class) and model_class else -1

            for fname in os.listdir(folder_path):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                full_path = str(folder_path / fname)
                rows.append({
                    "file_path": full_path,
                    "split": split,
                    "plantdoc_folder": folder,
                    "model_class": model_class if pd.notna(model_class) else "",
                    "label_idx": label_idx,
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

    # Cluster images with Hamming distance <= max_hamming_dist
    valid_mask = df["phash"] != ""
    valid_df = df[valid_mask].copy()
    valid_hashes = [imagehash.hex_to_hash(h) for h in valid_df["phash"]]
    n = len(valid_hashes)

    parent = list(range(n))
    def find(i):
        path = []
        while parent[i] != i:
            path.append(i)
            i = parent[i]
        for node in path:
            parent[node] = i
        return i

    def union(i, j):
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

    # Check leakage: clusters containing both train and test target images
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
    """Run model on df images, collect quality gate metrics, predictions, and 436-d features."""
    if cache_path and os.path.exists(cache_path + ".parquet") and os.path.exists(cache_path + "_feats.npy"):
        out_df = pd.read_parquet(cache_path + ".parquet")
        feats = np.load(cache_path + "_feats.npy")
        return out_df, feats

    cfg = cfg or iu.GateConfig.load()
    # D's input is the 436-d concatenation of [GAP(432-d), c_out(4-d)].
    # Obtain the 436-d vector directly from D's input tensor and also keep d_out.
    d_layer = model.get_layer("D")
    c_layer = model.get_layer("C")

    # Feature extractor model: inputs -> [436-d feature, c_out(4-d), d_out(15-d)]
    feat_extractor = tf.keras.Model(
        inputs=model.inputs,
        outputs=[d_layer.input, c_layer.output, d_layer.output]
    )

    rows = []
    all_feats = []
    batch_imgs = []
    batch_meta = []

    def flush_batch():
        if not batch_imgs:
            return
        x_tensor = tf.constant(np.stack(batch_imgs, axis=0), dtype=tf.float32)
        gap_out, c_out, d_out = feat_extractor(x_tensor, training=False)
        gap_np = gap_out.numpy()
        c_np = c_out.numpy()
        d_np = d_out.numpy()
        # 436-d feature vector = concat([gap, c_out])
        batch_feats = np.concatenate([gap_np, c_np], axis=-1)
        all_feats.append(batch_feats)

        for i, meta in enumerate(batch_meta):
            probs = d_np[i]
            order = np.argsort(probs)[::-1]
            p_top = float(probs[order[0]])
            p_second = float(probs[order[1]])
            entropy = float(-(probs * np.log(probs + 1e-12)).sum())
            meta.update({
                "pred_idx": int(order[0]),
                "confidence": p_top,
                "margin": p_top - p_second,
                "entropy": entropy,
            })
            if meta["label_idx"] >= 0:
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
        out_df.to_parquet(cache_path + ".parquet", index=False)
        np.save(cache_path + "_feats.npy", feats)

    return out_df, feats


def compute_field_metrics(target_df: pd.DataFrame, n_bootstrap: int = 1000) -> Dict:
    """Compute overall accuracy, bootstrap 95% CI, exact vs approx accuracy, and ECE."""
    corrects = target_df["correct"].values.astype(float)
    acc = float(np.mean(corrects))

    # Bootstrap 95% CI
    rng = np.random.default_rng(42)
    boot_accs = [float(np.mean(rng.choice(corrects, size=len(corrects), replace=True)))
                 for _ in range(n_bootstrap)]
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

    # Selective accuracy at threshold >= 0.95
    conf_95 = target_df[target_df["confidence"] >= 0.95]
    coverage_95 = float(len(conf_95) / len(target_df))
    acc_95 = float(conf_95["correct"].mean()) if len(conf_95) > 0 else 0.0

    # Confident wrong answers (conf >= 0.9 and wrong)
    wrong_df = target_df[~target_df["correct"]]
    confident_wrong = int((wrong_df["confidence"] >= 0.9).sum())

    # Per-class summary
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


def compute_negative_crop_metrics(neg_df: pd.DataFrame, target_df: pd.DataFrame) -> Dict:
    """Compute out-of-distribution other-crop confident answer rate and AUROC."""
    rate_ge_08 = float((neg_df["confidence"] >= 0.8).mean()) if len(neg_df) > 0 else 0.0

    # Binary classification: 1 for target (in-distribution), 0 for negative (out-of-distribution)
    y_true = np.concatenate([np.ones(len(target_df)), np.zeros(len(neg_df))])
    confs = np.concatenate([target_df["confidence"].values, neg_df["confidence"].values])
    margins = np.concatenate([target_df["margin"].values, neg_df["margin"].values])
    # Lower entropy means more confident -> use -entropy for AUROC
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
    gkf = GroupKFold(n_splits=n_splits)
    scores = []
    for train_idx, val_idx in gkf.split(features, labels, groups=groups):
        clf = LogisticRegression(max_iter=500, solver="lbfgs")
        clf.fit(features[train_idx], labels[train_idx])
        acc = clf.score(features[val_idx], labels[val_idx])
        scores.append(acc)
    return float(np.mean(scores))
