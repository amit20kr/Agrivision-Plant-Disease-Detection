"""
Single source of truth for dataset construction, undersampling, and the
stratified 70/15/15 split.

Why this exists: in the original project, X_train/X_val/X_test were built
as in-memory Python lists inside the training notebook and never persisted.
The inference notebook has no way to know which images were in the held-out
test set, so there is no guarantee that evaluation numbers reported anywhere
are actually computed on unseen data, and no way for a second run (or
Antigravity) to reproduce the exact same split.

Fix: build the dataframe + split here, write it once to manifest.csv with a
'split' column (train/val/test), and have every downstream script
(training, evaluate.py, gradcam demo, check_duplicates.py) read that same
manifest instead of recomputing or hardcoding paths.
"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

TARGET_SPECIES = ["Potato", "Tomato", "Pepper"]
MAX_IMAGES_PER_CLASS = 200
IMG_SIZE = 224
SEED = 42

SELECTED_CLASSES = [
    "Pepper__bell___Bacterial_spot", "Pepper__bell___healthy", "Potato___Early_blight",
    "Potato___Late_blight", "Potato___healthy", "Tomato_Bacterial_spot",
    "Tomato_Early_blight", "Tomato_Late_blight", "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot", "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot", "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus", "Tomato_healthy",
]
LABEL_MAP = {k: i for i, k in enumerate(SELECTED_CLASSES)}

CONCEPT_MAP = {
    "Pepper__bell___Bacterial_spot":               [0.0, 1.0, 0.0, 0.0],
    "Pepper__bell___healthy":                      [1.0, 0.0, 0.0, 0.0],
    "Potato___Early_blight":                       [0.0, 0.0, 1.0, 0.0],
    "Potato___Late_blight":                        [0.0, 0.0, 1.0, 0.0],
    "Potato___healthy":                            [1.0, 0.0, 0.0, 0.0],
    "Tomato_Bacterial_spot":                       [0.0, 1.0, 0.0, 0.0],
    "Tomato_Early_blight":                         [0.0, 0.0, 1.0, 0.0],
    "Tomato_Late_blight":                          [0.0, 0.0, 1.0, 0.0],
    "Tomato_Leaf_Mold":                            [0.0, 0.0, 0.0, 1.0],
    "Tomato_Septoria_leaf_spot":                   [0.0, 1.0, 0.0, 0.0],
    "Tomato_Spider_mites_Two_spotted_spider_mite": [0.0, 1.0, 0.0, 0.0],
    "Tomato__Target_Spot":                         [0.0, 1.0, 0.0, 0.0],
    "Tomato__Tomato_YellowLeaf__Curl_Virus":       [0.0, 0.0, 0.0, 1.0],
    "Tomato__Tomato_mosaic_virus":                 [0.0, 0.0, 0.0, 1.0],
    "Tomato_healthy":                              [1.0, 0.0, 0.0, 0.0],
}
# NOTE (honesty flag, keep this comment in the repo): these 4 "concepts" are a
# deterministic re-encoding of the 15 class labels, not independently
# annotated ground truth. Every image of a given class gets the identical
# concept vector. This is a valid auxiliary/hierarchical-label training
# signal, but it is not a from-scratch Concept Bottleneck Model in the
# Koh et al. sense (where concepts are independently labeled and can vary
# within a class). Say so explicitly in the README so it survives a
# follow-up question in an interview.


def build_dataframe(base_path):
    """Scan the raw dataset folder and build the metadata dataframe."""
    all_folders = os.listdir(base_path)
    selected = [f for f in all_folders if any(s in f for s in TARGET_SPECIES)]
    rows = []
    for c in selected:
        folder_path = os.path.join(base_path, c)
        if not os.path.isdir(folder_path):
            continue
        for img in os.listdir(folder_path):
            if img.lower().endswith((".jpg", ".jpeg")):
                rows.append({
                    "File_Name": img,
                    "Class_Name": c,
                    "File_Path": os.path.join(folder_path, img),
                })
    return pd.DataFrame(rows)


def undersample(df, cap=MAX_IMAGES_PER_CLASS, seed=SEED):
    return (
        df.groupby("Class_Name")
        .apply(lambda x: x.sample(n=min(len(x), cap), random_state=seed), include_groups=False)
        .reset_index()
    )


def stratified_split(df, seed=SEED):
    """70/15/15 stratified split, returns three dataframes with a 'split' column."""
    labels = df["Class_Name"].map(LABEL_MAP)
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=labels, random_state=seed
    )
    temp_labels = temp_df["Class_Name"].map(LABEL_MAP)
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_labels, random_state=seed
    )
    train_df = train_df.copy(); train_df["split"] = "train"
    val_df = val_df.copy(); val_df["split"] = "val"
    test_df = test_df.copy(); test_df["split"] = "test"
    return pd.concat([train_df, val_df, test_df], ignore_index=True)


def export_manifest(df, out_csv="manifest.csv"):
    """Persist the exact split to disk. Every other script reads this file --
    nobody recomputes train_test_split independently again."""
    out = df.copy()
    out["label_idx"] = out["Class_Name"].map(LABEL_MAP)
    out["concept_vec"] = out["Class_Name"].map(lambda c: CONCEPT_MAP[c])
    out.to_csv(out_csv, index=False)
    return out_csv


# ---- tf.data pipeline (mirrors the notebook's augmentation, factored out so
# training and evaluation use IDENTICAL preprocessing) ----

def _decode_resize(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    return img


def get_tf_dataset(df, split_name, batch_size=32, shuffle=False, augment_fn=None):
    """
    df: manifest dataframe already filtered to one split
    augment_fn: optional callable(img_tensor) -> img_tensor for extra
                augmentation (e.g. albumentations wrapped via tf.numpy_function,
                as in the original notebook). Kept pluggable here instead of
                hardcoded so the corrupted-test-set transform and the
                training-time transform are explicit, named arguments instead
                of a string-matched branch inside a closure.
    """
    paths = df["File_Path"].values
    labels = df["label_idx"].values.astype(np.int32)
    concepts = np.stack(df["concept_vec"].values).astype(np.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, concepts, labels))
    if shuffle:
        ds = ds.shuffle(len(df), seed=SEED)

    def map_fn(path, concept, label):
        img = _decode_resize(path)
        if augment_fn is not None:
            img = augment_fn(img)
        img = tf.cast(img, tf.float32)
        return img, (concept, label)

    return ds.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", required=True, help="path to raw extracted PlantVillage 'color' dir")
    parser.add_argument("--out", default="manifest.csv")
    args = parser.parse_args()

    df = build_dataframe(args.base_path)
    print(f"Found {len(df)} raw images across {df['Class_Name'].nunique()} classes")
    df = undersample(df)
    df = stratified_split(df)
    path = export_manifest(df, args.out)
    print(f"Wrote manifest with {len(df)} rows to {path}")
    print(df.groupby(["split"]).size())
