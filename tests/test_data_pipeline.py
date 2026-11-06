"""
Run this BEFORE spending any GPU time. Every assertion here failing means
"don't bother training yet, the data pipeline itself is broken."
"""
import numpy as np

import data_pipeline as dp


def test_manifest_covers_every_split(manifest_df):
    assert set(manifest_df["split"].unique()) == {"train", "val", "test"}
    for split in ["train", "val", "test"]:
        assert (manifest_df["split"] == split).sum() > 0, f"'{split}' split is empty"


def test_split_ratios_are_approximately_70_15_15(manifest_df):
    n = len(manifest_df)
    fracs = manifest_df["split"].value_counts(normalize=True)
    assert abs(fracs["train"] - 0.70) < 0.05
    assert abs(fracs["val"] - 0.15) < 0.05
    assert abs(fracs["test"] - 0.15) < 0.05


def test_no_exact_path_duplicated_across_splits(manifest_df):
    """The cheapest possible leakage check: same file path in two splits.
    (Near-duplicate, non-identical-path leakage is check_duplicates.py's job --
    this test only catches the trivial case of the same file listed twice.)"""
    dupes = manifest_df[manifest_df.duplicated("File_Path", keep=False)]
    assert dupes.empty, f"Found {len(dupes)} rows with a File_Path duplicated across splits"


def test_every_class_present_in_every_split(manifest_df):
    """A class with zero test examples silently reports 100% or 0% recall for
    that class -- worth catching before training, not after."""
    counts = manifest_df.groupby(["split", "Class_Name"]).size().unstack(fill_value=0)
    assert (counts > 0).all().all(), f"Some class is missing from some split:\n{counts}"


def test_label_idx_matches_label_map(manifest_df):
    expected = manifest_df["Class_Name"].map(dp.LABEL_MAP)
    assert (manifest_df["label_idx"] == expected).all()


def test_concept_vec_matches_concept_map_exactly(manifest_df):
    for _, row in manifest_df.iterrows():
        assert row["concept_vec"] == dp.CONCEPT_MAP[row["Class_Name"]], (
            f"concept_vec drifted from CONCEPT_MAP for class {row['Class_Name']}"
        )


def test_split_is_reproducible_given_same_seed(patch_dp_classes, synthetic_dataset_dir):
    """Same seed must produce the identical split every time. If this fails,
    two different training runs (or you vs. Antigravity) could silently
    train/evaluate on different data despite reading the 'same' pipeline."""
    df = dp.build_dataframe(synthetic_dataset_dir)
    df = dp.undersample(df, cap=200, seed=42)
    split_a = dp.stratified_split(df, seed=42).sort_values("File_Path").reset_index(drop=True)

    df2 = dp.build_dataframe(synthetic_dataset_dir)
    df2 = dp.undersample(df2, cap=200, seed=42)
    split_b = dp.stratified_split(df2, seed=42).sort_values("File_Path").reset_index(drop=True)

    assert split_a["File_Path"].equals(split_b["File_Path"])
    assert split_a["split"].equals(split_b["split"])


def test_get_tf_dataset_shapes_and_no_nans(manifest_df):
    train_df = manifest_df[manifest_df["split"] == "train"].reset_index(drop=True)
    ds = dp.get_tf_dataset(train_df, "train", batch_size=4, shuffle=True)
    imgs, (concepts, labels) = next(iter(ds))

    assert imgs.shape[1:] == (dp.IMG_SIZE, dp.IMG_SIZE, 3)
    assert concepts.shape[1] == 4
    assert not np.isnan(imgs.numpy()).any(), "NaN pixel values in decoded image batch"
    assert not np.isnan(concepts.numpy()).any()
    assert labels.numpy().max() < len(dp.SELECTED_CLASSES)
    assert labels.numpy().min() >= 0


def test_undersample_never_exceeds_cap(patch_dp_classes, synthetic_dataset_dir):
    df = dp.build_dataframe(synthetic_dataset_dir)
    df = dp.undersample(df, cap=5, seed=42)
    counts = df.groupby("Class_Name").size()
    assert (counts <= 5).all()
