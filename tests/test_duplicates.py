"""
Regression test for check_duplicates.py: must catch a real near-duplicate
pair split across train/test, and must NOT flag an unrelated image pair.
A false negative here means leakage silently inflates your reported
accuracy; a false positive means you'd waste time chasing fake leakage.
"""
import numpy as np
import pandas as pd
from PIL import Image

import check_duplicates as cd


def test_flags_true_near_duplicate_across_splits(tmp_path):
    rng = np.random.default_rng(0)
    base_img = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)

    img1 = str(tmp_path / "a.jpg")
    img2 = str(tmp_path / "b.jpg")  # near-identical to a.jpg (tiny noise)
    img3 = str(tmp_path / "c.jpg")  # unrelated

    Image.fromarray(base_img).save(img1)
    noisy = np.clip(base_img.astype(int) + rng.integers(-2, 2, base_img.shape), 0, 255).astype(np.uint8)
    Image.fromarray(noisy).save(img2)
    Image.fromarray(rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)).save(img3)

    df = pd.DataFrame([
        {"File_Path": img1, "Class_Name": "X", "split": "train"},
        {"File_Path": img2, "Class_Name": "X", "split": "test"},
        {"File_Path": img3, "Class_Name": "X", "split": "test"},
    ])

    hashes = cd.hash_all(df)
    dup_df = cd.find_cross_split_duplicates(df, hashes, threshold=5)

    assert len(dup_df) == 1
    flagged_paths = {dup_df.iloc[0]["path_a"], dup_df.iloc[0]["path_b"]}
    assert flagged_paths == {img1, img2}


def test_same_split_duplicates_are_not_flagged(tmp_path):
    """Duplicates within the SAME split aren't a leakage concern (both are
    train, or both are test) -- only cross-split pairs matter here."""
    rng = np.random.default_rng(1)
    base_img = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)

    img1 = str(tmp_path / "a.jpg")
    img2 = str(tmp_path / "b.jpg")
    Image.fromarray(base_img).save(img1)
    Image.fromarray(base_img).save(img2)  # exact duplicate, but same split

    df = pd.DataFrame([
        {"File_Path": img1, "Class_Name": "X", "split": "train"},
        {"File_Path": img2, "Class_Name": "X", "split": "train"},
    ])
    hashes = cd.hash_all(df)
    dup_df = cd.find_cross_split_duplicates(df, hashes, threshold=5)
    assert dup_df.empty
