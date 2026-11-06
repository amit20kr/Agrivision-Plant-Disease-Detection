"""
PlantVillage is documented to contain near-duplicate images within several
classes (repeated shots of the same leaf/background). A random stratified
split does not check for this, so it's possible for a near-duplicate of a
training image to land in val/test, inflating reported accuracy without the
model having learned anything more general.

This script perceptual-hashes every image in the manifest and flags any pair
of near-duplicates (hash distance <= threshold) that fall in DIFFERENT
splits -- that's the leakage that actually matters for evaluation validity.

Usage:
    python check_duplicates.py --manifest manifest.csv --threshold 5 --out duplicate_report.csv
"""
import argparse
from collections import defaultdict

import imagehash
import pandas as pd
from PIL import Image


def hash_all(df, hash_size=8):
    hashes = {}
    for _, row in df.iterrows():
        try:
            with Image.open(row["File_Path"]) as img:
                hashes[row["File_Path"]] = imagehash.phash(img, hash_size=hash_size)
        except Exception as e:
            print(f"[WARN] could not hash {row['File_Path']}: {e}")
    return hashes


def find_cross_split_duplicates(df, hashes, threshold=5):
    """O(n^2) pairwise comparison. Fine for a few thousand images (this
    project's dataset, 2952 images); for much larger datasets, bucket by
    hash prefix first."""
    paths = list(hashes.keys())
    split_by_path = dict(zip(df["File_Path"], df["split"]))
    class_by_path = dict(zip(df["File_Path"], df["Class_Name"]))

    flagged = []
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            p1, p2 = paths[i], paths[j]
            if split_by_path[p1] == split_by_path[p2]:
                continue  # only cross-split pairs are a leakage concern
            dist = hashes[p1] - hashes[p2]
            if dist <= threshold:
                flagged.append({
                    "path_a": p1, "split_a": split_by_path[p1], "class_a": class_by_path[p1],
                    "path_b": p2, "split_b": split_by_path[p2], "class_b": class_by_path[p2],
                    "hash_distance": dist,
                })
    return pd.DataFrame(flagged)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--threshold", type=int, default=5,
                         help="phash hamming distance; <=5 is a common near-duplicate cutoff")
    parser.add_argument("--out", default="duplicate_report.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    print(f"Hashing {len(df)} images...")
    hashes = hash_all(df)

    dup_df = find_cross_split_duplicates(df, hashes, threshold=args.threshold)
    dup_df.to_csv(args.out, index=False)

    if len(dup_df) == 0:
        print("No cross-split near-duplicates found at this threshold.")
    else:
        n_test_contaminated = dup_df[dup_df["split_a"].eq("test") | dup_df["split_b"].eq("test")].shape[0]
        n_val_contaminated = dup_df[dup_df["split_a"].eq("val") | dup_df["split_b"].eq("val")].shape[0]
        print(f"[FLAG] {len(dup_df)} cross-split near-duplicate pairs found.")
        print(f"  -> {n_test_contaminated} involve the test split")
        print(f"  -> {n_val_contaminated} involve the val split")
        print(f"Full report written to {args.out}. Any reported test/val accuracy is "
              f"potentially inflated by this amount of leakage until resolved "
              f"(e.g. drop one side of each flagged pair, or re-split at the "
              f"near-duplicate-cluster level instead of the image level).")


if __name__ == "__main__":
    main()
