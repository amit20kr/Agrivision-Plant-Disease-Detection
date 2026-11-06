"""
Replaces the training notebook's Colab-only auth flow:

    from google.colab import files
    if not os.path.exists('/root/.kaggle/kaggle.json'):
        uploaded = files.upload()   # <-- interactive, breaks outside Colab

This version works identically whether run in Colab, a local machine, or
inside Antigravity: it looks for credentials in the standard kaggle.json
location OR in KAGGLE_USERNAME / KAGGLE_KEY environment variables (the
`kaggle` CLI natively supports both). No prompt, no manual upload step.

Setup (one-time, not committed to git):
    Get an API token from https://www.kaggle.com/settings -> API -> Create New Token
    Either:
      a) place the downloaded kaggle.json at ~/.kaggle/kaggle.json (chmod 600), or
      b) export KAGGLE_USERNAME=... and KAGGLE_KEY=... as environment variables

Usage:
    python kaggle_download.py --dataset emmarex/plantdisease --dest data/raw
"""
import argparse
import os
import pathlib
import subprocess
import sys


def ensure_kaggle_credentials():
    kaggle_json = pathlib.Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        return
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return
    raise RuntimeError(
        "No Kaggle credentials found.\n"
        f"  Option A: place your token at {kaggle_json} (chmod 600)\n"
        "  Option B: set KAGGLE_USERNAME and KAGGLE_KEY environment variables\n"
        "Get a token from https://www.kaggle.com/settings -> API -> Create New Token"
    )


def download_dataset(dataset="emmarex/plantdisease", dest="data/raw"):
    ensure_kaggle_credentials()
    pathlib.Path(dest).mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "datasets", "download", "-d", dataset, "-p", dest, "--unzip", "-q"]
    print(f"[OPERATION] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"[SUCCESS] Dataset extracted to {dest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="emmarex/plantdisease")
    parser.add_argument("--dest", default="data/raw")
    args = parser.parse_args()
    try:
        download_dataset(args.dataset, args.dest)
    except RuntimeError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
