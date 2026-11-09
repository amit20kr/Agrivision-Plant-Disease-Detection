"""
Extract sample images for the dashboard demo gallery.
Picks representative examples (healthy, diseased, noisy/blurry) from data/
or generates synthetic examples if data is unavailable.
"""
import os
import shutil
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = ROOT / "app" / "demo_samples"


def prepare_demo_samples() -> list[str]:
    """Ensure at least 5 demo samples exist in app/demo_samples/."""
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    created = []

    # Check for existing real images in data/
    candidates = [
        ("Pepper_Bell_Healthy.jpg", "data/raw/PlantVillage/Pepper__bell___healthy"),
        ("Potato_Early_Blight.jpg", "data/raw/PlantVillage/Potato___Early_blight"),
        ("Tomato_Late_Blight.jpg", "data/raw/PlantVillage/Tomato_Late_blight"),
        ("Tomato_Mosaic_Virus.jpg", "data/raw/PlantVillage/Tomato__Tomato_mosaic_virus"),
    ]

    for dest_name, src_folder in candidates:
        dest_path = DEMO_DIR / dest_name
        src_path = ROOT / src_folder
        copied = False
        if src_path.is_dir():
            for f in src_path.glob("*.jpg"):
                try:
                    with open(f, "rb") as fh:
                        if fh.read(2) == b"\xff\xd8":
                            shutil.copy(f, dest_path)
                            created.append(str(dest_path))
                            copied = True
                            break
                except Exception:
                    continue
        if not copied and not dest_path.exists():
            # Generate clean synthetic leaf photo
            arr = np.full((224, 224, 3), 40, dtype=np.uint8)
            arr[40:180, 50:170, 1] = 180  # green leaf shape
            img = Image.fromarray(arr)
            img.save(dest_path, "JPEG")
            created.append(str(dest_path))

    # Add degraded samples for testing quality gate
    blurry_path = DEMO_DIR / "Degraded_Blurry_Photo.jpg"
    if not blurry_path.exists():
        base = Image.new("RGB", (224, 224), (30, 120, 40))
        blurred = base.filter(ImageFilter.GaussianBlur(radius=15))
        blurred.save(blurry_path, "JPEG")
        created.append(str(blurry_path))

    dark_path = DEMO_DIR / "Degraded_Dark_Photo.jpg"
    if not dark_path.exists():
        dark = Image.new("RGB", (224, 224), (5, 10, 5))
        dark.save(dark_path, "JPEG")
        created.append(str(dark_path))

    print(f"[make_demo_samples] Prepared {len(list(DEMO_DIR.glob('*.jpg')))} demo samples in {DEMO_DIR}")
    return [str(p) for p in DEMO_DIR.glob("*.jpg")]


if __name__ == "__main__":
    prepare_demo_samples()
