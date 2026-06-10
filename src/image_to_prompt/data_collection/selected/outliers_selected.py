"""
VM.AI — Detect and copy outliers from selected/ to selected/outliers/{cat}/.

Uses same ResNet18 + IsolationForest logic as analyze_dataset.py.
Copies only — originals stay in selected/{cat}/.
"""

import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.ensemble import IsolationForest
import torch
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights

SELECTED = Path("data/image_to_prompt/selected")
SELECTED_OUTLIERS = SELECTED / "outliers"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def _is_image(f: Path) -> bool:
    return f.suffix.lower() in IMAGE_EXTENSIONS


def collect_images() -> dict[str, list[Path]]:
    images_by_cat: dict[str, list[Path]] = defaultdict(list)
    for cat_dir in sorted(SELECTED.iterdir()):
        if not cat_dir.is_dir() or cat_dir.name == "outliers":
            continue
        cat = cat_dir.name
        for f in sorted(cat_dir.rglob("*")):
            if f.is_file() and _is_image(f):
                images_by_cat[cat].append(f)
    return dict(images_by_cat)


def main():
    if not SELECTED.is_dir():
        print("selected/ not found")
        return

    images_by_cat = collect_images()
    total_images = sum(len(v) for v in images_by_cat.values())
    print(f"Collected {total_images} images across {len(images_by_cat)} categories\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.eval()
    model.to(device)

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    total_outliers = 0

    for cat, paths in sorted(images_by_cat.items()):
        t0 = time.time()
        if len(paths) < 10:
            print(f"  {cat}: too few images ({len(paths)}), skipping")
            continue

        features = []
        valid_paths = []
        for p in paths:
            try:
                img = Image.open(p).convert("RGB")
                tensor = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    feat = model(tensor).squeeze().cpu().numpy()
                features.append(feat)
                valid_paths.append(p)
            except Exception:
                continue

        if len(features) < 10:
            print(f"  {cat}: too few valid images ({len(features)}), skipping")
            continue

        clf = IsolationForest(contamination=0.05, random_state=42)
        labels = clf.fit_predict(features)

        outlier_paths = [valid_paths[i] for i, l in enumerate(labels) if l == -1]
        total_outliers += len(outlier_paths)

        cat_out_dir = SELECTED_OUTLIERS / cat
        cat_out_dir.mkdir(parents=True, exist_ok=True)
        for op in outlier_paths:
            try:
                shutil.copy2(op, cat_out_dir / op.name)
            except Exception:
                pass

        elapsed = time.time() - t0
        print(f"  {cat}: {len(outlier_paths)} outliers copied ({elapsed:.0f}s)")
        for op in outlier_paths[:3]:
            print(f"    {op.relative_to(SELECTED)}")

    print(f"\nTotal outliers copied: {total_outliers}")
    print("Done.")


if __name__ == "__main__":
    main()
