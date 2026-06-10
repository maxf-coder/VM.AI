"""
VM.AI -- Dataset Analysis for Image Classifier

Runs 4 checks on the final/ dataset:
1. Class balance       -> bar chart
2. Duplicate detection -> perceptual hashing
3. Outlier detection   -> ResNet18 + IsolationForest
4. Color/brightness    -> per-category avg
Outputs: analysis_report.json + PNG charts
"""

import json
import shutil
import time
from collections import defaultdict
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import torch
import torchvision
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights

FINAL = Path("data/image_to_prompt/final")
ASSETS = Path("assets/image_classifier")
REPORT_PATH = Path("data/image_to_prompt/analysis_report.json")
OUTLIERS = Path("data/image_to_prompt/outliers")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def setup_style():
    plt.style.use("dark_background")
    plt.rcParams.update(
        {
            "figure.facecolor": "#0d1117",
            "axes.facecolor": "#0d1117",
            "axes.edgecolor": "#30363d",
            "axes.labelcolor": "#c9d1d9",
            "text.color": "#c9d1d9",
            "xtick.color": "#8b949e",
            "ytick.color": "#8b949e",
            "grid.color": "#21262d",
        }
    )


def _is_image(f: Path) -> bool:
    return f.suffix.lower() in IMAGE_EXTENSIONS


def collect_all_images() -> dict[str, list[Path]]:
    images_by_cat: dict[str, list[Path]] = defaultdict(list)
    for split_dir in ["train", "val", "test"]:
        split_path = FINAL / split_dir
        if not split_path.is_dir():
            continue
        for cat_dir in sorted(split_path.iterdir()):
            if not cat_dir.is_dir():
                continue
            for f in cat_dir.rglob("*"):
                if f.is_file() and _is_image(f):
                    images_by_cat[cat_dir.name].append(f)
    return dict(images_by_cat)


def analyze_class_balance(images_by_cat: dict[str, list[Path]]) -> dict:
    print("=" * 60)
    print("1. Class Balance Check")
    print("=" * 60)

    counts = {cat: len(paths) for cat, paths in sorted(images_by_cat.items())}
    total = sum(counts.values())

    for cat, count in sorted(counts.items()):
        flag = ""
        if count < 700:
            flag = "  !! LOW"
        elif count > 1200:
            flag = "  !! HIGH"
        print(f"  {cat}: {count}{flag}")

    min_cat = min(counts, key=counts.get)
    max_cat = max(counts, key=counts.get)
    balance_ratio = counts[min_cat] / counts[max_cat]
    print(f"\n  Min: {min_cat} ({counts[min_cat]})")
    print(f"  Max: {max_cat} ({counts[max_cat]})")
    print(f"  Balance ratio: {balance_ratio:.2f}  (target >= 0.70)")
    if balance_ratio < 0.70:
        print("  !! Below 0.70 -- consider rebalancing")

    ASSETS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    cats_sorted = sorted(counts.keys())
    values = [counts[c] for c in cats_sorted]
    bars = ax.bar(cats_sorted, values, color="steelblue")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                str(v), ha="center", fontsize=8)
    ax.set_xticks(range(len(cats_sorted)))
    ax.set_xticklabels(cats_sorted, rotation=45, ha="right")
    ax.set_ylabel("Images")
    ax.set_title("Images per category (all splits)")
    fig.tight_layout()
    fig.savefig(str(ASSETS / "class_balance.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart saved to {ASSETS / 'class_balance.png'}")
    print()

    return {
        "total_images": total,
        "per_category": counts,
        "min_category": min_cat,
        "min_count": counts[min_cat],
        "max_category": max_cat,
        "max_count": counts[max_cat],
        "balance_ratio": round(balance_ratio, 2),
    }


def analyze_duplicates(images_by_cat: dict[str, list[Path]]) -> dict:
    print("=" * 60)
    print("2. Duplicate Detection (perceptual hash)")
    print("=" * 60)

    hash_map: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    total = sum(len(v) for v in images_by_cat.values())
    processed = 0

    for cat, paths in sorted(images_by_cat.items()):
        for p in paths:
            try:
                h = str(imagehash.phash(Image.open(p)))
                hash_map[h].append((cat, p))
            except Exception:
                pass
            processed += 1
            if processed % 2000 == 0:
                print(f"    Hashed {processed}/{total}")

    cross_cat = 0
    within_cat = 0
    duplicate_groups = []
    per_cat_within: dict[str, int] = defaultdict(int)

    for h, group in hash_map.items():
        if len(group) <= 1:
            continue
        cats_in_group = {c for c, _ in group}
        if len(cats_in_group) > 1:
            cross_cat += len(group) - 1
            duplicate_groups.append(group)
        else:
            cat = next(iter(cats_in_group))
            per_cat_within[cat] += len(group) - 1
            within_cat += len(group) - 1

    print(f"  Cross-category duplicates: {cross_cat}")
    print(f"  Within-category duplicates: {within_cat}")
    if per_cat_within:
        print("\n  Within-category duplicates per category:")
        for cat in sorted(per_cat_within):
            print(f"    {cat}: {per_cat_within[cat]}")
    if duplicate_groups:
        print("\n  Cross-category groups (first 10):")
        for group in duplicate_groups[:10]:
            entries = [f"{cat}/{p.name}" for cat, p in group]
            print(f"    {'  vs  '.join(entries)}")

    print()
    return {
        "cross_category_duplicates": cross_cat,
        "within_category_duplicates": within_cat,
        "duplicates_per_category": dict(per_cat_within),
    }


def analyze_outliers(images_by_cat: dict[str, list[Path]]) -> dict:
    print("=" * 60)
    print("3. Outlier Detection (ResNet18 + IsolationForest)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Using device: {device}")

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.eval()
    model.to(device)

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    total_outliers = 0
    per_cat_outliers: dict[str, int] = {}

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
        per_cat_outliers[cat] = len(outlier_paths)
        total_outliers += len(outlier_paths)

        OUTLIERS.mkdir(parents=True, exist_ok=True)
        for op in outlier_paths:
            try:
                split = op.relative_to(FINAL).parts[0]
                new_name = f"{split}_{cat}_{op.name}"
                shutil.copy2(op, OUTLIERS / new_name)
            except Exception:
                pass

        elapsed = time.time() - t0
        print(f"  {cat}: {len(outlier_paths)} outliers flagged ({elapsed:.0f}s)")
        for op in outlier_paths[:3]:
            print(f"    {op.relative_to(FINAL)}")

    print(f"\n  Total outliers flagged: {total_outliers}")
    print()
    return {"total_outliers": total_outliers, "outliers_per_category": per_cat_outliers}


def analyze_brightness(images_by_cat: dict[str, list[Path]]) -> dict:
    print("=" * 60)
    print("4. Color / Brightness Distribution")
    print("=" * 60)

    per_cat_brightness: dict[str, float] = {}
    overall_brightness = []

    for cat, paths in sorted(images_by_cat.items()):
        brightnesses = []
        for p in paths:
            try:
                img = Image.open(p).convert("L")
                brightnesses.append(np.mean(np.array(img)))
            except Exception:
                continue
        if not brightnesses:
            continue
        avg = np.mean(brightnesses)
        per_cat_brightness[cat] = round(avg, 1)
        overall_brightness.extend(brightnesses)
        print(f"  {cat}: avg brightness = {avg:.1f}")

    global_avg = np.mean(overall_brightness) if overall_brightness else 0
    print(f"\n  Global average brightness: {global_avg:.1f}")

    deviant = [(cat, avg) for cat, avg in per_cat_brightness.items()
               if abs(avg - global_avg) > 20]
    if deviant:
        print("  Categories deviating >20 from global avg:")
        for cat, avg in deviant:
            print(f"    {cat}: {avg} (diff {avg - global_avg:+.1f})")
    print()

    return {
        "global_avg_brightness": round(global_avg, 1),
        "per_category_brightness": per_cat_brightness,
    }


def main():
    t_start = time.time()

    if not FINAL.is_dir():
        print("final/ not found. Run split_dataset.py first.")
        return

    setup_style()
    images_by_cat = collect_all_images()
    total_images = sum(len(v) for v in images_by_cat.values())
    print(f"Collected {total_images} images across {len(images_by_cat)} categories\n")

    balance_data = analyze_class_balance(images_by_cat)
    dup_data = analyze_duplicates(images_by_cat)
    outlier_data = analyze_outliers(images_by_cat)
    brightness_data = analyze_brightness(images_by_cat)

    report = {
        "total_images": balance_data["total_images"],
        "categories_analyzed": len(images_by_cat),
        "images_per_category": balance_data["per_category"],
        "min_category": balance_data["min_category"],
        "min_count": balance_data["min_count"],
        "max_category": balance_data["max_category"],
        "max_count": balance_data["max_count"],
        "balance_ratio": balance_data["balance_ratio"],
        "cross_category_duplicates": dup_data["cross_category_duplicates"],
        "within_category_duplicates": dup_data["within_category_duplicates"],
        "duplicates_per_category": dup_data["duplicates_per_category"],
        "total_outliers": outlier_data["total_outliers"],
        "outliers_per_category": outlier_data["outliers_per_category"],
        "global_avg_brightness": brightness_data["global_avg_brightness"],
        "per_category_brightness": brightness_data["per_category_brightness"],
        "elapsed_seconds": round(time.time() - t_start, 1),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report written to {REPORT_PATH}")
    print(f"Total time: {report['elapsed_seconds']:.0f}s")
    print("Done.")


if __name__ == "__main__":
    main()
