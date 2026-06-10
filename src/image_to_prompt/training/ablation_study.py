"""
VM.AI — Ablation study for EfficientNet-B4 classifier.

Runs 4 sequential experiments comparing augmentation vs frozen phase effects.
Generates per-experiment and comparison charts.

Usage:
  uv run python src/image_to_prompt/training/ablation_study.py
  uv run python src/image_to_prompt/training/ablation_study.py --experiment baseline
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent.parent / "evaluation"))
sys.path.insert(0, str(Path(__file__).parent))
from evaluate_classifier import (
    compute_topk_accuracy,
    get_all_predictions,
    plot_confusion_matrix,
    plot_per_class_metrics,
    plot_topk_curve,
    setup_style,
)
from train_classifier import (
    DATA_ROOT,
    IMAGENET_MEAN,
    IMAGENET_STD,
    ImageDataset,
    build_model,
    train_epoch,
    val_epoch,
)

EXPERIMENTS = {
    "baseline": {
        "augmentation": True,
        "frozen_phase": True,
        "epochs_frozen": 5,
        "epochs_unfrozen": 25,
        "description": "Full pipeline: augmentation + frozen phase",
    },
    "no_augmentation": {
        "augmentation": False,
        "frozen_phase": True,
        "epochs_frozen": 5,
        "epochs_unfrozen": 25,
        "description": "No augmentation: only resize + normalize",
    },
    "no_frozen_phase": {
        "augmentation": True,
        "frozen_phase": False,
        "epochs_frozen": 5,
        "epochs_unfrozen": 25,
        "description": "No frozen phase: all layers trained from epoch 1",
    },
    "no_aug_no_frozen": {
        "augmentation": False,
        "frozen_phase": False,
        "epochs_frozen": 5,
        "epochs_unfrozen": 25,
        "description": "Bare minimum: no augmentation + no frozen phase",
    },
}

BASE_CONFIG = {
    "num_classes": 14,
    "batch_size": 32,
    "lr_head": 1e-3,
    "lr_backbone": 1e-5,
    "weight_decay": 1e-4,
    "label_smoothing": 0.1,
}

ABLATION_MODEL_DIR = Path("models/ablation")
ABLATION_ASSETS_DIR = Path("assets/image_classifier/ablation")


def get_transforms(augmentation: bool):
    if augmentation:
        train_tf = transforms.Compose([
            transforms.Resize((420, 420)),
            transforms.RandomCrop((380, 380)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize((380, 380)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    val_tf = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return train_tf, val_tf


def run_experiment(name: str, config: dict, device: torch.device) -> dict:
    print(f"\n{'=' * 60}")
    print(f"Experiment: {name}")
    print(f"Description: {config['description']}")
    print(f"{'=' * 60}\n")

    model_dir = ABLATION_MODEL_DIR / name
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = str(model_dir / "checkpoint.pth")

    train_tf, val_tf = get_transforms(config["augmentation"])

    train_dataset = ImageDataset(str(DATA_ROOT / "train.csv"), transform=train_tf)
    val_dataset = ImageDataset(str(DATA_ROOT / "val.csv"), transform=val_tf)
    test_dataset = ImageDataset(str(DATA_ROOT / "test.csv"), transform=val_tf)

    train_loader = DataLoader(
        train_dataset, batch_size=BASE_CONFIG["batch_size"],
        shuffle=True, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BASE_CONFIG["batch_size"],
        shuffle=False, num_workers=2, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BASE_CONFIG["batch_size"],
        shuffle=False, num_workers=2, pin_memory=True,
    )

    model = build_model(BASE_CONFIG["num_classes"]).to(device)

    # Weighted loss to compensate for smaller classes (cleaning, running)
    class_counts = {
        "basketball": 700,
        "cleaning": 673,
        "computer_work": 700,
        "cooking": 700,
        "cycling": 700,
        "driving": 700,
        "football": 700,
        "gym": 700,
        "office": 700,
        "pet_care": 700,
        "reading": 700,
        "restaurant": 700,
        "running": 624,
        "shopping": 700,
    }
    counts = [class_counts[c] for c in sorted(class_counts.keys())]
    total = sum(counts)
    class_weights = torch.tensor(
        [total / (len(counts) * c) for c in counts],
        dtype=torch.float,
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=BASE_CONFIG["label_smoothing"],
    )
    scaler = GradScaler()
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0

    # ── Phase A: Frozen backbone ──
    if config["frozen_phase"] and config["epochs_frozen"] > 0:
        print("Phase A: Frozen backbone")
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True

        optimizer_A = torch.optim.AdamW(
            model.classifier.parameters(),
            lr=BASE_CONFIG["lr_head"],
            weight_decay=BASE_CONFIG["weight_decay"],
        )

        for epoch in range(config["epochs_frozen"]):
            t0 = time.time()
            tl, ta = train_epoch(model, train_loader, optimizer_A, criterion, device, scaler)
            vl, va = val_epoch(model, val_loader, criterion, device)
            elapsed = time.time() - t0
            history["train_loss"].append(tl)
            history["val_loss"].append(vl)
            history["train_acc"].append(ta)
            history["val_acc"].append(va)
            print(f"  [A {epoch+1}/{config['epochs_frozen']}] "
                  f"train_acc={ta:.3f} val_acc={va:.3f} ({elapsed:.0f}s)")
            if va > best_val_acc:
                best_val_acc = va
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "best_val_acc": best_val_acc,
                    "history": history,
                }, checkpoint_path)

    # ── Phase B: Partial unfreeze ──
    print("Phase B: Partial unfreeze")

    if not config["frozen_phase"]:
        for param in model.parameters():
            param.requires_grad = True
        optimizer_B = torch.optim.AdamW([
            {"params": model.classifier.parameters(), "lr": BASE_CONFIG["lr_head"] / 10},
            {"params": model.blocks.parameters(), "lr": BASE_CONFIG["lr_backbone"]},
        ], weight_decay=BASE_CONFIG["weight_decay"])
    else:
        for param in model.blocks[-2:].parameters():
            param.requires_grad = True
        optimizer_B = torch.optim.AdamW([
            {"params": model.classifier.parameters(), "lr": BASE_CONFIG["lr_head"] / 10},
            {"params": model.blocks[-2:].parameters(), "lr": BASE_CONFIG["lr_backbone"]},
        ], weight_decay=BASE_CONFIG["weight_decay"])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_B, T_max=config["epochs_unfrozen"], eta_min=1e-6,
    )

    stopped_epoch = config["epochs_frozen"] + config["epochs_unfrozen"]
    for i in range(config["epochs_unfrozen"]):
        t0 = time.time()
        tl, ta = train_epoch(model, train_loader, optimizer_B, criterion, device, scaler)
        vl, va = val_epoch(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["train_acc"].append(ta)
        history["val_acc"].append(va)
        epoch_num = config["epochs_frozen"] + i + 1
        total = config["epochs_frozen"] + config["epochs_unfrozen"]
        print(f"  [B {epoch_num}/{total}] "
              f"train_acc={ta:.3f} val_acc={va:.3f} ({elapsed:.0f}s)")
        if va > best_val_acc:
            best_val_acc = va
            torch.save({
                "model_state_dict": model.state_dict(),
                "best_val_acc": best_val_acc,
                "history": history,
            }, checkpoint_path)
            print(f"  * Best saved (val_acc={va:.3f})")

    # ── Test evaluation ──
    best_ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_loss, test_acc = val_epoch(model, test_loader, criterion, device)
    history["test_loss"] = test_loss
    history["test_acc"] = test_acc
    history["stopped_epoch"] = stopped_epoch
    print(f"\n  Test accuracy: {test_acc:.3f}")

    with open(model_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    return {
        "name": name,
        "description": config["description"],
        "best_val_acc": round(best_val_acc, 4),
        "test_acc": round(test_acc, 4),
        "test_loss": round(test_loss, 4),
        "stopped_epoch": stopped_epoch,
        "history": history,
        "checkpoint_path": checkpoint_path,
        "class_names": train_dataset.classes,
        "test_loader": test_loader,
        "model": model,
        "device": device,
    }


def plot_training_curves(history: dict, experiment_name: str, save_path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="Train loss", color="steelblue")
    ax1.plot(epochs, history["val_loss"], label="Val loss", color="orange")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"Loss — {experiment_name}")
    ax1.legend()

    ax2.plot(epochs, history["train_acc"], label="Train acc", color="steelblue")
    ax2.plot(epochs, history["val_acc"], label="Val acc", color="orange")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"Accuracy — {experiment_name}")
    ax2.legend()
    ax2.set_ylim(min(history["val_acc"]) - 0.05, 1.02)

    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Training curves saved to {save_path}")


def evaluate_experiment(result: dict):
    name = result["name"]
    assets_dir = ABLATION_ASSETS_DIR / name
    assets_dir.mkdir(parents=True, exist_ok=True)

    model = result["model"]
    device = result["device"]
    test_loader = result["test_loader"]
    class_names = result["class_names"]
    history = result["history"]

    y_true, y_pred, y_scores = get_all_predictions(model, test_loader, device, BASE_CONFIG["num_classes"])

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names, assets_dir / "confusion_matrix.png")

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    per_class = {
        c: {
            "precision": report[c]["precision"],
            "recall": report[c]["recall"],
            "f1": report[c]["f1-score"],
        }
        for c in class_names
    }
    plot_per_class_metrics(per_class, class_names, assets_dir / "per_class_metrics.png")

    topk_accs = compute_topk_accuracy(y_scores, y_true, max_k=14)
    plot_topk_curve(topk_accs, assets_dir / "topk_accuracy.png")

    plot_training_curves(history, name, assets_dir / "training_curves.png")

    return topk_accs, per_class


def plot_comparison_topk(all_results: list[dict], save_path: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["steelblue", "orange", "green", "red"]

    for result, color in zip(all_results, colors):
        topk = result["topk_accs"]
        ks = list(range(1, len(topk) + 1))
        ax.plot(ks, topk, marker="o", label=result["name"], color=color, linewidth=2)

    ax.set_xlabel("K")
    ax.set_ylabel("Accuracy")
    ax.set_title("Top-K Accuracy Comparison")
    ax.set_xticks(ks)
    y_min = min(r["topk_accs"][0] for r in all_results) - 0.02
    ax.set_ylim(y_min, 1.02)
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_comparison_per_class_f1(all_results: list[dict], class_names: list, save_path: Path):
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(class_names))
    n = len(all_results)
    width = 0.8 / n
    colors = ["steelblue", "orange", "green", "red"]

    for idx, (result, color) in enumerate(zip(all_results, colors)):
        f1_scores = [result["per_class"][c]["f1"] for c in class_names]
        offset = (idx - n / 2 + 0.5) * width
        ax.bar(x + offset, f1_scores, width, label=result["name"], color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1 Comparison")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_comparison_training_curves(all_results: list[dict], save_path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["steelblue", "orange", "green", "red"]

    for result, color in zip(all_results, colors):
        h = result["history"]
        epochs = range(1, len(h["val_loss"]) + 1)
        ax1.plot(epochs, h["val_loss"], label=result["name"], color=color, linewidth=2)
        ax2.plot(epochs, h["val_acc"], label=result["name"], color=color, linewidth=2)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Val Loss Comparison")
    ax1.legend()

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Val Accuracy Comparison")
    ax2.legend()
    y_min = min(min(r["history"]["val_acc"]) for r in all_results) - 0.05
    ax2.set_ylim(y_min, 1.02)

    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_summary(all_results: list[dict]):
    summary = []
    for r in all_results:
        summary.append({
            "experiment": r["name"],
            "description": r["description"],
            "best_val_acc": r["best_val_acc"],
            "test_acc": r["test_acc"],
            "test_loss": r["test_loss"],
            "stopped_epoch": r["stopped_epoch"],
        })

    path = ABLATION_MODEL_DIR / "ablation_results.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("ABLATION STUDY RESULTS")
    print("=" * 70)
    print(f"{'Experiment':<22} {'Val Acc':>10} {'Test Acc':>10} {'Stopped':>10}")
    print("-" * 70)
    for r in summary:
        print(f"{r['experiment']:<22} {r['best_val_acc']:>10.4f} "
              f"{r['test_acc']:>10.4f} {r['stopped_epoch']:>10}")
    print("=" * 70)
    print(f"\nResults saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Ablation study for EfficientNet-B4")
    parser.add_argument(
        "--experiment", default=None,
        choices=["baseline", "no_augmentation", "no_frozen_phase", "no_aug_no_frozen"],
        help="Run specific experiment only. Default: run all.",
    )
    args = parser.parse_args()

    setup_style()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ABLATION_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ABLATION_ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    if args.experiment:
        experiments_to_run = {args.experiment: EXPERIMENTS[args.experiment]}
    else:
        experiments_to_run = EXPERIMENTS

    all_results = []

    for name, config in experiments_to_run.items():
        result = run_experiment(name, config, device)
        topk_accs, per_class = evaluate_experiment(result)
        result["topk_accs"] = topk_accs
        result["per_class"] = per_class
        all_results.append(result)

    if len(all_results) == len(EXPERIMENTS):
        print("\nGenerating comparison charts...")
        class_names = all_results[0]["class_names"]
        plot_comparison_topk(all_results, ABLATION_ASSETS_DIR / "comparison_topk.png")
        plot_comparison_per_class_f1(all_results, class_names, ABLATION_ASSETS_DIR / "comparison_per_class_f1.png")
        plot_comparison_training_curves(all_results, ABLATION_ASSETS_DIR / "comparison_training_curves.png")

    save_summary(all_results)
    print("\nAblation study complete.")


if __name__ == "__main__":
    main()
