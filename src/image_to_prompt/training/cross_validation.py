"""
VM.AI — 5-fold stratified cross-validation with locked test set for EfficientNet-B4.

Uses train+val for 5-fold CV. Each fold trains on 4/5, validates on 1/5 (early stopping).
Each fold evaluates its best model on the locked test set.
All charts and metrics are aggregated across all 5 folds.

Usage:
  uv run python src/image_to_prompt/training/cross_validation.py
  uv run python src/image_to_prompt/training/cross_validation.py --fold 1
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
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent.parent / "evaluation"))
sys.path.insert(0, str(Path(__file__).parent))
from evaluate_classifier import (
    compute_topk_accuracy,
    get_all_predictions,
    plot_confusion_matrix,
    setup_style,
)
from train_classifier import (
    DATA_ROOT,
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_model,
    train_epoch,
    val_epoch,
)

CV_CONFIG = {
    "n_splits": 5,
    "seed": 42,
    "num_classes": 14,
    "batch_size": 32,
    "epochs_frozen": 5,
    "epochs_unfrozen": 25,
    "lr_head": 1e-3,
    "lr_backbone": 1e-5,
    "weight_decay": 1e-4,
    "label_smoothing": 0.1,
    "early_stopping_patience": 7,
    "early_stopping_min_delta": 0.001,
}

CV_MODEL_DIR = Path("models/cross_validation")
CV_ASSETS_DIR = Path("assets/image_classifier/cross_validation")

train_transforms = transforms.Compose([
    transforms.Resize((420, 420)),
    transforms.RandomCrop((380, 380)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_transforms = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def load_train_val_dataset() -> pd.DataFrame:
    """Load train + val only. Test set locked for final evaluation."""
    dfs = []
    for split in ["train", "val"]:
        df = pd.read_csv(DATA_ROOT / f"{split}.csv", quoting=1)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def load_test_dataset() -> pd.DataFrame:
    """Load test set. Used only for evaluation, never for training."""
    return pd.read_csv(DATA_ROOT / "test.csv", quoting=1)


class FullImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.classes = sorted(df["label"].unique())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.labels = [self.class_to_idx[l] for l in df["label"]]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(DATA_ROOT / row["path"]).convert("RGB")
        label = self.class_to_idx[row["label"]]
        if self.transform:
            img = self.transform(img)
        return img, label


class EarlyStopping:
    def __init__(self, patience: int = 7, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_acc = 0.0
        self.should_stop = False

    def step(self, val_acc: float) -> bool:
        if val_acc > self.best_acc + self.min_delta:
            self.best_acc = val_acc
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def compute_class_weights(dataset: FullImageDataset, device: torch.device) -> torch.Tensor:
    counts = np.bincount(dataset.labels, minlength=CV_CONFIG["num_classes"])
    counts = counts.clip(min=1)
    total = counts.sum()
    weights = total / (CV_CONFIG["num_classes"] * counts.astype(float))
    return torch.tensor(weights, dtype=torch.float).to(device)


def train_fold(fold: int, train_idx: list, val_idx: list, dataset: FullImageDataset, device: torch.device) -> dict:
    print(f"\n{'=' * 60}")
    print(f"Fold {fold}/{CV_CONFIG['n_splits']}")
    print(f"  Train: {len(train_idx)} | Val (early stop): {len(val_idx)}")
    print(f"{'=' * 60}\n")

    fold_dir = CV_MODEL_DIR / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = str(fold_dir / "checkpoint.pth")

    train_dataset = FullImageDataset(dataset.df.iloc[train_idx], transform=train_transforms)
    val_dataset = FullImageDataset(dataset.df.iloc[val_idx], transform=val_transforms)

    train_loader = DataLoader(
        train_dataset, batch_size=CV_CONFIG["batch_size"],
        shuffle=True, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CV_CONFIG["batch_size"],
        shuffle=False, num_workers=2, pin_memory=True,
    )

    class_weights = compute_class_weights(train_dataset, device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=CV_CONFIG["label_smoothing"],
    )

    model = build_model(CV_CONFIG["num_classes"]).to(device)
    scaler = GradScaler()
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    early_stopping = EarlyStopping(
        patience=CV_CONFIG["early_stopping_patience"],
        min_delta=CV_CONFIG["early_stopping_min_delta"],
    )

    # ── Phase A: Frozen backbone ──
    print("Phase A: Frozen backbone")
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    optimizer_A = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=CV_CONFIG["lr_head"],
        weight_decay=CV_CONFIG["weight_decay"],
    )

    for epoch in range(CV_CONFIG["epochs_frozen"]):
        t0 = time.time()
        tl, ta = train_epoch(model, train_loader, optimizer_A, criterion, device, scaler)
        vl, va = val_epoch(model, val_loader, criterion, device)
        elapsed = time.time() - t0
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["train_acc"].append(ta)
        history["val_acc"].append(va)
        print(f"  [A {epoch+1}/{CV_CONFIG['epochs_frozen']}] "
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
    for param in model.blocks[-2:].parameters():
        param.requires_grad = True

    optimizer_B = torch.optim.AdamW([
        {"params": model.classifier.parameters(), "lr": CV_CONFIG["lr_head"] / 10},
        {"params": model.blocks[-2:].parameters(), "lr": CV_CONFIG["lr_backbone"]},
    ], weight_decay=CV_CONFIG["weight_decay"])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_B, T_max=CV_CONFIG["epochs_unfrozen"], eta_min=1e-6,
    )

    stopped_epoch = CV_CONFIG["epochs_unfrozen"]
    for i in range(CV_CONFIG["epochs_unfrozen"]):
        t0 = time.time()
        tl, ta = train_epoch(model, train_loader, optimizer_B, criterion, device, scaler)
        vl, va = val_epoch(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["train_acc"].append(ta)
        history["val_acc"].append(va)
        epoch_num = CV_CONFIG["epochs_frozen"] + i + 1
        total = CV_CONFIG["epochs_frozen"] + CV_CONFIG["epochs_unfrozen"]
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
        if early_stopping.step(va):
            print(f"  Early stopping at epoch {epoch_num}")
            stopped_epoch = epoch_num
            break

    history["stopped_epoch"] = stopped_epoch
    with open(fold_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    return {
        "fold": fold,
        "best_val_acc": best_val_acc,
        "stopped_epoch": stopped_epoch,
        "history": history,
        "checkpoint_path": checkpoint_path,
        "model": model,
        "device": device,
    }


def evaluate_fold_on_test(result: dict, test_loader: DataLoader, class_names: list) -> tuple:
    """Evaluate fold's best model on locked test set."""
    model = result["model"]
    device = result["device"]

    ckpt = torch.load(result["checkpoint_path"], map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    y_true, y_pred, y_scores = get_all_predictions(model, test_loader, device, len(class_names))

    top1 = compute_topk_accuracy(y_scores, y_true, max_k=1)[0]
    print(f"  Fold {result['fold']} test_acc={top1:.4f}")

    return y_true, y_pred, y_scores, top1


def plot_cv_boxplot(fold_accs: list, save_path: Path):
    """Test accuracy per fold as a bar chart with mean ± std band."""
    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)
    n_folds = len(fold_accs)

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(1, n_folds + 1)
    colors = ["steelblue"] * n_folds
    best_idx = int(np.argmax(fold_accs))
    worst_idx = int(np.argmin(fold_accs))
    colors[best_idx] = "#2ea043"
    colors[worst_idx] = "#f85149"

    bars = ax.bar(x, fold_accs, color=colors, alpha=0.85, width=0.5, zorder=3)

    for i, (bar, acc) in enumerate(zip(bars, fold_accs)):
        label = f"{acc:.4f}"
        if i == best_idx:
            label += " ▲"
        elif i == worst_idx:
            label += " ▼"
        ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.0002, label,
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.axhline(mean_acc, color="orange", linestyle="--", linewidth=1.5,
               label=f"Mean: {mean_acc:.4f} ± {std_acc:.4f}", zorder=4)
    ax.axhspan(mean_acc - std_acc, mean_acc + std_acc,
               alpha=0.15, color="orange", label="±1 std", zorder=2)

    y_margin = max((max(fold_accs) - min(fold_accs)) * 3, 0.005)
    ax.set_ylim(min(fold_accs) - y_margin, max(fold_accs) + y_margin * 2)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=1)
    ax.set_axisbelow(True)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {i}" for i in x], fontsize=11)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title("Cross-Validation Test Accuracy per Fold", fontsize=14, pad=15)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#2ea043", alpha=0.85, label="Best fold"),
        Patch(facecolor="#f85149", alpha=0.85, label="Worst fold"),
        Patch(facecolor="steelblue", alpha=0.85, label="Other folds"),
        plt.Line2D([0], [0], color="orange", linestyle="--",
                    label=f"Mean: {mean_acc:.4f} ± {std_acc:.4f}"),
        plt.Rectangle((0, 0), 1, 1, fc="orange", alpha=0.15, label="±1 std"),
    ], fontsize=9, loc="lower right")

    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  CV boxplot saved to {save_path}")


def plot_aggregated_confusion_matrix(all_y_true: list, all_y_pred: list, class_names: list, save_path: Path):
    """Combined confusion matrix from all 5 folds × test set."""
    cm = confusion_matrix(all_y_true, all_y_pred)
    plot_confusion_matrix(cm, class_names, save_path)
    print(f"  Aggregated confusion matrix saved to {save_path}")


def plot_aggregated_per_class_metrics(all_per_class: list, class_names: list, save_path: Path):
    """Mean F1 per class with std error bars across all folds."""
    mean_f1, std_f1 = [], []
    for c in class_names:
        f1s = [fold[c]["f1"] for fold in all_per_class]
        mean_f1.append(np.mean(f1s))
        std_f1.append(np.std(f1s))

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(class_names))
    ax.bar(x, mean_f1, yerr=std_f1, capsize=5, color="steelblue", alpha=0.85, error_kw={"color": "white"})
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylabel("F1 Score")
    ax.set_title("Mean Per-Class F1 ± Std — Aggregated 5-Fold CV (Test Set)")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Aggregated per-class F1 saved to {save_path}")


def save_cv_results(fold_results: list, fold_accs: list, all_per_class: list, class_names: list):
    mean_acc = float(np.mean(fold_accs))
    std_acc = float(np.std(fold_accs))

    summary = {
        "n_splits": CV_CONFIG["n_splits"],
        "evaluation": "test set (locked)",
        "mean_test_accuracy": round(mean_acc, 4),
        "std_test_accuracy": round(std_acc, 4),
        "per_fold": [
            {
                "fold": r["fold"],
                "test_acc": round(fold_accs[r["fold"] - 1], 4),
                "best_val_acc": round(r["best_val_acc"], 4),
                "stopped_epoch": r["stopped_epoch"],
            }
            for r in fold_results
        ],
        "per_class_mean_f1": {
            c: round(float(np.mean([f[c]["f1"] for f in all_per_class])), 4)
            for c in class_names
        },
    }

    path = CV_MODEL_DIR / "cv_results.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 55)
    print("CROSS-VALIDATION RESULTS (Test Set)")
    print("=" * 55)
    for r in fold_results:
        print(f"  Fold {r['fold']}: test_acc={fold_accs[r['fold']-1]:.4f} "
              f"val_acc={r['best_val_acc']:.4f} "
              f"(stopped epoch {r['stopped_epoch']})")
    print("-" * 55)
    print(f"  Mean test acc: {mean_acc:.4f} ± {std_acc:.4f}")
    print("=" * 55)
    print(f"\nResults saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="5-fold CV with fixed test set")
    parser.add_argument("--fold", type=int, default=None, help="Run specific fold only (1-5). Default: run all.")
    args = parser.parse_args()

    setup_style()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    CV_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    CV_ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    # Load datasets
    train_val_df = load_train_val_dataset()
    test_df = load_test_dataset()

    dataset = FullImageDataset(train_val_df, transform=None)
    class_names = dataset.classes
    labels = np.array(dataset.labels)

    # Build test loader once — reused for every fold
    test_dataset = FullImageDataset(test_df, transform=val_transforms)
    test_loader = DataLoader(
        test_dataset, batch_size=CV_CONFIG["batch_size"],
        shuffle=False, num_workers=2, pin_memory=True,
    )

    print(f"Train+val: {len(dataset)} images")
    print(f"Test:      {len(test_dataset)} images (locked)")
    print(f"Classes:   {class_names}")

    # Stratified K-Fold on train+val
    skf = StratifiedKFold(
        n_splits=CV_CONFIG["n_splits"],
        shuffle=True,
        random_state=CV_CONFIG["seed"],
    )
    folds = list(skf.split(np.zeros(len(labels)), labels))

    if args.fold:
        folds_to_run = [(args.fold - 1, folds[args.fold - 1])]
    else:
        folds_to_run = list(enumerate(folds))

    fold_results = []
    fold_accs = []
    all_y_true = []
    all_y_pred = []
    all_per_class = []

    for fold_idx, (train_idx, val_idx) in folds_to_run:
        fold_num = fold_idx + 1

        # Train fold (val used for early stopping only)
        result = train_fold(fold_num, train_idx.tolist(), val_idx.tolist(), dataset, device)

        # Evaluate on locked test set
        print(f"\nEvaluating fold {fold_num} on test set...")
        y_true, y_pred, y_scores, top1 = evaluate_fold_on_test(result, test_loader, class_names)

        # Collect aggregated predictions
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)

        # Per-fold per-class metrics
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
        per_class = {
            c: {
                "precision": report[c]["precision"],
                "recall": report[c]["recall"],
                "f1": report[c]["f1-score"],
            }
            for c in class_names
        }
        all_per_class.append(per_class)
        fold_accs.append(top1)
        fold_results.append(result)

    # Aggregated charts (only if all folds ran)
    if len(fold_results) == CV_CONFIG["n_splits"]:
        print("\nGenerating aggregated charts...")

        plot_cv_boxplot(fold_accs, CV_ASSETS_DIR / "cv_accuracy_boxplot.png")
        plot_aggregated_confusion_matrix(
            all_y_true, all_y_pred, class_names,
            CV_ASSETS_DIR / "aggregated_confusion_matrix.png",
        )
        plot_aggregated_per_class_metrics(
            all_per_class, class_names,
            CV_ASSETS_DIR / "aggregated_per_class_metrics.png",
        )

    save_cv_results(fold_results, fold_accs, all_per_class, class_names)
    print("\nCross-validation complete.")


if __name__ == "__main__":
    main()
