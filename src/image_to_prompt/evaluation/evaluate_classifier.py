"""
VM.AI — Evaluate trained EfficientNet-B4 classifier.

Loads the best model checkpoint, runs on the test set, and produces:
  - Per-class precision / recall / F1 / support
  - Confusion matrix heatmap
  - Top-K accuracy curve
  - Grouped per-class metrics bar chart
  - JSON report

Usage:
  uv run python src/image_to_prompt/evaluation/evaluate_classifier.py
  uv run python src/image_to_prompt/evaluation/evaluate_classifier.py --checkpoint path/to/model.pth
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent.parent / "training"))
from train_classifier import ImageDataset, build_model, val_epoch

DATA_ROOT = Path("data/image_to_prompt/final")
MODEL_DIR = Path("models") / "efficientnet_b4_classifier"
MODEL_PATH = MODEL_DIR / "efficientnet_b4_classifier.pth"
ASSETS = Path("assets/image_classifier")
REPORT_PATH = MODEL_DIR / "evaluation_report.json"
BATCH_SIZE = 32
NUM_CLASSES = 14
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

test_transforms = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def setup_style():
    plt.style.use("dark_background")
    plt.rcParams.update({
        "figure.facecolor": "#0d1117",
        "axes.facecolor": "#0d1117",
        "axes.edgecolor": "#30363d",
        "axes.labelcolor": "#c9d1d9",
        "text.color": "#c9d1d9",
        "xtick.color": "#8b949e",
        "ytick.color": "#8b949e",
        "grid.color": "#21262d",
    })


def get_all_predictions(model, loader, device, num_classes):
    model.eval()
    y_true = []
    y_pred = []
    y_scores = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.autocast(device_type=device.type):
                outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(outputs.argmax(1).cpu().numpy().tolist())
            y_scores.extend(probs.cpu().numpy().tolist())

    return y_true, y_pred, y_scores


def compute_topk_accuracy(y_scores, y_true, max_k=14):
    y_scores = np.array(y_scores)
    y_true = np.array(y_true)
    accuracies = []
    for k in range(1, max_k + 1):
        topk_preds = np.argsort(y_scores, axis=1)[:, -k:]
        correct = sum(y_true[i] in topk_preds[i] for i in range(len(y_true)))
        accuracies.append(correct / len(y_true))
    return accuracies


def plot_confusion_matrix(cm, class_names, save_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, cbar_kws={"label": "Count"},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion matrix saved to {save_path}")


def plot_per_class_metrics(metrics_dict, class_names, save_path):
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(class_names))
    width = 0.25

    prec = [metrics_dict[c]["precision"] for c in class_names]
    rec = [metrics_dict[c]["recall"] for c in class_names]
    f1 = [metrics_dict[c]["f1"] for c in class_names]

    ax.bar(x - width, prec, width, label="Precision")
    ax.bar(x, rec, width, label="Recall")
    ax.bar(x + width, f1, width, label="F1")

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Metrics")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Per-class metrics saved to {save_path}")


def plot_topk_curve(accuracies, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ks = list(range(1, len(accuracies) + 1))
    
    ax.plot(ks, accuracies, marker="o", color="steelblue", linewidth=2)
    ax.fill_between(ks, min(accuracies) - 0.02, accuracies, alpha=0.2, color="steelblue")
    
    for k, acc in enumerate(accuracies):
        ax.annotate(f"{acc:.3f}",
                    xy=(k+1, acc),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    color="#c9d1d9")
    
    y_min = min(accuracies) - 0.02
    ax.set_ylim(y_min, 1.02)
    
    ax.set_xlabel("K")
    ax.set_ylabel("Accuracy")
    ax.set_title("Top-K Accuracy")
    ax.set_xticks(ks)
    ax.axhline(y=accuracies[0], color="#8b949e", linestyle="--", 
               alpha=0.5, label=f"Top-1: {accuracies[0]:.3f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate EfficientNet-B4 classifier")
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint path")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else MODEL_PATH
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return

    setup_style()
    device = torch.device(DEVICE)
    ASSETS.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print()

    # Load model
    model = build_model(NUM_CLASSES).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    print(f"Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")
    print()

    # Data
    test_csv = str(DATA_ROOT / "test.csv")
    test_dataset = ImageDataset(test_csv, transform=test_transforms)
    class_names = test_dataset.classes
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )
    print(f"Test samples: {len(test_dataset)}")
    print(f"Classes ({len(class_names)}): {class_names}")
    print()

    # Predictions
    y_true, y_pred, y_scores = get_all_predictions(model, test_loader, device, NUM_CLASSES)
    topk_accs = compute_topk_accuracy(y_scores, y_true, max_k=14)
    top1 = topk_accs[0]
    print(f"Top-1 accuracy: {top1:.3f}")
    print(f"Top-3 accuracy: {topk_accs[2]:.3f}")
    print()

    # Per-class metrics
    print("-" * 70)
    print(f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>8} {'Support':>8}")
    print("-" * 70)
    per_class = {}
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    for cn in class_names:
        m = report_dict[cn]
        per_class[cn] = {
            "precision": round(m["precision"], 4),
            "recall": round(m["recall"], 4),
            "f1": round(m["f1-score"], 4),
            "support": int(m["support"]),
        }
        print(f"{cn:<20} {per_class[cn]['precision']:>10.4f} {per_class[cn]['recall']:>10.4f} {per_class[cn]['f1']:>8.4f} {per_class[cn]['support']:>8}")

    macro_f1 = round(report_dict["macro avg"]["f1-score"], 4)
    weighted_f1 = round(report_dict["weighted avg"]["f1-score"], 4)
    print("-" * 70)
    print(f"{'Macro avg':<20} {report_dict['macro avg']['precision']:>10.4f} {report_dict['macro avg']['recall']:>10.4f} {macro_f1:>8.4f} {int(report_dict['macro avg']['support']):>8}")
    print(f"{'Weighted avg':<20} {report_dict['weighted avg']['precision']:>10.4f} {report_dict['weighted avg']['recall']:>10.4f} {weighted_f1:>8.4f} {int(report_dict['weighted avg']['support']):>8}")
    print()

    # Charts
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names, ASSETS / "confusion_matrix.png")
    plot_per_class_metrics(per_class, class_names, ASSETS / "per_class_metrics.png")
    plot_topk_curve(topk_accs, ASSETS / "topk_accuracy.png")

    # Report JSON
    report = {
        "checkpoint": str(checkpoint_path),
        "test_samples": len(test_dataset),
        "top1_accuracy": round(top1, 4),
        "topk_accuracy": {f"top{k+1}": round(v, 4) for k, v in enumerate(topk_accs)},
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {REPORT_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()
