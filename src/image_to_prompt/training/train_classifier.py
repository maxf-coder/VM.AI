"""
VM.AI — Train EfficientNet-B4 image classifier locally.

Trains on data/image_to_prompt/final/ using two-phase strategy:
  Phase A: frozen backbone, train head only (5 epochs)
  Phase B: partial unfreeze + cosine annealing (20 epochs)

Usage:
  uv run python src/image_to_prompt/training/train_classifier.py
  uv run python src/image_to_prompt/training/train_classifier.py --epochs_frozen 3 --epochs_unfrozen 15
   uv run python src/image_to_prompt/training/train_classifier.py --resume models/efficientnet_b4_classifier/checkpoint_epoch_10.pth
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

DATA_ROOT = Path("data/image_to_prompt/final")
SAVE_DIR = Path("models") / "efficientnet_b4_classifier"
DEFAULT_SAVE_PATH = str(SAVE_DIR / "efficientnet_b4_classifier.pth")

CONFIG = {
    "num_classes": 14,
    "image_size": 380,
    "batch_size": 32,
    "epochs_frozen": 5,
    "epochs_unfrozen": 25,
    "lr_head": 1e-3,
    "lr_backbone": 1e-5,
    "weight_decay": 1e-4,
    "label_smoothing": 0.1,
    "early_stopping_patience": 7,
    "early_stopping_min_delta": 0.001,
    "data_root": str(DATA_ROOT),
    "save_path": DEFAULT_SAVE_PATH,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

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


class ImageDataset(Dataset):
    def __init__(self, csv_path: str, transform=None):
        self.df = pd.read_csv(csv_path, quoting=1)
        self.transform = transform
        self.classes = sorted(self.df["label"].unique())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = DATA_ROOT / row["path"]
        img = Image.open(img_path).convert("RGB")
        label = self.class_to_idx[row["label"]]
        if self.transform:
            img = self.transform(img)
        return img, label


def build_model(num_classes: int) -> nn.Module:
    model = timm.create_model(
        "efficientnet_b4",
        pretrained=True,
        num_classes=num_classes,
    )
    return model


def train_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        with autocast():
            outputs = model(imgs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), correct / total


def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            with autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(loader), correct / total


class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.001):
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
            print(f"  Early stopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def main():
    parser = argparse.ArgumentParser(description="Train EfficientNet-B4 classifier")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint path")
    parser.add_argument("--epochs_frozen", type=int, default=None, help="Override frozen epochs")
    parser.add_argument("--epochs_unfrozen", type=int, default=None, help="Override unfrozen epochs")
    args = parser.parse_args()

    if args.epochs_frozen is not None:
        CONFIG["epochs_frozen"] = args.epochs_frozen
    if args.epochs_unfrozen is not None:
        CONFIG["epochs_unfrozen"] = args.epochs_unfrozen

    device = torch.device(CONFIG["device"])
    if device.type == "cpu":
        print("WARNING: Training on CPU — expected ~3 hours. Use a GPU for faster training.")

    if not DATA_ROOT.is_dir():
        print(f"ERROR: {DATA_ROOT} not found. Run split_dataset.py first.")
        return

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Data:   {DATA_ROOT}")
    print(f"Config: {json.dumps({k: v for k, v in CONFIG.items() if k != 'device'}, indent=2)}")
    print()

    train_csv = str(DATA_ROOT / "train.csv")
    val_csv = str(DATA_ROOT / "val.csv")
    test_csv = str(DATA_ROOT / "test.csv")

    train_dataset = ImageDataset(train_csv, transform=train_transforms)
    val_dataset = ImageDataset(val_csv, transform=val_transforms)
    test_dataset = ImageDataset(test_csv, transform=val_transforms)

    num_workers = 2
    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG["batch_size"], shuffle=True,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG["batch_size"], shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_dataset, batch_size=CONFIG["batch_size"], shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    print(f"Classes: {train_dataset.classes}")
    print()

    model = build_model(CONFIG["num_classes"]).to(device)
    start_epoch = 0
    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    # Create Phase B optimizer + scheduler early so resume can restore their state
    optimizer_B = torch.optim.AdamW([
        {"params": model.classifier.parameters(), "lr": CONFIG["lr_head"] / 10},
        {"params": model.blocks[-2:].parameters(), "lr": CONFIG["lr_backbone"]},
    ], weight_decay=CONFIG["weight_decay"])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_B,
        T_max=CONFIG["epochs_unfrozen"],
        eta_min=1e-6,
    )

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint.get("epoch", 0)
        best_val_acc = checkpoint.get("best_val_acc", 0.0)
        history = checkpoint.get("history", history)
        if "scheduler_state_dict" in checkpoint:
            optimizer_B.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print(f"Resumed from epoch {start_epoch} (best val_acc={best_val_acc:.3f})")

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
        label_smoothing=CONFIG["label_smoothing"],
    )
    scaler = GradScaler()

    # ── Phase A: Frozen backbone ──
    print("=" * 60)
    print("Phase A: Frozen backbone")
    print("=" * 60)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    optimizer_A = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=CONFIG["lr_head"],
        weight_decay=CONFIG["weight_decay"],
    )

    for epoch in range(start_epoch, CONFIG["epochs_frozen"]):
        t0 = time.time()
        tl, ta = train_epoch(model, train_loader, optimizer_A, criterion, device, scaler)
        vl, va = val_epoch(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["train_acc"].append(ta)
        history["val_acc"].append(va)

        print(f"  [{epoch + 1}/{CONFIG['epochs_frozen']}] "
              f"train_loss={tl:.4f} train_acc={ta:.3f}  "
              f"val_loss={vl:.4f} val_acc={va:.3f}  ({elapsed:.0f}s)")

        if va > best_val_acc:
            best_val_acc = va
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer_A.state_dict(),
                "best_val_acc": best_val_acc,
                "history": history,
            }, CONFIG["save_path"])
            print(f"  * Best model saved (val_acc={va:.3f})")

    # ── Phase B: Partial unfreeze ──
    early_stopping = EarlyStopping(
        patience=CONFIG["early_stopping_patience"],
        min_delta=CONFIG["early_stopping_min_delta"],
    )
    print()
    print("=" * 60)
    print("Phase B: Partial unfreeze")
    print("=" * 60)

    for param in model.blocks[-2:].parameters():
        param.requires_grad = True

    phase_b_start = max(start_epoch - CONFIG["epochs_frozen"], 0)
    for i in range(phase_b_start, CONFIG["epochs_unfrozen"]):
        t0 = time.time()
        tl, ta = train_epoch(model, train_loader, optimizer_B, criterion, device, scaler)
        vl, va = val_epoch(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["train_acc"].append(ta)
        history["val_acc"].append(va)

        current_lr = scheduler.get_last_lr()[0]
        epoch_num = CONFIG["epochs_frozen"] + i + 1
        total_epochs = CONFIG["epochs_frozen"] + CONFIG["epochs_unfrozen"]
        print(f"  [{epoch_num}/{total_epochs}] "
              f"train_loss={tl:.4f} train_acc={ta:.3f}  "
              f"val_loss={vl:.4f} val_acc={va:.3f}  "
              f"lr={current_lr:.2e}  ({elapsed:.0f}s)")

        if va > best_val_acc:
            best_val_acc = va
            torch.save({
                "epoch": epoch_num,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer_B.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_acc": best_val_acc,
                "history": history,
            }, CONFIG["save_path"])
            print(f"  * Best model saved (val_acc={va:.3f})")

        if early_stopping.step(va):
            print(f"\n  Early stopping triggered at epoch {epoch_num}")
            print(f"  No improvement for {CONFIG['early_stopping_patience']} epochs")
            break

    history["stopped_epoch"] = epoch_num
    history["early_stopping_triggered"] = early_stopping.should_stop

    print()
    print(f"Best val accuracy: {best_val_acc:.3f}")

    # ── Test evaluation ──
    print()
    print("=" * 60)
    print("Test Evaluation")
    print("=" * 60)

    best_checkpoint = torch.load(CONFIG["save_path"], map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    test_loss, test_acc = val_epoch(model, test_loader, criterion, device)
    history["test_loss"] = test_loss
    history["test_acc"] = test_acc
    print(f"  Test loss: {test_loss:.4f}  Test accuracy: {test_acc:.3f}")
    print()

    # ── Save history ──
    history_path = SAVE_DIR / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")
    print("Done.")


if __name__ == "__main__":
    main()
