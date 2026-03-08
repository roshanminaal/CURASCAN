"""
Training script for DenseNet121 classification models.
Supports both X-ray (binary) and MRI (multi-class) tasks.

Usage:
    python train/train_classification.py --task xray --epochs 30 --batch_size 32
    python train/train_classification.py --task mri   --epochs 50 --batch_size 16
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.densenet_classifier import build_xray_classifier, build_mri_classifier, build_ct_classifier
from utils.datasets import ChestXrayDataset, MRIDataset, CTDataset
from utils.augmentations import get_train_augmentation, get_val_augmentation
from utils.losses import FocalLoss
from utils.metrics import binary_metrics, MetricTracker


# ─── Config ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CURASCAN Classification Training")
    parser.add_argument("--task", choices=["xray", "mri", "ct"], default="xray")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Freeze DenseNet backbone, train classifier head only")
    return parser.parse_args()


# ─── Training loop ────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, tracker):
    model.train()
    tracker.reset()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        if outputs.shape[-1] == 1:
            loss = criterion(outputs.squeeze(1), labels)
        else:
            loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()
        tracker.update(loss=loss.item())
    return tracker.averages()


@torch.no_grad()
def evaluate(model, loader, criterion, device, task="xray"):
    model.eval()
    all_probs, all_labels = [], []
    total_loss = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        if outputs.shape[-1] == 1:
            loss = criterion(outputs.squeeze(1), labels)
            probs = torch.sigmoid(outputs.squeeze(1))
        else:
            loss = criterion(outputs, labels.long())
            probs = torch.softmax(outputs, dim=-1).max(dim=-1).values

        total_loss += loss.item()
        all_probs.extend(probs.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    metrics = binary_metrics(all_probs, all_labels) if task in ["xray", "ct"] else {}
    metrics["loss"] = total_loss / len(loader)
    return metrics


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device_str = args.device
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"Using device: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ── Datasets ──
    data_root = Path(args.data_dir)
    if args.task == "xray":
        train_tf = get_train_augmentation()
        val_tf = get_val_augmentation()
        train_ds = ChestXrayDataset(data_root / "xray" / "train", transform=train_tf)
        val_ds = ChestXrayDataset(data_root / "xray" / "val", transform=val_tf)
        model = build_xray_classifier(pretrained=True).to(device)
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
    elif args.task == "ct":
        train_tf = get_train_augmentation()
        val_tf = get_val_augmentation()
        # Fallback to creating datasets even if 'val' doesn't exist, using 'test' folder
        try:
            train_ds = CTDataset(data_root / "ct" / "train", transform=train_tf)
        except Exception:
            train_ds = CTDataset(data_root / "ct" / "test", transform=train_tf)
        val_ds = CTDataset(data_root / "ct" / "test", transform=val_tf)
        
        model = build_ct_classifier(pretrained=True).to(device)
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
    else:
        train_tf = get_train_augmentation()
        val_tf = get_val_augmentation()
        train_ds = MRIDataset(data_root / "mri" / "train", transform=train_tf)
        val_ds = MRIDataset(data_root / "mri" / "test", transform=val_tf)
        model = build_mri_classifier(pretrained=True).to(device)
        criterion = nn.CrossEntropyLoss()

    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                               shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # ── Freeze backbone (optional) ──
    if args.freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
        print("Backbone frozen - training classifier head only.")

    # ── Optimizer & scheduler ──
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Training ──
    tracker = MetricTracker("loss")
    best_val_score = float("-inf")
    
    if args.task == "xray":
        checkpoint_name = "cls_best.pth"
    elif args.task == "ct":
        checkpoint_name = "cls_ct_best.pth"
    else:
        checkpoint_name = "cls_mri_best.pth"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, tracker)
        val_metrics = evaluate(model, val_loader, criterion, device, task=args.task)
        scheduler.step()

        val_score = val_metrics.get("f1", 1 - val_metrics["loss"])
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"F1: {val_metrics.get('f1', 0):.4f} | "
            f"AUC: {val_metrics.get('auc', 0):.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        if val_score > best_val_score:
            best_val_score = val_score
            save_path = os.path.join(args.checkpoint_dir, checkpoint_name)
            torch.save(model.state_dict(), save_path)
            print(f"  Saved best model -> {save_path}")

    print(f"\nTraining complete. Best validation score: {best_val_score:.4f}")


if __name__ == "__main__":
    main()
