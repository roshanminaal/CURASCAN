"""
K-Fold cross-validation training for the classification model.
Trains k models and reports per-fold and averaged metrics.

Usage:
    python train/kfold_training.py --task xray --k 5 --epochs 20
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.densenet_classifier import build_xray_classifier, build_mri_classifier
from utils.datasets import ChestXrayDataset, MRIDataset
from utils.augmentations import get_train_augmentation, get_val_augmentation
from utils.losses import FocalLoss
from utils.metrics import binary_metrics, MetricTracker


def parse_args():
    parser = argparse.ArgumentParser(description="K-Fold Cross-Validation Training")
    parser.add_argument("--task", choices=["xray", "mri"], default="xray")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/kfold")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def train_fold(model, train_loader, val_loader, criterion, args, device, fold_idx):
    """Train a single fold and return best validation metrics."""
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    tracker = MetricTracker("loss")
    best_metrics = {}
    best_score = 0.0

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        tracker.reset()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(images)
            if out.shape[-1] == 1:
                loss = criterion(out.squeeze(1), labels)
            else:
                loss = criterion(out, labels.long())
            loss.backward()
            optimizer.step()
            tracker.update(loss=loss.item())

        # Validate
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                out = model(images)
                if out.shape[-1] == 1:
                    probs = torch.sigmoid(out.squeeze(1))
                else:
                    probs = torch.softmax(out, dim=-1).max(dim=-1).values
                all_probs.extend(probs.cpu().numpy().tolist())
                all_labels.extend(labels.numpy().tolist())

        metrics = binary_metrics(all_probs, all_labels)
        score = metrics.get("f1", 0.0)
        if score > best_score:
            best_score = score
            best_metrics = metrics.copy()
            # Save fold checkpoint
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(),
                       os.path.join(args.checkpoint_dir, f"fold_{fold_idx}_best.pth"))

        scheduler.step()
        print(f"  Fold {fold_idx} | Epoch {epoch:02d}/{args.epochs} | "
              f"Loss: {tracker.avg('loss'):.4f} | F1: {metrics.get('f1',0):.4f} | "
              f"AUC: {metrics.get('auc',0):.4f}")

    return best_metrics


def main():
    args = parse_args()
    device_str = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    if device_str == "auto":
        device_str = "cpu"
    device = torch.device(device_str)
    print(f"Device: {device} | Task: {args.task} | K: {args.k}")

    data_root = Path(args.data_dir)

    # Build full dataset (train + val combined)
    if args.task == "xray":
        train_ds = ChestXrayDataset(data_root / "xray" / "train", transform=get_val_augmentation())
        val_ds = ChestXrayDataset(data_root / "xray" / "val", transform=get_val_augmentation())
        full_ds = ConcatDataset([train_ds, val_ds])
        all_labels = [s[1] for s in train_ds.samples] + [s[1] for s in val_ds.samples]
        build_fn = build_xray_classifier
        criterion = FocalLoss()
    else:
        train_ds = MRIDataset(data_root / "mri" / "train", transform=get_val_augmentation())
        val_ds = MRIDataset(data_root / "mri" / "test", transform=get_val_augmentation())
        full_ds = ConcatDataset([train_ds, val_ds])
        all_labels = [s[1] for s in train_ds.samples] + [s[1] for s in val_ds.samples]
        build_fn = build_mri_classifier
        criterion = nn.CrossEntropyLoss()

    all_labels = np.array(all_labels)
    indices = np.arange(len(all_labels))
    skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=42)

    fold_results = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(indices, all_labels), 1):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx}/{args.k} | Train: {len(train_idx)} | Val: {len(val_idx)}")
        print(f"{'='*60}")

        # Apply augmentation to train subset
        train_subset = Subset(full_ds, train_idx)
        val_subset = Subset(full_ds, val_idx)

        train_loader = DataLoader(train_subset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=4, pin_memory=True)

        model = build_fn(pretrained=True).to(device)
        metrics = train_fold(model, train_loader, val_loader, criterion, args, device, fold_idx)
        fold_results.append(metrics)
        print(f"  ✅ Fold {fold_idx} best: {metrics}")

    # Summary
    print("\n" + "="*60)
    print(f"K-FOLD CV RESULTS ({args.k} folds)")
    print("="*60)
    for key in ["accuracy", "precision", "recall", "f1", "auc"]:
        vals = [r[key] for r in fold_results if key in r]
        if vals:
            print(f"  {key:12s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")


if __name__ == "__main__":
    main()
