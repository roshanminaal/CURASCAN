"""
Training script for UNet segmentation model on CT scan data.

Usage:
    python train/train_segmentation.py --epochs 50 --batch_size 8
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.unet_segmenter import UNet
from utils.datasets import CTSegDataset
from utils.augmentations import SegmentationAugmentation, SegmentationValTransform
from utils.losses import DiceBCELoss
from utils.metrics import segmentation_metrics, MetricTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CURASCAN Segmentation Training")
    parser.add_argument("--data_dir", type=str, default="data/ct")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--image_size", type=int, default=224)
    return parser.parse_args()


def train_one_epoch(model, loader, optimizer, criterion, device, tracker):
    model.train()
    tracker.reset()
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        tracker.update(loss=loss.item())
    return tracker.averages()


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_dice, all_iou = 0.0, [], []

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        loss = criterion(outputs, masks)
        total_loss += loss.item()

        pred_masks = (torch.sigmoid(outputs) > 0.5).float()
        for p, t in zip(pred_masks.cpu(), masks.cpu()):
            m = segmentation_metrics(p, t)
            all_dice.append(m["dice"])
            all_iou.append(m["iou"])

    return {
        "loss": total_loss / len(loader),
        "dice": sum(all_dice) / len(all_dice) if all_dice else 0.0,
        "iou": sum(all_iou) / len(all_iou) if all_iou else 0.0,
    }


def main():
    args = parse_args()
    device_str = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    if device_str == "auto":
        device_str = "cpu"
    device = torch.device(device_str)
    print(f"Using device: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Dataset
    full_dataset = CTSegDataset(args.data_dir, image_size=(args.image_size, args.image_size))
    val_size = max(1, int(len(full_dataset) * args.val_split))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    print(f"Train: {train_size} | Val: {val_size}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    # Model
    model = UNet(in_channels=3, out_channels=1, features=[64, 128, 256, 512]).to(device)
    criterion = DiceBCELoss(dice_weight=0.5, bce_weight=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5)

    tracker = MetricTracker("loss")
    best_dice = float("-inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_m = train_one_epoch(model, train_loader, optimizer, criterion, device, tracker)
        val_m = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_m["dice"])

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Train Loss: {train_m['loss']:.4f} | "
            f"Val Loss: {val_m['loss']:.4f} | "
            f"Dice: {val_m['dice']:.4f} | "
            f"IoU: {val_m['iou']:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        if val_m["dice"] > best_dice:
            best_dice = val_m["dice"]
            save_path = os.path.join(args.checkpoint_dir, "seg_best.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_dice": best_dice,
            }, save_path)
            print(f"  Saved best model (Dice={best_dice:.4f}) -> {save_path}")

    print(f"\nTraining complete. Best Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()
