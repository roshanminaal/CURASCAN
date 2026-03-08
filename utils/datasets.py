"""
PyTorch Dataset classes for CURASCAN data modalities:
  - ChestXrayDataset  : binary classification (Normal / Pneumonia)
  - MRIDataset        : 4-class brain tumor classification
  - CTSegDataset      : CT scan segmentation with mask pairs
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


# ─── Default transforms ───────────────────────────────────────────────────────

def classification_transform(augment: bool = False) -> transforms.Compose:
    ops = [transforms.Resize((224, 224))]
    if augment:
        ops += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]
    return transforms.Compose(ops)


def segmentation_transform(image_size: tuple[int, int] = (224, 224)):
    img_tf = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    mask_tf = transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])
    return img_tf, mask_tf


# ─── Chest X-ray Dataset ──────────────────────────────────────────────────────

class ChestXrayDataset(Dataset):
    """
    Expects directory layout:
        root/
          normal/    *.jpg | *.png   (or Normal)
          pneumonia/                 (or Pneumonia)

    Returns (tensor, label) where label ∈ {0: normal, 1: pneumonia}.
    """

    CLASSES = ["normal", "pneumonia"]
    _DIR_TO_LABEL: dict[str, int] = {
        "normal": 0, "Normal": 0,
        "pneumonia": 1, "Pneumonia": 1,
    }

    def __init__(
        self,
        root: str | Path,
        augment: bool = False,
        transform: Optional[Callable] = None,
    ):
        self.root = Path(root)
        self.transform = transform or classification_transform(augment)
        self.samples: list[tuple[Path, int]] = []

        for cls_dir in self.root.iterdir():
            if not cls_dir.is_dir():
                continue
            name = cls_dir.name
            if name not in self._DIR_TO_LABEL:
                continue
            label = self._DIR_TO_LABEL[name]
            for fp in cls_dir.iterdir():
                if fp.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    self.samples.append((fp, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), torch.tensor(label, dtype=torch.float32)


# ─── MRI Dataset ──────────────────────────────────────────────────────────────

class MRIDataset(Dataset):
    """
    Expects directory layout:
        root/
          glioma/      *.jpg | *.png   (or gilioma)
          meningioma/                  (or menigomia, minigilioma)
          notumor/
          pituitary/                  (or pitutiary)

    Returns (tensor, label) where label ∈ {0, 1, 2, 3}.
    """

    CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
    # Alternate folder names (e.g. typos) -> label index
    _DIR_TO_LABEL: dict[str, int] = {
        "glioma": 0, "gilioma": 0,
        "meningioma": 1, "menigomia": 1, "minigilioma": 1,
        "notumor": 2,
        "pituitary": 3, "pitutiary": 3,
    }

    def __init__(
        self,
        root: str | Path,
        augment: bool = False,
        transform: Optional[Callable] = None,
    ):
        self.root = Path(root)
        self.transform = transform or classification_transform(augment)
        self.samples: list[tuple[Path, int]] = []

        for cls_dir in self.root.iterdir():
            if not cls_dir.is_dir():
                continue
            name = cls_dir.name.lower()
            if name not in self._DIR_TO_LABEL:
                continue
            label = self._DIR_TO_LABEL[name]
            for fp in cls_dir.iterdir():
                if fp.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    self.samples.append((fp, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), torch.tensor(label, dtype=torch.long)


# ─── CT Dataset ───────────────────────────────────────────────────────────────

class CTDataset(Dataset):
    """
    Expects directory layout:
        root/
          healthy/     *.jpg | *.png
          tumor/       *.jpg | *.png

    Returns (tensor, label) where label ∈ {0: healthy, 1: tumor}.
    """

    CLASSES = ["healthy", "tumor"]
    _DIR_TO_LABEL: dict[str, int] = {
        "healthy": 0, "Healthy": 0,
        "tumor": 1, "Tumor": 1,
    }

    def __init__(
        self,
        root: str | Path,
        augment: bool = False,
        transform: Optional[Callable] = None,
    ):
        self.root = Path(root)
        self.transform = transform or classification_transform(augment)
        self.samples: list[tuple[Path, int]] = []

        for cls_dir in self.root.iterdir():
            if not cls_dir.is_dir():
                continue
            name = cls_dir.name
            if name not in self._DIR_TO_LABEL:
                continue
            label = self._DIR_TO_LABEL[name]
            for fp in cls_dir.iterdir():
                if fp.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    self.samples.append((fp, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), torch.tensor(label, dtype=torch.float32)



# ─── CT Segmentation Dataset ──────────────────────────────────────────────────

class CTSegDataset(Dataset):
    """
    CT segmentation dataset.

    Expected layout:
        data/ct/
          images/   <id>.png
          masks/    <id>.png  (binary: 0 background, 255 foreground)
          labels.csv           columns: [filename, label]

    Returns (image_tensor, mask_tensor) pair.
    """

    def __init__(
        self,
        root: str | Path,
        image_size: tuple[int, int] = (224, 224),
        augment: bool = False,
    ):
        self.root = Path(root)
        self.image_dir = self.root / "images"
        self.mask_dir = self.root / "masks"
        self.img_tf, self.mask_tf = segmentation_transform(image_size)
        self.augment = augment

        labels_csv = self.root / "labels.csv"
        if labels_csv.exists():
            df = pd.read_csv(labels_csv)
            self.filenames = df["filename"].tolist()
        else:
            self.filenames = [
                f.name for f in self.image_dir.iterdir()
                if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ]

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int):
        fname = self.filenames[idx]
        image = Image.open(self.image_dir / fname).convert("RGB")
        mask_path = self.mask_dir / fname
        if mask_path.exists():
            mask = Image.open(mask_path).convert("L")
        else:
            # Empty mask if not available
            mask = Image.fromarray(np.zeros((image.height, image.width), dtype=np.uint8))

        if self.augment:
            image, mask = self._augment(image, mask)

        return self.img_tf(image), (self.mask_tf(mask) > 0.5).float()

    @staticmethod
    def _augment(image: Image.Image, mask: Image.Image):
        import random
        if random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        angle = random.uniform(-15, 15)
        image = image.rotate(angle)
        mask = mask.rotate(angle)
        return image, mask
