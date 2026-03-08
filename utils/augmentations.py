"""
Augmentation pipelines for medical imaging using torchvision transforms.
Separate pipelines for classification and segmentation (to keep mask in sync).
"""

from __future__ import annotations

import random
from typing import Tuple

import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF


# ─── Classification Augmentations ─────────────────────────────────────────────

def get_train_augmentation(image_size: int = 224) -> transforms.Compose:
    """Standard augmentation pipeline for classification training."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_val_augmentation(image_size: int = 224) -> transforms.Compose:
    """Minimal transform for validation/test (no random ops)."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ─── Segmentation Augmentations (image + mask synchronized) ──────────────────

class SegmentationAugmentation:
    """
    Applies identical geometric transforms to both image and mask.
    Color/intensity transforms are applied to the image only.
    """

    def __init__(
        self,
        image_size: int = 224,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.1,
        rotation_range: float = 15.0,
        brightness: float = 0.2,
        contrast: float = 0.2,
    ):
        self.image_size = image_size
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.rotation_range = rotation_range
        self.brightness = brightness
        self.contrast = contrast

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def __call__(self, image: Image.Image, mask: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        # Resize
        image = TF.resize(image, (self.image_size, self.image_size))
        mask = TF.resize(mask, (self.image_size, self.image_size),
                         interpolation=TF.InterpolationMode.NEAREST)

        # Random horizontal flip
        if random.random() < self.hflip_prob:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flip
        if random.random() < self.vflip_prob:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # Random rotation
        angle = random.uniform(-self.rotation_range, self.rotation_range)
        image = TF.rotate(image, angle)
        mask = TF.rotate(mask, angle)

        # Color jitter (image only)
        brightness_factor = 1.0 + random.uniform(-self.brightness, self.brightness)
        contrast_factor = 1.0 + random.uniform(-self.contrast, self.contrast)
        image = TF.adjust_brightness(image, brightness_factor)
        image = TF.adjust_contrast(image, contrast_factor)

        # To tensor
        img_tensor = TF.to_tensor(image)
        mask_tensor = (TF.to_tensor(mask) > 0.5).float()

        # Normalize image
        img_tensor = self.normalize(img_tensor)

        return img_tensor, mask_tensor


class SegmentationValTransform:
    """Deterministic resize-only transform for seg validation."""

    def __init__(self, image_size: int = 224):
        self.image_size = image_size
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def __call__(self, image: Image.Image, mask: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        image = TF.resize(image, (self.image_size, self.image_size))
        mask = TF.resize(mask, (self.image_size, self.image_size),
                         interpolation=TF.InterpolationMode.NEAREST)
        img_tensor = self.normalize(TF.to_tensor(image))
        mask_tensor = (TF.to_tensor(mask) > 0.5).float()
        return img_tensor, mask_tensor


# ─── Test-Time Augmentation (TTA) ─────────────────────────────────────────────

def tta_transforms(image_size: int = 224) -> list[transforms.Compose]:
    """
    Returns a list of TTA transforms.
    Average predictions over all transforms for improved robustness.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    base = [transforms.Resize((image_size, image_size)), transforms.ToTensor(), normalize]

    return [
        transforms.Compose(base),
        transforms.Compose([transforms.Resize((image_size, image_size)),
                             transforms.RandomHorizontalFlip(p=1.0),
                             transforms.ToTensor(), normalize]),
        transforms.Compose([transforms.Resize((image_size, image_size)),
                             transforms.RandomRotation((10, 10)),
                             transforms.ToTensor(), normalize]),
        transforms.Compose([transforms.Resize((image_size, image_size)),
                             transforms.RandomRotation((-10, -10)),
                             transforms.ToTensor(), normalize]),
    ]
