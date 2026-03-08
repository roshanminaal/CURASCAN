"""
Loss functions for CURASCAN training:
  - DiceLoss       : for segmentation
  - DiceBCELoss    : combined Dice + BCE for segmentation
  - FocalLoss      : for imbalanced classification
  - WeightedBCE    : weighted binary cross-entropy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft Dice loss for binary segmentation.
    Expects raw logits; applies sigmoid internally.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        return 1.0 - dice


class DiceBCELoss(nn.Module):
    """Combination of Dice loss and binary cross-entropy."""

    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.dice = DiceLoss(smooth)
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        dice = self.dice(logits, targets)
        return self.bce_weight * bce + self.dice_weight * dice


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in classification.

    Args:
        alpha (float): Weighting factor for the rare class.
        gamma (float): Focusing parameter. 0 → standard BCE.
        reduction (str): 'mean' | 'sum' | 'none'.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class WeightedBCE(nn.Module):
    """
    Binary cross-entropy with a scalar positive-class weight.
    Useful when positives are rare (e.g., pneumonia detection).

    Args:
        pos_weight (float): Weight multiplier for positive examples.
    """

    def __init__(self, pos_weight: float = 2.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weight_tensor = torch.tensor([self.pos_weight], device=logits.device)
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=weight_tensor)
