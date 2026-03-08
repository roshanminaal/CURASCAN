"""
Evaluation metrics for CURASCAN models.
All functions accept numpy arrays or PyTorch tensors.
"""

from __future__ import annotations

import numpy as np
import torch


def to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# ─── Classification Metrics ───────────────────────────────────────────────────

def accuracy(preds: np.ndarray | torch.Tensor, targets: np.ndarray | torch.Tensor) -> float:
    """Binary or multi-class accuracy."""
    p, t = to_numpy(preds), to_numpy(targets)
    return float((p == t).mean())


def binary_metrics(
    probs: np.ndarray | torch.Tensor,
    targets: np.ndarray | torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Compute precision, recall, F1, and accuracy for binary classification.

    Args:
        probs: Predicted probabilities (after sigmoid), shape (N,).
        targets: Ground-truth labels {0, 1}, shape (N,).
        threshold: Decision threshold.

    Returns:
        Dictionary with keys: accuracy, precision, recall, f1, auc (if scikit-learn available).
    """
    p, t = to_numpy(probs).ravel(), to_numpy(targets).ravel()
    preds = (p >= threshold).astype(int)

    tp = int(((preds == 1) & (t == 1)).sum())
    fp = int(((preds == 1) & (t == 0)).sum())
    fn = int(((preds == 0) & (t == 1)).sum())
    tn = int(((preds == 0) & (t == 0)).sum())

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    result = {
        "accuracy": round(acc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }

    try:
        from sklearn.metrics import roc_auc_score
        result["auc"] = round(float(roc_auc_score(t, p)), 4)
    except Exception:
        pass

    return result


# ─── Segmentation Metrics ─────────────────────────────────────────────────────

def dice_coefficient(
    pred_mask: np.ndarray | torch.Tensor,
    true_mask: np.ndarray | torch.Tensor,
    smooth: float = 1.0,
) -> float:
    """
    Dice coefficient for binary segmentation masks.
    Both inputs should be binary (0/1) arrays.
    """
    p, t = to_numpy(pred_mask).ravel().astype(float), to_numpy(true_mask).ravel().astype(float)
    intersection = (p * t).sum()
    return float((2.0 * intersection + smooth) / (p.sum() + t.sum() + smooth))


def iou_score(
    pred_mask: np.ndarray | torch.Tensor,
    true_mask: np.ndarray | torch.Tensor,
    smooth: float = 1.0,
) -> float:
    """Intersection over Union (Jaccard Index)."""
    p, t = to_numpy(pred_mask).ravel().astype(float), to_numpy(true_mask).ravel().astype(float)
    intersection = (p * t).sum()
    union = p.sum() + t.sum() - intersection
    return float((intersection + smooth) / (union + smooth))


def pixel_accuracy(
    pred_mask: np.ndarray | torch.Tensor,
    true_mask: np.ndarray | torch.Tensor,
) -> float:
    """Simple pixel-wise accuracy."""
    p, t = to_numpy(pred_mask).ravel(), to_numpy(true_mask).ravel()
    return float((p == t).mean())


def segmentation_metrics(
    pred_mask: np.ndarray | torch.Tensor,
    true_mask: np.ndarray | torch.Tensor,
) -> dict[str, float]:
    """Convenience function returning Dice, IoU, and pixel accuracy in one call."""
    return {
        "dice": dice_coefficient(pred_mask, true_mask),
        "iou": iou_score(pred_mask, true_mask),
        "pixel_accuracy": pixel_accuracy(pred_mask, true_mask),
    }


# ─── Training Utilities ───────────────────────────────────────────────────────

class MetricTracker:
    """Accumulates metrics over batches and computes epoch-level averages."""

    def __init__(self, *keys: str):
        self._totals: dict[str, float] = {k: 0.0 for k in keys}
        self._counts: dict[str, int] = {k: 0 for k in keys}

    def update(self, **kwargs: float):
        for k, v in kwargs.items():
            self._totals[k] = self._totals.get(k, 0.0) + v
            self._counts[k] = self._counts.get(k, 0) + 1

    def avg(self, key: str) -> float:
        return self._totals[key] / (self._counts[key] + 1e-8)

    def averages(self) -> dict[str, float]:
        return {k: self.avg(k) for k in self._totals}

    def reset(self):
        for k in self._totals:
            self._totals[k] = 0.0
            self._counts[k] = 0
