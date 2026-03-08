"""
Grad-CAM (Gradient-weighted Class Activation Mapping) utilities.
Generates saliency heatmaps for DenseNet-based classifiers.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


class GradCAM:
    """
    Grad-CAM implementation compatible with DenseNet feature extractors.

    Usage:
        cam = GradCAM(model, target_layer=model.features.denseblock4)
        heatmap = cam.generate(image_tensor, class_idx=None)  # None → argmax
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._gradients: Optional[torch.Tensor] = None
        self._activations: Optional[torch.Tensor] = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(_, __, output):
            self._activations = output.detach()

        def backward_hook(_, __, grad_output):
            self._gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(
        self,
        image_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
        output_size: tuple[int, int] = (224, 224),
    ) -> np.ndarray:
        """
        Produce a Grad-CAM heatmap as a uint8 BGR image (cv2 colormap applied).

        Args:
            image_tensor: (1, C, H, W) tensor, already preprocessed.
            class_idx: Target class index. If None, uses the predicted class.
            output_size: (height, width) for the returned heatmap.

        Returns:
            Colored heatmap as np.ndarray of shape (H, W, 3) in BGR uint8.
        """
        self.model.eval()
        image_tensor = image_tensor.requires_grad_(True)

        # Forward
        logits = self.model(image_tensor)

        if class_idx is None:
            if logits.shape[-1] == 1:
                class_idx = 0
            else:
                class_idx = int(logits.argmax(dim=-1).item())

        # Backward for target class
        self.model.zero_grad()
        score = logits[0, class_idx] if logits.dim() > 1 else logits[0]
        score.backward()

        # Grad-CAM weights: global average pooling of gradients
        gradients = self._gradients  # (1, C, H, W)
        activations = self._activations  # (1, C, H, W)

        if gradients is None or activations is None:
            raise RuntimeError("Hooks did not capture gradients/activations.")

        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * activations).sum(dim=1).squeeze(0)  # (H, W)
        cam = torch.relu(cam).cpu().numpy()

        # Normalize and resize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = (cam * 255).astype(np.uint8)
        cam = cv2.resize(cam, (output_size[1], output_size[0]))
        heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
        return heatmap

    def overlay(
        self,
        original_image: Image.Image,
        heatmap: np.ndarray,
        alpha: float = 0.4,
    ) -> np.ndarray:
        """
        Blend Grad-CAM heatmap with the original image.

        Args:
            original_image: PIL image.
            heatmap: Colored heatmap from `generate()`.
            alpha: Heatmap opacity (0 = original only, 1 = heatmap only).

        Returns:
            Blended BGR np.ndarray.
        """
        img_np = np.array(original_image.convert("RGB"))[:, :, ::-1]  # RGB → BGR
        heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
        blended = cv2.addWeighted(img_np, 1 - alpha, heatmap_resized, alpha, 0)
        return blended


# ─── Convenience functions ────────────────────────────────────────────────────

def get_last_densenet_layer(model: nn.Module) -> nn.Module:
    """Return the final DenseBlock of a DenseNet feature extractor."""
    return model.features.denseblock4


def generate_gradcam_overlay(
    model: nn.Module,
    image_tensor: torch.Tensor,
    original_image: Image.Image,
    class_idx: Optional[int] = None,
    alpha: float = 0.4,
) -> tuple[np.ndarray, float]:
    """
    End-to-end Grad-CAM: returns (overlay_bgr, confidence_score).

    Args:
        model: DenseNetClassifier in eval mode.
        image_tensor: Preprocessed (1, C, H, W) tensor.
        original_image: Original PIL image for overlay.
        class_idx: Target class. None → predicted class.
        alpha: Heatmap blend opacity.

    Returns:
        Tuple of (overlay np.ndarray, confidence float).
    """
    target_layer = get_last_densenet_layer(model)
    cam = GradCAM(model, target_layer)

    heatmap = cam.generate(image_tensor, class_idx=class_idx)
    overlay = cam.overlay(original_image, heatmap, alpha=alpha)

    # Confidence score
    with torch.no_grad():
        logits = model(image_tensor)
        if logits.shape[-1] == 1:
            confidence = float(torch.sigmoid(logits).item())
        else:
            probs = torch.softmax(logits, dim=-1)
            confidence = float(probs.max().item())

    return overlay, confidence
