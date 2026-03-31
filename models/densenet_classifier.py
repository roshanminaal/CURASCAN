"""
DenseNet121-based classifier for medical image classification.
Supports binary classification (Normal vs Anomaly) and multi-class (MRI tumor types).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class DenseNetClassifier(nn.Module):
    """
    DenseNet121 classifier with a custom head.

    Args:
        num_classes (int): Number of output classes. Use 1 for binary (sigmoid), >1 for multi-class (softmax).
        pretrained (bool): Load ImageNet pretrained weights.
        dropout (float): Dropout rate in the classifier head.
    """

    def __init__(self, num_classes: int = 1, pretrained: bool = True, dropout: float = 0.5):
        super().__init__()
        self.num_classes = num_classes

        # Load backbone
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.densenet121(weights=weights)

        # Extract feature layers (everything except the original classifier)
        self.features = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        in_features = backbone.classifier.in_features  # 1024 for DenseNet121

        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

    def predict(self, x: torch.Tensor):
        """
        Returns (logits, probabilities) tuple.
        For binary (num_classes=1): applies sigmoid.
        For multi-class: applies softmax.
        """
        logits = self.forward(x)
        if self.num_classes == 1:
            probs = torch.sigmoid(logits)
        else:
            probs = torch.softmax(logits, dim=-1)
        return logits, probs
        
def build_xray_classifier(pretrained: bool = True) -> DenseNetClassifier:
    """Binary classifier for chest X-ray pneumonia detection."""
    return DenseNetClassifier(num_classes=1, pretrained=pretrained)

def build_ct_classifier(pretrained: bool = True) -> DenseNetClassifier:
    """Binary classifier for CT scan tumor detection."""
    return DenseNetClassifier(num_classes=1, pretrained=pretrained)


def build_mri_classifier(pretrained: bool = True) -> DenseNetClassifier:
    """4-class MRI brain tumor classifier (glioma, meningioma, notumor, pituitary)."""
    return DenseNetClassifier(num_classes=4, pretrained=pretrained)


def load_from_checkpoint(model: DenseNetClassifier, checkpoint_path: str, device: str = "cpu") -> DenseNetClassifier:
    """
    Load model weights from a checkpoint file.

    Args:
        model: Instantiated DenseNetClassifier.
        checkpoint_path: Path to .pth checkpoint.
        device: Torch device string.

    Returns:
        Model with loaded weights in eval mode.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict):
        state_dict = (
            checkpoint.get("model_state_dict")
            or checkpoint.get("state_dict")
            or checkpoint
        )
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
