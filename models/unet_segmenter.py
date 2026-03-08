"""
UNet segmentation model for medical image segmentation.
Designed for CT scan lesion segmentation; adaptable to other modalities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two consecutive Conv-BN-ReLU blocks."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    """Encoder block: MaxPool then DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    """Decoder block: upsample, concatenate skip, DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad if spatial dims differ
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    Standard UNet with configurable depth.

    Args:
        in_channels (int): Input image channels (1 for grayscale, 3 for RGB).
        out_channels (int): Number of output segmentation classes (1 for binary).
        features (list[int]): Feature map sizes at each encoder level.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: list = None,
    ):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        # Encoder
        self.input_conv = DoubleConv(in_channels, features[0])
        for i in range(1, len(features)):
            self.encoder.append(Down(features[i - 1], features[i]))

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder
        dec_features = [features[-1] * 2] + list(reversed(features))
        for i in range(len(features)):
            self.decoder.append(Up(dec_features[i], dec_features[i + 1]))

        # Final 1×1 conv
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []

        # Encoder pass
        x = self.input_conv(x)
        skips.append(x)
        for down in self.encoder:
            x = down(x)
            skips.append(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder pass
        skips = skips[::-1]  # reverse: deepest first
        for i, up in enumerate(self.decoder):
            x = up(x, skips[i])

        return self.final_conv(x)

    def predict_mask(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Returns binary segmentation mask after sigmoid thresholding."""
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        return (probs > threshold).float()


def load_from_checkpoint(model: UNet, checkpoint_path: str, device: str = "cpu") -> UNet:
    """Load UNet weights from a checkpoint, handling various save formats."""
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
