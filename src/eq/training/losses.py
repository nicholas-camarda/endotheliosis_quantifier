#!/usr/bin/env python3
"""
Custom segmentation losses compatible with 2-class logits.
"""

from typing import Optional
import torch
import torch.nn.functional as F


def _one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    if labels.ndim == 4 and labels.shape[1] == 1:
        labels = labels[:, 0]
    return F.one_hot(labels.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()


class DiceLoss(torch.nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: [B, C, H, W], target: [B, H, W] or [B,1,H,W]
        probs = torch.softmax(logits, dim=1)
        target_oh = _one_hot(target, num_classes=logits.shape[1])
        # foreground channel assumed index 1 for binary
        probs_fg = probs[:, 1]
        target_fg = target_oh[:, 1]
        inter = (probs_fg * target_fg).sum(dim=(1, 2))
        union = probs_fg.sum(dim=(1, 2)) + target_fg.sum(dim=(1, 2))
        dice = (2.0 * inter + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class BCEDiceLoss(torch.nn.Module):
    def __init__(self, dice_weight: float = 0.5):
        super().__init__()
        self.dice = DiceLoss()
        self.dice_weight = float(dice_weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Use CE for multiclass equivalent to BCE for two-class
        ce = F.cross_entropy(logits, target.long())
        return (1 - self.dice_weight) * ce + self.dice_weight * self.dice(logits, target)


class TverskyLoss(torch.nn.Module):
    def __init__(self, alpha: float = 0.5, beta: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        target_oh = _one_hot(target, num_classes=logits.shape[1])
        p = probs[:, 1]
        t = target_oh[:, 1]
        tp = (p * t).sum(dim=(1, 2))
        fp = (p * (1 - t)).sum(dim=(1, 2))
        fn = ((1 - p) * t).sum(dim=(1, 2))
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky.mean()


def make_loss(name: str):
    key = (name or "").strip().lower()
    if key in ("dice", "diceloss"):
        return DiceLoss()
    if key in ("bcedice", "bce_dice", "bce+dice"):
        return BCEDiceLoss()
    if key in ("tversky", "focal_tversky"):  # default alpha=beta=0.5 here
        return TverskyLoss()
    return None


