#!/usr/bin/env python3
"""Custom segmentation losses compatible with 2-class logits."""

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


def _parse_loss_name(name: str) -> tuple[str, dict[str, float]]:
    raw = (name or "").strip().lower()
    if not raw:
        return "", {}
    if ":" not in raw:
        return raw, {}
    key, raw_params = raw.split(":", 1)
    params: dict[str, float] = {}
    for part in raw_params.split(","):
        if not part.strip():
            continue
        if "=" not in part:
            raise ValueError(f"Invalid loss parameter {part!r}; expected key=value.")
        param_key, param_value = part.split("=", 1)
        params[param_key.strip()] = float(param_value)
    return key.strip(), params


def loss_metadata(name: str) -> dict[str, object]:
    key, params = _parse_loss_name(name)
    if key == "":
        return {
            "requested_loss_name": None,
            "resolved_loss_class": None,
            "loss_parameters": {},
            "false_positive_penalizing": False,
        }
    if key in ("dice", "diceloss"):
        return {
            "requested_loss_name": name,
            "resolved_loss_class": "DiceLoss",
            "loss_parameters": {"smooth": 1.0},
            "false_positive_penalizing": False,
        }
    if key in ("bcedice", "bce_dice", "bce+dice"):
        return {
            "requested_loss_name": name,
            "resolved_loss_class": "BCEDiceLoss",
            "loss_parameters": {"dice_weight": 0.5},
            "false_positive_penalizing": False,
        }
    if key in ("tversky", "focal_tversky", "tversky_fp"):
        alpha = float(params.get("alpha", 0.7 if key == "tversky_fp" else 0.5))
        beta = float(params.get("beta", 0.3 if key == "tversky_fp" else 0.5))
        smooth = float(params.get("smooth", 1.0))
        if alpha < 0 or beta < 0 or smooth <= 0:
            raise ValueError("Tversky loss requires alpha >= 0, beta >= 0, and smooth > 0.")
        return {
            "requested_loss_name": name,
            "resolved_loss_class": "TverskyLoss",
            "loss_parameters": {"alpha": alpha, "beta": beta, "smooth": smooth},
            "false_positive_penalizing": bool(alpha > beta),
        }
    raise ValueError(
        f"Unsupported segmentation loss {name!r}. "
        "Use dice, bcedice, tversky, tversky_fp, or tversky:alpha=<float>,beta=<float>."
    )


def make_loss(name: str):
    metadata = loss_metadata(name)
    cls = metadata["resolved_loss_class"]
    params = metadata["loss_parameters"]
    if cls is None:
        return None
    if cls == "DiceLoss":
        return DiceLoss(smooth=float(params["smooth"]))
    if cls == "BCEDiceLoss":
        return BCEDiceLoss(dice_weight=float(params["dice_weight"]))
    if cls == "TverskyLoss":
        return TverskyLoss(
            alpha=float(params["alpha"]),
            beta=float(params["beta"]),
            smooth=float(params["smooth"]),
        )
    raise ValueError(f"Unsupported resolved loss class {cls!r}.")

