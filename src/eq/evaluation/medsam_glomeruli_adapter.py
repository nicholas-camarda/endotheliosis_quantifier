"""Constrained MedSAM glomeruli training adapter.

This adapter imports the local MedSAM repository modules at runtime. It does not
vendor MedSAM code into `src/eq`.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class MedSAMNpyDataset(Dataset):
    """Read MedSAM `imgs/` and `gts/` paired `.npy` training arrays."""

    def __init__(self, data_root: Path, *, bbox_shift: int = 20, max_examples: int = 0):
        self.data_root = Path(data_root)
        self.img_path = self.data_root / "imgs"
        self.gt_path = self.data_root / "gts"
        files = sorted(path for path in self.gt_path.glob("*.npy") if (self.img_path / path.name).exists())
        if max_examples > 0:
            files = files[: int(max_examples)]
        self.gt_files = files
        self.bbox_shift = int(bbox_shift)

    def __len__(self) -> int:
        return len(self.gt_files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        gt_path = self.gt_files[index]
        img_name = gt_path.name
        image = np.load(self.img_path / img_name, allow_pickle=True).astype(np.float32)
        mask = (np.load(gt_path, allow_pickle=True) > 0).astype(np.uint8)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Expected RGB image array HxWx3, got {image.shape}: {self.img_path / img_name}")
        if mask.ndim != 2:
            raise ValueError(f"Expected 2D mask array, got {mask.shape}: {gt_path}")
        if int(mask.sum()) <= 0:
            raise ValueError(f"Mask has no foreground pixels: {gt_path}")
        ys, xs = np.where(mask > 0)
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        height, width = mask.shape
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(width, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(height, y_max + random.randint(0, self.bbox_shift))
        image_chw = np.transpose(np.clip(image, 0.0, 1.0), (2, 0, 1))
        return (
            torch.tensor(image_chw).float(),
            torch.tensor(mask[None, :, :]).float(),
            torch.tensor([x_min, y_min, x_max, y_max]).float(),
            img_name,
        )


def _soft_dice_loss_with_logits(
    logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Differentiable soft Dice loss on logits (no MONAI dependency)."""
    prob = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = (prob * target).sum(dim=dims)
    denom = prob.pow(2).sum(dim=dims) + target.pow(2).sum(dim=dims)
    dice = (2 * intersection + eps) / (denom + eps)
    return 1 - dice.mean()


class MedSAMMaskDecoderAdapter(nn.Module):
    """Train a MedSAM mask decoder with frozen image and prompt encoders."""

    def __init__(self, image_encoder: nn.Module, mask_decoder: nn.Module, prompt_encoder: nn.Module):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            image_embedding = self.image_encoder(image)
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        return F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )


def _device(name: str) -> torch.device:
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Requested MPS device, but torch.backends.mps.is_available() is false")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device, but torch.cuda.is_available() is false")
    return torch.device(name)


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    medsam_repo = Path(args.medsam_repo).expanduser().resolve()
    sys.path.insert(0, str(medsam_repo))
    from segment_anything import sam_model_registry

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = _device(str(args.device))
    dataset = MedSAMNpyDataset(
        Path(args.train_npy_root),
        bbox_shift=int(args.bbox_shift),
        max_examples=int(args.max_examples),
    )
    if len(dataset) == 0:
        raise ValueError(f"No MedSAM training arrays found under {args.train_npy_root}")
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=0,
    )
    sam_model = sam_model_registry[str(args.model_type)](checkpoint=str(args.checkpoint))
    model = MedSAMMaskDecoderAdapter(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        [param for param in model.mask_decoder.parameters() if param.requires_grad],
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    losses: list[dict[str, Any]] = []
    best_loss = float("inf")
    for epoch in range(int(args.epochs)):
        epoch_loss = 0.0
        for image, gt, boxes, _names in loader:
            optimizer.zero_grad()
            image = image.to(device)
            gt = gt.to(device)
            prediction = model(image, boxes.detach().cpu().numpy())
            loss = _soft_dice_loss_with_logits(prediction, gt) + ce_loss(prediction, gt)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu().item())
        mean_loss = epoch_loss / max(1, len(loader))
        losses.append({"epoch": epoch, "loss": mean_loss})
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "loss": mean_loss,
            "adaptation_mode": "frozen_image_encoder_mask_decoder",
        }
        torch.save(checkpoint, work_dir / "medsam_glomeruli_latest.pth")
        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(checkpoint, work_dir / "medsam_glomeruli_best.pth")
    summary = {
        "status": "completed",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "train_npy_root": str(Path(args.train_npy_root)),
        "medsam_repo": str(medsam_repo),
        "checkpoint": str(Path(args.checkpoint)),
        "work_dir": str(work_dir),
        "device": str(device),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "training_examples": len(dataset),
        "losses": losses,
        "checkpoint_files": [
            str(work_dir / "medsam_glomeruli_latest.pth"),
            str(work_dir / "medsam_glomeruli_best.pth"),
        ],
    }
    (work_dir / "adapter_training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train constrained MedSAM glomeruli adapter.")
    parser.add_argument("--medsam-repo", required=True)
    parser.add_argument("--train-npy-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--model-type", default="vit_b")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--bbox-shift", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2023)
    return parser


def main() -> None:
    summary = run_training(build_arg_parser().parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
