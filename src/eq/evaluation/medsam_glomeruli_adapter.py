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


class MedSAMNpyEvalDataset(Dataset):
    """Same NPY layout as training, but a fixed tight box (no bbox jitter) for val metrics."""

    def __init__(self, data_root: Path, *, max_examples: int = 0):
        self.data_root = Path(data_root)
        self.img_path = self.data_root / "imgs"
        self.gt_path = self.data_root / "gts"
        files = sorted(
            path for path in self.gt_path.glob("*.npy") if (self.img_path / path.name).exists()
        )
        if max_examples > 0:
            files = files[: int(max_examples)]
        self.gt_files = files

    def __len__(self) -> int:
        return len(self.gt_files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        gt_path = self.gt_files[index]
        img_name = gt_path.name
        image = np.load(self.img_path / img_name, allow_pickle=True).astype(np.float32)
        mask = (np.load(gt_path, allow_pickle=True) > 0).astype(np.uint8)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Expected RGB image array HxWx3, got {image.shape}")
        if mask.ndim != 2:
            raise ValueError(f"Expected 2D mask array, got {mask.shape}")
        if int(mask.sum()) <= 0:
            raise ValueError(f"Mask has no foreground pixels: {gt_path}")
        ys, xs = np.where(mask > 0)
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
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


@torch.no_grad()
def _validation_epoch_loss_and_dice(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    ce_loss: nn.Module,
) -> tuple[float, float]:
    """Mean batch loss and mean hard Dice (threshold 0.5) on the validation loader."""
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    n_batches = 0
    for image, gt, boxes, _names in loader:
        image = image.to(device)
        gt = gt.to(device)
        prediction = model(image, boxes.detach().cpu().numpy())
        loss = _soft_dice_loss_with_logits(prediction, gt) + ce_loss(prediction, gt)
        prob = torch.sigmoid(prediction)
        pred_bin = (prob > 0.5).float()
        intersection = (pred_bin * gt).sum(dim=(1, 2, 3))
        denom = pred_bin.sum(dim=(1, 2, 3)) + gt.sum(dim=(1, 2, 3))
        eps = 1e-6
        dice = ((2 * intersection + eps) / (denom + eps)).mean()
        total_loss += float(loss.detach().cpu().item())
        total_dice += float(dice.detach().cpu().item())
        n_batches += 1
    if n_batches == 0:
        return 0.0, 0.0
    return total_loss / n_batches, total_dice / n_batches


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
    val_loader: DataLoader | None = None
    val_root = str(getattr(args, "val_npy_root", "") or "").strip()
    if val_root:
        val_path = Path(val_root)
        val_mx = int(getattr(args, "val_max_examples", 0) or 0)
        val_ds = MedSAMNpyEvalDataset(val_path, max_examples=val_mx)
        if len(val_ds) > 0:
            val_loader = DataLoader(
                val_ds,
                batch_size=int(args.batch_size),
                shuffle=False,
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
    scheduler = None
    sched_name = str(args.lr_scheduler).strip().lower()
    if sched_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(args.epochs)),
            eta_min=float(args.min_lr),
        )
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    epoch_log_path = work_dir / "training_epochs.jsonl"
    losses: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_val_dice = -1.0
    best_val_epoch = -1
    total_epochs = int(args.epochs)
    val_b = len(val_loader) if val_loader is not None else 0
    print(
        "[medsam_glomeruli_adapter] "
        f"starting training device={device} epochs={total_epochs} "
        f"train_examples={len(dataset)} train_batches_per_epoch={len(loader)} "
        f"val_batches_per_epoch={val_b} epoch_log={epoch_log_path}",
        file=sys.stderr,
        flush=True,
    )
    with epoch_log_path.open("w", encoding="utf-8") as epoch_log:
        for epoch in range(total_epochs):
            print(
                "[medsam_glomeruli_adapter] "
                f"epoch {epoch + 1}/{total_epochs} train phase started "
                f"({len(loader)} batches; no per-batch logs)",
                file=sys.stderr,
                flush=True,
            )
            epoch_loss = 0.0
            model.train()
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
            if scheduler is not None:
                scheduler.step()
            lr_now = float(optimizer.param_groups[0]["lr"])
            val_mean_loss: float | None = None
            val_mean_dice: float | None = None
            if val_loader is not None:
                val_mean_loss, val_mean_dice = _validation_epoch_loss_and_dice(
                    model, val_loader, device, ce_loss
                )
                model.train()
            loss_row: dict[str, Any] = {
                "epoch": epoch,
                "loss": mean_loss,
                "lr": lr_now,
            }
            if val_mean_loss is not None and val_mean_dice is not None:
                loss_row["val_loss"] = val_mean_loss
                loss_row["val_dice"] = val_mean_dice
            losses.append(loss_row)
            row = {
                "epoch_index": epoch,
                "epoch_display": epoch + 1,
                "epochs_total": total_epochs,
                "mean_loss": mean_loss,
                "learning_rate": lr_now,
                "device": str(device),
                "batches_per_epoch": len(loader),
            }
            if val_mean_loss is not None:
                row["val_mean_loss"] = val_mean_loss
            if val_mean_dice is not None:
                row["val_mean_dice"] = val_mean_dice
            if val_loader is not None:
                row["val_batches_per_epoch"] = len(val_loader)
            epoch_log.write(json.dumps(row) + "\n")
            epoch_log.flush()
            msg = (
                "[medsam_glomeruli_adapter] "
                f"epoch {epoch + 1}/{total_epochs} "
                f"train_loss={mean_loss:.6f} lr={lr_now:.2e} "
                f"device={device} train_batches={len(loader)}"
            )
            if val_mean_loss is not None and val_mean_dice is not None:
                msg += (
                    f" | val_loss={val_mean_loss:.6f} val_dice={val_mean_dice:.6f}"
                    f" (hard Dice, thr=0.5)"
                )
            print(msg, file=sys.stderr, flush=True)
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
            if val_mean_dice is not None and val_mean_dice > best_val_dice:
                best_val_dice = val_mean_dice
                best_val_epoch = epoch
                torch.save(
                    checkpoint,
                    work_dir / "medsam_glomeruli_best_val_dice.pth",
                )
    checkpoint_files = [
        str(work_dir / "medsam_glomeruli_latest.pth"),
        str(work_dir / "medsam_glomeruli_best.pth"),
    ]
    best_val_path = work_dir / "medsam_glomeruli_best_val_dice.pth"
    if best_val_path.exists():
        checkpoint_files.append(str(best_val_path))
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
        "validation_examples": len(val_loader.dataset) if val_loader is not None else 0,
        "losses": losses,
        "best_val_dice": best_val_dice if val_loader is not None else None,
        "best_val_dice_epoch": best_val_epoch if val_loader is not None else None,
        "checkpoint_files": checkpoint_files,
        "epoch_progress_log": str(epoch_log_path),
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
    parser.add_argument(
        "--lr-scheduler",
        default="none",
        help="Learning-rate schedule: none | cosine (CosineAnnealingLR over all epochs).",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=0.0,
        help="Minimum learning rate when lr-scheduler=cosine (eta_min).",
    )
    parser.add_argument(
        "--val-npy-root",
        default="",
        help="Optional MedSAM NPY root (imgs/ + gts/) for validation metrics each epoch.",
    )
    parser.add_argument(
        "--val-max-examples",
        type=int,
        default=0,
        help="Cap validation examples (0 = all).",
    )
    return parser


def main() -> None:
    summary = run_training(build_arg_parser().parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
