from pathlib import Path

import cv2
import numpy as np
import pytest

from fastai.vision.all import PILMask

from eq.data_management.datablock_loader import (
    build_segmentation_datablock,
    default_get_y,
)


def _make_pair(tmp_path: Path, name: str, size: int = 96):
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[20:60, 20:60, :] = 180
    ip = images_dir / f"{name}.jpg"
    cv2.imwrite(str(ip), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    msk = np.zeros((size, size), dtype=np.uint8)
    msk[25:55, 25:55] = 255
    mp = masks_dir / f"{name}_mask.png"
    cv2.imwrite(str(mp), msk)

    return ip, mp


def test_default_get_y(tmp_path: Path):
    ip, mp = _make_pair(tmp_path, "case1")
    m = default_get_y(ip)
    assert isinstance(m, PILMask)


def test_build_datablock_and_dataloaders(tmp_path: Path):
    # Create multiple pairs
    for i in range(8):
        _make_pair(tmp_path, f"s{i}")

    db = build_segmentation_datablock()
    dls = db.dataloaders(tmp_path, bs=4, num_workers=0)

    xb, yb = dls.one_batch()
    # Image batch: (B, C, H, W), Mask batch: (B, H, W)
    assert xb.ndim == 4 and yb.ndim == 3
    assert xb.shape[0] == yb.shape[0]
    # Check mask codes are binary (0 or 1) after initial categorical processing
    unique = np.unique(yb.cpu().numpy())
    assert set(unique.tolist()).issubset({0, 1})


