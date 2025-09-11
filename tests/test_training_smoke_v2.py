from pathlib import Path

import cv2
import numpy as np
import pytest

from fastai.vision.all import resnet18, unet_learner
from fastai.losses import BCEWithLogitsLossFlat

from eq.data_management.datablock_loader import build_segmentation_dls


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


def test_unet_learner_initializes_with_v2_dls(tmp_path: Path):
    for i in range(6):
        _make_pair(tmp_path, f"s{i}")

    dls = build_segmentation_dls(tmp_path, bs=4, num_workers=0)
    learn = unet_learner(dls, resnet18, n_out=1)
    assert hasattr(learn, 'model') and learn.model is not None


def test_minimal_training_step_with_v2_dls(tmp_path: Path):
    """Test that we can actually run a minimal training step."""
    for i in range(8):
        _make_pair(tmp_path, f"s{i}")

    dls = build_segmentation_dls(tmp_path, bs=2, num_workers=0)
    learn = unet_learner(dls, resnet18, n_out=1)
    
    # Set the correct loss function for binary segmentation
    learn.loss_func = BCEWithLogitsLossFlat()  # type: ignore
    
    # Run one minimal training step
    learn.fit_one_cycle(1, 1e-3)
    
    # Verify training completed
    assert hasattr(learn, 'recorder')
    assert len(learn.recorder.log) > 0


