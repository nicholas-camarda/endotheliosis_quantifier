import os
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from eq.data_management.data_loader import (
    DataConfig,
    SegmentationDataLoader,
    create_data_loader,
    validate_dataset,
)


def test_find_data_files_and_validation(tmp_path: Path):
    # Create synthetic dataset structure
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    # Create one valid RGB image and matching binary mask
    img1 = np.zeros((128, 128, 3), dtype=np.uint8)
    img1[32:96, 32:96, :] = 255
    img1_path = images_dir / "sample_img1.jpg"
    cv2.imwrite(str(img1_path), cv2.cvtColor(img1, cv2.COLOR_RGB2BGR))

    mask1 = np.zeros((128, 128), dtype=np.uint8)
    mask1[40:88, 40:88] = 255
    mask1_path = masks_dir / "sample_img1_mask.png"
    cv2.imwrite(str(mask1_path), mask1)

    # Create one image without a corresponding mask (should be ignored with a warning)
    img2 = np.full((128, 128, 3), 127, dtype=np.uint8)
    img2_path = images_dir / "lonely_image.jpg"
    cv2.imwrite(str(img2_path), cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))

    loader = create_data_loader(DataConfig(validate_data=True))

    image_files, mask_files = loader.find_data_files(tmp_path, task_type="glomeruli")
    assert len(image_files) == 1
    assert len(mask_files) == 1
    assert image_files[0].name == img1_path.name
    assert mask_files[0].name == mask1_path.name

    # Validation should pass for valid pair
    assert loader.validate_image(img1_path) is True
    assert loader.validate_mask(mask1_path) is True


def test_load_and_preprocess(tmp_path: Path):
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    img = np.zeros((128, 128, 3), dtype=np.uint8)
    img[:, :, 0] = 255
    img_path = images_dir / "im.jpg"
    cv2.imwrite(str(img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[10:50, 10:50] = 255
    mask_path = masks_dir / "im_mask.png"
    cv2.imwrite(str(mask_path), mask)

    cfg = DataConfig(image_size=64)
    loader = SegmentationDataLoader(cfg)

    loaded_img = loader.load_image(img_path)
    loaded_mask = loader.load_mask(mask_path)
    assert loaded_img.shape == (128, 128, 3)
    assert loaded_mask.shape == (128, 128)

    t_img = loader.preprocess_image(loaded_img, split="val")
    assert isinstance(t_img, torch.Tensor)
    # Albumentations ToTensorV2 returns CHW
    assert t_img.shape == (3, 64, 64)
    assert torch.is_floating_point(t_img)

    t_mask = loader.preprocess_mask(loaded_mask, num_classes=2)
    assert t_mask.dtype == torch.long
    assert t_mask.shape == (128, 128)
    # Binary conversion (0/1)
    unique_vals = torch.unique(t_mask).tolist()
    assert set(unique_vals).issubset({0, 1})


def test_create_dataset_and_splits(tmp_path: Path):
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    # Create 10 paired samples
    num_pairs = 10
    for i in range(num_pairs):
        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        img_path = images_dir / f"img_{i}.jpg"
        cv2.imwrite(str(img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        mask = np.zeros((128, 128), dtype=np.uint8)
        mask[20:40, 20:40] = 255
        mask_path = masks_dir / f"img_{i}_mask.png"
        cv2.imwrite(str(mask_path), mask)

    loader = create_data_loader(DataConfig())
    dataset = loader.create_dataset(tmp_path, task_type="glomeruli", split_ratio=0.7, seed=123)

    assert set(dataset.keys()) == {"train", "val"}
    assert len(dataset["train"]) == int(0.7 * num_pairs)
    assert len(dataset["val"]) == num_pairs - int(0.7 * num_pairs)

    # Ensure pairs are Paths and exist
    for img_p, mask_p in dataset["train"] + dataset["val"]:
        assert isinstance(img_p, Path) and isinstance(mask_p, Path)
        assert img_p.exists() and mask_p.exists()


def test_get_data_info_and_validate_dataset(tmp_path: Path):
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    # Create 3 pairs of 128x128
    for i in range(3):
        img = np.zeros((128, 128, 3), dtype=np.uint8)
        img[30:90, 30:90, :] = 200
        ip = images_dir / f"case_{i}.png"
        cv2.imwrite(str(ip), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        mk = np.zeros((128, 128), dtype=np.uint8)
        mk[40:80, 40:80] = 255
        mp = masks_dir / f"case_{i}_mask.png"
        cv2.imwrite(str(mp), mk)

    loader = create_data_loader()
    info = loader.get_data_info(tmp_path, task_type="mitochondria")
    assert "error" not in info
    assert info["total_pairs"] == 3
    assert all(isinstance(s, tuple) and len(s) == 2 for s in info["sample_image_sizes"])
    assert info["image_channels"] == 3

    res = validate_dataset(tmp_path, task_type="mitochondria")
    assert res["valid"] is True
    assert res["total_samples"] == res["train_samples"] + res["val_samples"]


