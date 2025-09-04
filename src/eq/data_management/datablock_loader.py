"""
FastAI v2 DataBlock loader for binary segmentation.

Provides a minimal, explicit DataBlock for image-mask datasets using
FastAI v2 APIs. This is the migration target from legacy
SegmentationDataLoaders to the DataBlock approach.
"""

from pathlib import Path
from typing import Callable, Optional, List, Any, Union

from fastai.data.block import DataBlock
from fastai.data.transforms import RandomSplitter
from fastai.vision.all import (
    ImageBlock, MaskBlock, PILMask, get_image_files,
    Resize, aug_transforms,
    Normalize, IntToFloatTensor, imagenet_stats
)
import numpy as np
from eq.core.constants import (
    DEFAULT_VAL_RATIO,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_FLIP_VERT,
    DEFAULT_MAX_ROTATE,
    DEFAULT_MIN_ZOOM,
    DEFAULT_MAX_ZOOM,
    DEFAULT_MAX_WARP,
    DEFAULT_MAX_LIGHTING
)
from eq.data_management.standard_getters import get_y
from eq.utils.logger import get_logger


def build_segmentation_datablock(
    codes: Optional[List[int]] = None,
    get_items: Optional[Callable[[Path], List[Any]]] = None,
    get_y_func: Optional[Callable[[Path], PILMask]] = None,
    splitter: Optional[Callable] = None,
    item_tfms: Optional[List] = None,
    batch_tfms: Optional[List] = None,
):
    """
    Create a DataBlock for binary segmentation with FastAI v2 best practices.

    Args:
        codes: Label codes for MaskBlock (binary by default)
        get_items: Function to list image files (defaults to get_image_files on images/)
        get_y_func: Function mapping image path -> PILMask
        splitter: Train/valid splitter (defaults to 80/20 RandomSplitter)
        item_tfms: Item transforms for augmentation (defaults to segmentation-appropriate transforms)
        batch_tfms: Batch transforms for normalization (defaults to standard transforms)
    """
    if codes is None:
        codes = [0, 1]
    
    if get_items is None:
        def _get_items(path: Path) -> List[Any]:
            # Try both possible directory structures
            images_dir = Path(path) / "images"
            if not images_dir.exists():
                images_dir = Path(path) / "image_patches"
            
            # Filter images to only include those with corresponding masks
            all_images = get_image_files(images_dir)
            valid_images = []
            skipped_count = 0
            
            logger = get_logger("eq.datablock_loader")
            
            for img_path in all_images:
                # Check if mask exists using same logic as get_y function
                mask_found = False
                
                # Strategy 1: Try derived data structure (mask_patches directory)
                mask_path = img_path.parent.parent / "mask_patches" / f"{img_path.stem}_mask{img_path.suffix}"
                if mask_path.exists():
                    mask_found = True
                
                # Strategy 2: Try standard path replacement (img_ -> mask_, .jpg -> .png)
                if not mask_found:
                    standard_mask_path = str(img_path).replace('.jpg', '.png').replace('.jpeg', '.png').replace('img_', 'mask_')
                    if Path(standard_mask_path).exists():
                        mask_found = True
                
                if mask_found:
                    valid_images.append(img_path)
                else:
                    skipped_count += 1
                    logger.warning(f"Skipping image {img_path.name} - no corresponding mask found")
            
            logger.info(f"DataBlock: Found {len(valid_images)} valid image-mask pairs, skipped {skipped_count} images without masks")
            return valid_images
        get_items = _get_items

    if get_y_func is None:
        get_y_func = get_y

    if splitter is None:
        splitter = RandomSplitter(valid_pct=DEFAULT_VAL_RATIO, seed=42)

    # Modern FastAI v2 best practices: augmentations in item_tfms, normalization in batch_tfms
    if item_tfms is None:
        item_tfms = [
            Resize(DEFAULT_IMAGE_SIZE),
        ]

    if batch_tfms is None:
        batch_tfms = [
            IntToFloatTensor(),  # MUST be first in batch_tfms for augmentations to work
            *aug_transforms(
                size=DEFAULT_IMAGE_SIZE,
                max_rotate=DEFAULT_MAX_ROTATE,
                flip_vert=DEFAULT_FLIP_VERT,
                min_zoom=DEFAULT_MIN_ZOOM,
                max_zoom=DEFAULT_MAX_ZOOM,
                max_warp=DEFAULT_MAX_WARP,
                # Enable lighting augmentation for medical imaging robustness
                max_lighting=DEFAULT_MAX_LIGHTING,
            ),
            # Use ImageNet normalization - critical for transfer learning performance
            Normalize.from_stats(*imagenet_stats),
        ]

    # Use list for blocks parameter as expected by FastAI v2
    block = DataBlock(
        blocks=[ImageBlock, MaskBlock(codes=codes)],
        get_items=get_items,
        get_y=get_y_func,
        splitter=splitter,
        item_tfms=item_tfms,
        batch_tfms=batch_tfms,
    )
    return block


def build_segmentation_dls(data_root: Union[str, Path], bs: int = 8, num_workers: int = 0):
    """
    Create DataLoaders for binary segmentation.
    
    Args:
        data_root: Path to data directory with images/ and masks/ subdirectories
        bs: Batch size
        num_workers: Number of workers for data loading
        
    Returns:
        DataLoaders configured for binary segmentation
    """
    db = build_segmentation_datablock()
    dls = db.dataloaders(Path(data_root), bs=bs, num_workers=num_workers)

    # Preflight: enforce 1:1 mapping between image_patches and mask_patches
    try:
        root = Path(data_root)
        img_dir = root / 'image_patches'
        msk_dir = root / 'mask_patches'
        if img_dir.exists() and msk_dir.exists():
            img_paths = list(get_image_files(img_dir))
            # Build expected mask paths map
            missing = []
            for p in img_paths[:10000]:  # cap to avoid extreme scans
                expected = msk_dir / f"{p.stem}_mask{p.suffix}"
                if not expected.exists():
                    # Fallback transform like get_y_standard
                    alt = Path(str(p).replace('.jpeg', '.png').replace('.jpg', '.png').replace('img_', 'mask_'))
                    if not alt.exists():
                        missing.append((p, expected))
            if missing:
                examples = "\n".join([f"- image: {im} -> expected mask: {em}" for im, em in missing[:10]])
                raise ValueError(
                    "1:1 mapping check failed: some masks are missing.\n" +
                    f"Checked {min(len(img_paths),10000)} images, missing: {len(missing)}.\n" +
                    examples
                )

            # Mask content sanity check on a small validation sample
            try:
                logger = get_logger("eq.datablock_loader")
                val_items = list(getattr(dls.valid_ds, 'items', []))
                sample_items = val_items[:64]
                checked = 0
                all_zero = 0
                some_positive = 0
                for ip in sample_items:
                    ip = Path(ip)
                    mp = msk_dir / f"{ip.stem}_mask{ip.suffix}"
                    if not mp.exists():
                        # already guarded above; skip content check
                        continue
                    m = np.array(PILMask.create(mp))
                    if m.max() == 0:
                        all_zero += 1
                    else:
                        some_positive += 1
                    checked += 1
                if checked > 0 and all_zero/checked > 0.5:
                    logger.warning(f"⚠️  High empty-mask rate in validation: {all_zero/checked:.2%} ({all_zero}/{checked})")
                else:
                    logger.info(f"✅ Mask validation: {some_positive}/{checked} masks have positive pixels ({some_positive/checked:.1%})")
            except Exception:
                # Non-fatal; continue
                pass
    except Exception as _e:
        # Surface mapping issues clearly
        raise

    return dls


