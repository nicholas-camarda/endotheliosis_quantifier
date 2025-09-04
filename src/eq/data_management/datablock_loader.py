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
from fastai.vision.all import ImageBlock, MaskBlock, PILMask, get_image_files
import numpy as np
from eq.core.constants import DEFAULT_VAL_RATIO, DEFAULT_IMAGE_SIZE
from eq.data_management.standard_getters import get_y_universal


def default_get_y(image_path: Path) -> PILMask:
    """Delegate to the canonical universal getter for masks."""
    return get_y_universal(image_path)


def build_segmentation_datablock(
    codes: Optional[List[int]] = None,
    get_items: Optional[Callable[[Path], List[Any]]] = None,
    get_y: Optional[Callable[[Path], PILMask]] = None,
    splitter: Optional[Callable] = None,
):
    """
    Create a minimal DataBlock for binary segmentation.

    Args:
        codes: Label codes for MaskBlock (binary by default)
        get_items: Function to list image files (defaults to get_image_files on images/)
        get_y: Function mapping image path -> PILMask
        splitter: Train/valid splitter (defaults to 80/20 RandomSplitter)
    """
    if codes is None:
        codes = [0, 1]
    
    if get_items is None:
        def _get_items(path: Path) -> List[Any]:
            # Try both possible directory structures
            images_dir = Path(path) / "images"
            if not images_dir.exists():
                images_dir = Path(path) / "image_patches"
            return list(get_image_files(images_dir))
        get_items = _get_items

    if get_y is None:
        get_y = default_get_y

    if splitter is None:
        splitter = RandomSplitter(valid_pct=DEFAULT_VAL_RATIO, seed=42)

    # Use list for blocks parameter as expected by FastAI v2
    block = DataBlock(
        blocks=[ImageBlock, MaskBlock(codes=codes)],
        get_items=get_items,
        get_y=get_y,
        splitter=splitter,
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
    return dls


