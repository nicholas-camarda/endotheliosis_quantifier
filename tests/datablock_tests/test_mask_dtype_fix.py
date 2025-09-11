#!/usr/bin/env python3
"""
Test different MaskBlock configurations to fix the dtype issue.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

print("Testing MaskBlock dtype configurations...")

try:
    from fastai.data.block import DataBlock
    from fastai.data.transforms import RandomSplitter
    from fastai.vision.all import ImageBlock, MaskBlock, IntToFloatTensor, aug_transforms
    from eq.core.constants import (
        DEFAULT_VAL_RATIO, DEFAULT_IMAGE_SIZE, DEFAULT_MAX_ROTATE,
        DEFAULT_FLIP_VERT, DEFAULT_MIN_ZOOM, DEFAULT_MAX_ZOOM,
        DEFAULT_MAX_WARP, DEFAULT_MAX_LIGHTING
    )
    from eq.data_management.standard_getters import get_y
    from eq.data_management.datablock_loader import get_items_full_images
    
    data_root = "/home/ncamarda/endotheliosis_quantifier/data/derived_data/mito"
    
    # Test 1: Original configuration (problematic)
    print("\n--- Test 1: Original MaskBlock(codes=[0, 1]) ---")
    block1 = DataBlock(
        blocks=[ImageBlock, MaskBlock(codes=[0, 1])],
        get_items=get_items_full_images,
        get_y=get_y,
        splitter=RandomSplitter(valid_pct=DEFAULT_VAL_RATIO, seed=42),
        item_tfms=[],
        batch_tfms=[IntToFloatTensor()],
    )
    
    dls1 = block1.dataloaders(Path(data_root), bs=2, num_workers=0)
    batch1 = dls1.train.one_batch()
    if len(batch1) == 2:
        images1, masks1 = batch1
        print(f"Masks dtype: {masks1.dtype}")
        print(f"Masks unique values: {masks1.unique()}")
    
    # Test 2: MaskBlock without codes (should preserve float32)
    print("\n--- Test 2: MaskBlock() without codes ---")
    block2 = DataBlock(
        blocks=[ImageBlock, MaskBlock()],
        get_items=get_items_full_images,
        get_y=get_y,
        splitter=RandomSplitter(valid_pct=DEFAULT_VAL_RATIO, seed=42),
        item_tfms=[],
        batch_tfms=[IntToFloatTensor()],
    )
    
    dls2 = block2.dataloaders(Path(data_root), bs=2, num_workers=0)
    batch2 = dls2.train.one_batch()
    if len(batch2) == 2:
        images2, masks2 = batch2
        print(f"Masks dtype: {masks2.dtype}")
        print(f"Masks unique values: {masks2.unique()}")
    
    # Test 3: Custom mask block that preserves float32
    print("\n--- Test 3: Custom mask handling ---")
    from fastai.data.block import TransformBlock
    
    def get_y_float32(x):
        """Get mask and ensure it's float32."""
        mask = get_y(x)
        import numpy as np
        mask_array = np.array(mask)
        if mask_array.dtype != np.float32:
            mask_array = mask_array.astype(np.float32)
        from fastai.vision.all import PILMask
        return PILMask.create(mask_array)
    
    block3 = DataBlock(
        blocks=[ImageBlock, TransformBlock],
        get_items=get_items_full_images,
        get_y=get_y_float32,
        splitter=RandomSplitter(valid_pct=DEFAULT_VAL_RATIO, seed=42),
        item_tfms=[],
        batch_tfms=[IntToFloatTensor()],
    )
    
    dls3 = block3.dataloaders(Path(data_root), bs=2, num_workers=0)
    batch3 = dls3.train.one_batch()
    if len(batch3) == 2:
        images3, masks3 = batch3
        print(f"Masks dtype: {masks3.dtype}")
        print(f"Masks unique values: {masks3.unique()}")
    
    # Test 4: Check what happens with different codes
    print("\n--- Test 4: MaskBlock with different codes ---")
    block4 = DataBlock(
        blocks=[ImageBlock, MaskBlock(codes=[0.0, 1.0])],  # Float codes
        get_items=get_items_full_images,
        get_y=get_y,
        splitter=RandomSplitter(valid_pct=DEFAULT_VAL_RATIO, seed=42),
        item_tfms=[],
        batch_tfms=[IntToFloatTensor()],
    )
    
    dls4 = block4.dataloaders(Path(data_root), bs=2, num_workers=0)
    batch4 = dls4.train.one_batch()
    if len(batch4) == 2:
        images4, masks4 = batch4
        print(f"Masks dtype: {masks4.dtype}")
        print(f"Masks unique values: {masks4.unique()}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("Mask dtype testing complete.")
