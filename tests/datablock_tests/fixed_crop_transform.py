#!/usr/bin/env python3
"""
Fixed CropTransform implementation for synchronized image-mask cropping.

This addresses the issues in the original CropTransform:
1. Better handling of PIL vs tensor formats
2. More robust size checking
3. Proper synchronization between image and mask crops
"""

import numpy as np
from fastai.data.transforms import Transform
from fastai.vision.all import PILImage, PILMask, Resize
from eq.core.constants import DEFAULT_IMAGE_SIZE

class FixedCropTransform(Transform):
    """
    Improved synchronized crop transform for image-mask pairs.
    
    Key improvements:
    1. Better PIL vs tensor detection
    2. More robust size handling
    3. Proper synchronization
    4. Better error handling
    """
    def __init__(self, size: int = DEFAULT_IMAGE_SIZE):
        self.size = size
    
    def encodes(self, x: tuple):
        """Apply synchronized crop to image-mask pair."""
        if not isinstance(x, (tuple, list)) or len(x) != 2:
            return x
        
        img, mask = x
        
        # Handle None values
        if img is None or mask is None:
            return x
        
        # Get dimensions - handle both PIL and tensor formats
        if hasattr(img, 'size'):  # PIL Image
            h, w = img.size[1], img.size[0]  # PIL: (width, height)
            is_pil = True
        elif hasattr(img, 'shape'):  # Tensor
            if len(img.shape) == 3:  # (C, H, W)
                h, w = img.shape[-2:]
            elif len(img.shape) == 2:  # (H, W)
                h, w = img.shape
            else:
                print(f"⚠️  Unexpected tensor shape: {img.shape}")
                return x
            is_pil = False
        else:
            print(f"⚠️  Unexpected image type: {type(img)}")
            return x
        
        # Ensure we have enough space to crop
        if h < self.size or w < self.size:
            # If image is smaller than crop size, resize first
            if is_pil:
                img = Resize(self.size)(img)
                mask = Resize(self.size)(mask)
            else:
                # For tensors, we'd need to implement tensor resize
                # For now, just return the original
                print(f"⚠️  Image too small for crop ({h}x{w} < {self.size}x{self.size}), returning original")
                return x
            h, w = self.size, self.size
        
        # Generate random crop coordinates
        top = np.random.randint(0, h - self.size + 1)
        left = np.random.randint(0, w - self.size + 1)
        
        # Apply crop based on format
        if is_pil:
            # PIL Images
            img_cropped = img.crop((left, top, left + self.size, top + self.size))
            mask_cropped = mask.crop((left, top, left + self.size, top + self.size))
        else:
            # Tensors
            if len(img.shape) == 3:  # (C, H, W)
                img_cropped = img[:, top:top+self.size, left:left+self.size]
                mask_cropped = mask[:, top:top+self.size, left:left+self.size]
            else:  # (H, W)
                img_cropped = img[top:top+self.size, left:left+self.size]
                mask_cropped = mask[top:top+self.size, left:left+self.size]
        
        return img_cropped, mask_cropped

def create_fixed_datablock():
    """Create a DataBlock using the fixed CropTransform."""
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
    
    # Create DataBlock with fixed CropTransform
    block = DataBlock(
        blocks=[ImageBlock, MaskBlock(codes=[0, 1])],
        get_items=get_items_full_images,
        get_y=get_y,
        splitter=RandomSplitter(valid_pct=DEFAULT_VAL_RATIO, seed=42),
        item_tfms=[FixedCropTransform(size=DEFAULT_IMAGE_SIZE)],
        batch_tfms=[
            IntToFloatTensor(),
            *aug_transforms(
                size=DEFAULT_IMAGE_SIZE,
                max_rotate=int(DEFAULT_MAX_ROTATE),
                flip_vert=DEFAULT_FLIP_VERT,
                min_zoom=DEFAULT_MIN_ZOOM,
                max_zoom=DEFAULT_MAX_ZOOM,
                max_warp=DEFAULT_MAX_WARP,
                max_lighting=DEFAULT_MAX_LIGHTING,
            ),
            # Note: No ImageNet normalization for medical images
        ],
    )
    return block

if __name__ == "__main__":
    print("Fixed CropTransform implementation created.")
    print("Use create_fixed_datablock() to create a DataBlock with the fixed transform.")
