#!/usr/bin/env python3
"""
Standard Getter Functions for FastAI v2

This module provides standardized getter functions that can be used across
all training modules to ensure compatibility during model loading and transfer learning.
"""

from pathlib import Path
import numpy as np
from fastai.vision.all import PILMask
from eq.core.constants import DEFAULT_MASK_THRESHOLD
from eq.utils.logger import get_logger


def get_y(x):
    """
    Universal get_y function that works for all segmentation tasks.
    
    Handles multiple mask location patterns:
    1. Derived data structure: image_patches/ -> mask_patches/
    2. Standard structure: img_*.jpg -> mask_*.png
    
    Args:
        x: Image file path (Path object)
        
    Returns:
        PILMask: Corresponding mask as float32 for FastAI v2 compatibility
    """
    logger = get_logger("eq.mask_loading")
    
    # Strategy 1: Try derived data structure (mask_patches directory)
    mask_path = x.parent.parent / "mask_patches" / f"{x.stem}_mask{x.suffix}"
    
    if mask_path.exists():
        msk = np.array(PILMask.create(mask_path))
        original_max = msk.max()
        original_unique = np.unique(msk)
        
        # Handle both binary (0/1) and grayscale (0-255) masks
        # Convert to float32 for FastAI v2 augmentation compatibility
        if msk.max() <= 1:
            # Already binary, convert to float32
            msk = msk.astype(np.float32)
            logger.debug(f"Binary mask {mask_path.name}: max={original_max}, unique={original_unique} -> converted to float32")
        else:
            # Grayscale mask, apply threshold and convert to float32
            msk = (msk > DEFAULT_MASK_THRESHOLD).astype(np.float32)
            positive_pixels = (msk > 0).sum()
            logger.debug(f"Grayscale mask {mask_path.name}: max={original_max}, unique={len(original_unique)} values -> thresholded at {DEFAULT_MASK_THRESHOLD}, {positive_pixels} positive pixels, converted to float32")
        
        final_positive = (msk > 0).sum()
        if final_positive == 0:
            logger.debug(f"All-zero mask after processing: {mask_path.name}")
        else:
            logger.debug(f"Mask {mask_path.name}: {final_positive} positive pixels")
        
        return PILMask.create(msk)
    
    # Strategy 2: Try standard path replacement (img_ -> mask_, .jpg -> .png)
    standard_mask_path = str(x).replace('.jpg', '.png').replace('.jpeg', '.png').replace('img_', 'mask_')
    if Path(standard_mask_path).exists():
        msk = np.array(PILMask.create(standard_mask_path))
        original_max = msk.max()
        original_unique = np.unique(msk)
        
        # Handle both binary (0/1) and grayscale (0-255) masks  
        # Convert to float32 for FastAI v2 augmentation compatibility
        if msk.max() <= 1:
            # Already binary, convert to float32
            msk = msk.astype(np.float32)
            logger.debug(f"Binary mask {Path(standard_mask_path).name}: max={original_max}, unique={original_unique} -> converted to float32")
        else:
            # Grayscale mask, apply threshold and convert to float32
            msk = (msk > DEFAULT_MASK_THRESHOLD).astype(np.float32)
            positive_pixels = (msk > 0).sum()
            logger.debug(f"Grayscale mask {Path(standard_mask_path).name}: max={original_max}, unique={len(original_unique)} values -> thresholded at {DEFAULT_MASK_THRESHOLD}, {positive_pixels} positive pixels, converted to float32")
        
        final_positive = (msk > 0).sum()
        if final_positive == 0:
            logger.warning(f"⚠️  All-zero mask after processing: {Path(standard_mask_path).name}")
        else:
            logger.debug(f"✅ Mask {Path(standard_mask_path).name}: {final_positive} positive pixels")
        
        return PILMask.create(msk)
    
    # Strategy 3: No mask found - this should not happen if get_items filters correctly
    # This indicates a data integrity issue that needs to be fixed
    raise FileNotFoundError(f"❌ CRITICAL: No mask found for {x.name} - this should not happen if get_items filtering is working correctly. Check data integrity.")


# Legacy function aliases for backward compatibility
def get_y_universal(x):
    """Legacy alias for get_y function."""
    return get_y(x)

def get_y_mitochondria(x):
    """Legacy alias for get_y function.""" 
    return get_y(x)

def get_y_glomeruli(x):
    """Legacy alias for get_y function."""
    return get_y(x)
