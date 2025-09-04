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


def get_y_standard(x):
    """
    Standard get_y function for segmentation tasks.
    
    Converts image path to mask path by:
    1. Replacing image extensions with .png
    2. Replacing 'img_' with 'mask_' in filename
    
    Args:
        x: Image file path
        
    Returns:
        str: Corresponding mask file path
    """
    return str(x).replace('.jpg', '.png').replace('.jpeg', '.png').replace('img_', 'mask_')


def get_y_mitochondria(x):
    """
    Mitochondria-specific get_y function.
    
    Args:
        x: Image file path
        
    Returns:
        str: Corresponding mask file path
    """
    return get_y_standard(x)


def get_y_glomeruli(x):
    """
    Glomeruli-specific get_y function.
    
    For derived data structure: image_patches/ -> mask_patches/
    
    Args:
        x: Image file path
        
    Returns:
        PILMask: Corresponding mask
    """
    # Convert image path to mask path
    mask_path = x.parent.parent / "mask_patches" / f"{x.stem}_mask{x.suffix}"
    
    # Load and process mask
    if mask_path.exists():
        msk = np.array(PILMask.create(mask_path))
        # Ensure binary mask (0/1)
        msk = (msk > DEFAULT_MASK_THRESHOLD).astype(np.uint8)
        return PILMask.create(msk)
    else:
        # Return zero mask if no mask found
        return PILMask.create(np.zeros((256, 256), dtype=np.uint8))


def get_y_universal(x):
    """
    Universal get_y function that works for both mitochondria and glomeruli.
    
    Tries to find mask in the standard derived data structure.
    
    Args:
        x: Image file path
        
    Returns:
        PILMask: Corresponding mask
    """
    # Try glomeruli-style mask first (mask_patches directory)
    mask_path = x.parent.parent / "mask_patches" / f"{x.stem}_mask{x.suffix}"
    
    if mask_path.exists():
        msk = np.array(PILMask.create(mask_path))
        msk = (msk > DEFAULT_MASK_THRESHOLD).astype(np.uint8)
        return PILMask.create(msk)
    
    # Fallback to standard path replacement
    standard_mask_path = get_y_standard(x)
    if Path(standard_mask_path).exists():
        msk = np.array(PILMask.create(standard_mask_path))
        msk = (msk > DEFAULT_MASK_THRESHOLD).astype(np.uint8)
        return PILMask.create(msk)
    
    # Return zero mask if no mask found
    return PILMask.create(np.zeros((256, 256), dtype=np.uint8))
