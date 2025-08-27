#!/usr/bin/env python3
"""
Core Data Loading Functions

This module consolidates all the data loading functions that were previously 
duplicated across multiple files in the codebase. Uses the correct, principled
binary segmentation approach for H&E glomeruli masks.

Functions previously found in:
- __main__.py
- segmentation/fastai_segmenter.py  
- utils/model_loader.py
- pipeline/extract_model_weights.py
- pipeline/retrain_glomeruli_original.py
- inference/run_glomeruli_prediction.py
# Note: historical_glomeruli_inference.py has been removed during consolidation
"""

from pathlib import Path
from typing import List, Optional, Union

import numpy as np

# Import FastAI components (with fallback handling)
try:
    from fastai.vision.all import PILMask
except ImportError:
    # Fallback for when FastAI is not available
    PILMask = None

from eq.core.constants import BINARY_P2C, DEFAULT_MASK_THRESHOLD


def get_glom_mask_file(image_file: Union[str, Path], 
                      p2c: list = None, 
                      thresh: int = DEFAULT_MASK_THRESHOLD) -> Optional[object]:
    """
    Get mask file for a given image file using proper binary conversion.
    
    This function converts colored H&E mask annotations to clean binary masks
    using the proven threshold approach (thresh=127).
    
    Args:
        image_file: Path to the image file
        p2c: Pixel-to-class mapping (defaults to BINARY_P2C [0, 1])
        thresh: Threshold for binary conversion (default: 127 - proven for H&E)
        
    Returns:
        PILMask object or None if mask not found
    """
    if p2c is None:
        p2c = BINARY_P2C
        
    if PILMask is None:
        raise ImportError("FastAI not available. Please install fastai to use this function.")
    
    try:
        image_path_str = str(image_file)
        
        # Adapt to current data structure patterns
        if 'image_patches' in image_path_str:
            # Current data structure uses .jpg for both images and masks
            mask_path = Path(image_path_str.replace('image_patches', 'mask_patches'))
            # Keep .jpg extension for masks
        elif '/data/images/' in image_path_str:
            mask_path = Path(image_path_str.replace('/data/images/', '/data/masks/').replace('.jpg', '_mask.jpg'))
        else:
            # Try generic patterns - try .jpg first, then .png
            mask_path = Path(str(image_file).replace('.jpg', '_mask.jpg'))
            if not mask_path.exists():
                mask_path = Path(str(image_file).replace('.jpg', '_mask.png'))
        
        if not mask_path.exists():
            return None
        
        # CRITICAL: Convert colored H&E annotations to clean binary masks
        # This is your proven approach for handling colored mask annotations
        msk = np.array(PILMask.create(mask_path))
        
        # Apply threshold to convert ANY colored annotation to binary
        msk[msk <= thresh] = 0  # Background
        msk[msk > thresh] = 1   # Glomeruli
        
        # For binary [0,1], this step does nothing, but maintains compatibility
        if isinstance(p2c, list) and len(p2c) == 2:
            # Standard binary case - already correct
            pass
        else:
            # Should not happen with binary approach, but handle gracefully
            for i, val in enumerate(p2c):
                if i < len(p2c):
                    msk[msk == i] = val
        
        return PILMask.create(msk)
        
    except Exception:
        return None


def get_glom_y(image_file: Union[str, Path], p2c: list = None) -> Optional[object]:
    """
    Get glomeruli mask for a given image file.
    
    This is the canonical version required for FastAI model loading.
    Uses proper binary conversion for H&E mask annotations.
    
    Args:
        image_file: Path to the image file
        p2c: Pixel-to-class mapping (defaults to BINARY_P2C [0, 1])
        
    Returns:
        PILMask object or None if mask not found
    """
    if p2c is None:
        p2c = BINARY_P2C
    return get_glom_mask_file(image_file, p2c)


def n_glom_codes(mask_files: List[Union[str, Path]]) -> List[int]:
    """
    Get unique codes from mask files AFTER proper binary conversion.
    
    This function applies the same binary thresholding as get_glom_mask_file()
    to ensure consistency between training and inference.
    
    Args:
        mask_files: List of mask file paths
        
    Returns:
        Sorted list of unique codes found in the masks (should be [0, 1])
    """
    if PILMask is None:
        raise ImportError("FastAI not available. Please install fastai to use this function.")
    
    codes = set()
    for mask_file in mask_files:
        try:
            # Apply the same binary conversion as get_glom_mask_file
            mask = np.array(PILMask.create(mask_file))
            mask[mask <= DEFAULT_MASK_THRESHOLD] = 0  # Background
            mask[mask > DEFAULT_MASK_THRESHOLD] = 1   # Glomeruli
            codes.update(np.unique(mask))
        except Exception:
            continue  # Skip files that can't be loaded
    
    result = sorted(list(codes))
    
    # Sanity check: should only be [0, 1] for binary segmentation
    if len(result) > 2:
        print(f"⚠️  Warning: Found {len(result)} codes after binary conversion: {result}")
        print("   Expected: [0, 1] for binary segmentation")
    
    return result


def get_mask_path_patterns(image_path: Union[str, Path]) -> List[Path]:
    """
    Get possible mask paths for a given image path.
    
    This function tries multiple common patterns to find mask files.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        List of possible mask paths to try
    """
    image_path = Path(image_path)
    possible_paths = []
    
    # Pattern 1: image_patches -> mask_patches
    if 'image_patches' in str(image_path):
        mask_path = Path(str(image_path).replace('image_patches', 'mask_patches'))
        possible_paths.append(mask_path.with_suffix('.png'))
        possible_paths.append(mask_path)
    
    # Pattern 2: /data/images/ -> /data/masks/
    if '/data/images/' in str(image_path):
        mask_path = Path(str(image_path).replace('/data/images/', '/data/masks/'))
        possible_paths.append(mask_path.with_name(mask_path.stem + '_mask' + mask_path.suffix))
        possible_paths.append(mask_path.with_suffix('_mask.jpg'))
    
    # Pattern 3: Generic _mask suffix
    possible_paths.append(image_path.with_name(image_path.stem + '_mask.png'))
    possible_paths.append(image_path.with_name(image_path.stem + '_mask.jpg'))
    
    # Pattern 4: Same directory, different extension
    possible_paths.append(image_path.with_suffix('.png'))
    
    return possible_paths


def setup_global_functions(p2c: list = None):
    """
    Set up global functions required for FastAI model loading.
    
    This function injects the required functions into the global namespace
    so that FastAI can find them when loading pickled models.
    
    Args:
        p2c: Pixel-to-class mapping (defaults to BINARY_P2C [0, 1])
    """
    import __main__
    
    if p2c is None:
        p2c = BINARY_P2C
    
    # Make p2c available globally
    globals()['p2c'] = p2c
    
    # Make functions available globally  
    globals()['get_glom_y'] = get_glom_y
    globals()['get_glom_mask_file'] = get_glom_mask_file
    globals()['n_glom_codes'] = n_glom_codes
    
    # Also inject into main module namespace for pickle loading
    if hasattr(__main__, '__dict__'):
        __main__.__dict__['p2c'] = p2c
        __main__.__dict__['get_y'] = get_glom_y  # Legacy name for compatibility
        __main__.__dict__['get_glom_y'] = get_glom_y
        __main__.__dict__['get_glom_mask_file'] = get_glom_mask_file
        __main__.__dict__['n_glom_codes'] = n_glom_codes


# Legacy compatibility - use binary approach
p2c = BINARY_P2C
