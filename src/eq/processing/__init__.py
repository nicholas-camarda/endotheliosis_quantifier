#!/usr/bin/env python3
"""
Processing Module

This module exposes current full-image preprocessing and extraction helpers.
"""

# Import processing functionality
from .image_mask_preprocessing import extract_large_images
from .preprocessing import (  # Core preprocessing functions
    normalize_image_array,
    preprocess_image_for_model,
    resize_image_large,
    resize_image_standard,
)

# Note: annotation_processor removed - use PNG exports from Label Studio instead

__all__ = [
    # Image preprocessing
    'resize_image_standard',
    'resize_image_large',
    'preprocess_image_for_model',
    'normalize_image_array',
    
    # Full-image extraction
    'extract_large_images',
]

# Version info
__version__ = "1.0.0"
__description__ = "Current full-image preprocessing and extraction helpers"
