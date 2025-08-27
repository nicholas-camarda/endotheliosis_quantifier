#!/usr/bin/env python3
"""
Processing Module

This module consolidates image processing functionality that was previously
scattered across multiple folders:
- patches/ (image patchification)
- io/ (file conversion and I/O operations)
- features/ (feature extraction and processing, minus data loading)

Provides unified image processing, patchification, and file conversion tools.
"""

# Import processing functionality
from .convert_files import convert_tif_to_jpg  # File conversion (from io/convert_files_to_jpg.py)
from .image_mask_preprocessing import (  # Image patchification (from patches/patchify_images.py)
    patchify_image_and_mask_dirs,
    patchify_image_dir,
)
from .preprocessing import (  # Core preprocessing functions
    resize_image_standard,
    resize_image_large,
    preprocess_image_for_model,
    normalize_image_array,
)
from .create_mitochondria_patches import (  # Mitochondria-specific processing
    create_patches_from_image,
    create_mitochondria_patches,
)

# TODO: Add feature extraction functionality from features/ (non-data-loading parts)

__all__ = [
    # File conversion
    'convert_tif_to_jpg',
    
    # Image preprocessing
    'resize_image_standard',
    'resize_image_large',
    'preprocess_image_for_model',
    'normalize_image_array',
    
    # Image patchification
    'patchify_image_dir',
    'patchify_image_and_mask_dirs',
    
    # Mitochondria patch creation
    'create_patches_from_image',
    'create_mitochondria_patches',
]

# Version info
__version__ = "1.0.0"
__description__ = "Unified image processing and file operations"
