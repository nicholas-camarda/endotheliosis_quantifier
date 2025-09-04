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
from .image_mask_preprocessing import patchify_dataset  # Unified image patchification
from .preprocessing import (  # Core preprocessing functions
    resize_image_standard,
    resize_image_large,
    preprocess_image_for_model,
    normalize_image_array,
)
# Note: create_mitochondria_patches.py removed - functionality replaced by patchify_dataset

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
    'patchify_dataset',
    
    # Note: Mitochondria patch creation functions removed - use patchify_dataset instead
]

# Version info
__version__ = "1.0.0"
__description__ = "Unified image processing and file operations"
