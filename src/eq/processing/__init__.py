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
from .patchify import (  # Image patchification (from patches/patchify_images.py)
    patchify_image_and_mask_dirs,
    patchify_image_dir,
)

# TODO: Add feature extraction functionality from features/ (non-data-loading parts)

__all__ = [
    # Patchification
    'patchify_image_dir',
    'patchify_image_and_mask_dirs',
    
    # File conversion
    'convert_tif_to_jpg',
]

# Version info
__version__ = "1.0.0"
__description__ = "Unified image processing and file operations"
