#!/usr/bin/env python3
"""
Core Constants Module

This module contains all shared constants used across the endotheliosis quantifier package.
Uses the standard, principled binary segmentation approach.
"""

# STANDARD P2C (pixel-to-class) mapping for binary segmentation
# This is the correct, principled approach for glomeruli segmentation
BINARY_P2C = [0, 1]  # 0 = background, 1 = glomeruli

# Default P2C mapping - always use binary
DEFAULT_P2C = BINARY_P2C

# Default image sizes
DEFAULT_IMAGE_SIZE = 224
LARGE_IMAGE_SIZE = 512  # For models that need larger input

# Mask processing thresholds
DEFAULT_MASK_THRESHOLD = 127  # Your proven threshold for H&E mask conversion
DEFAULT_PREDICTION_THRESHOLD = 0.5

# File extensions
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
MASK_EXTENSIONS = ['.png', '.jpg', '.tif', '.tiff']

# Model loading constants
FASTAI_MODEL_EXTENSION = '.pkl'

# Environment variables
MPS_FALLBACK_ENV_VAR = 'PYTORCH_ENABLE_MPS_FALLBACK'

# Cache file patterns
CACHE_PATTERNS = {
    'train_images': 'train_images.pickle',
    'train_masks': 'train_masks.pickle',
    'val_images': 'val_images.pickle',
    'val_masks': 'val_masks.pickle',
    'test_images': 'test_images.pickle',
}
