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

# === Image Processing Constants ===

# Default patch sizes (what we extract from input images)
DEFAULT_PATCH_SIZE = 256  # Standard patch size for training/inference
LEGACY_PATCH_SIZE = 224   # Legacy size for backward compatibility
LARGE_PATCH_SIZE = 512    # For high-resolution analysis

# Default image sizes (what we resize to for model input)
DEFAULT_IMAGE_SIZE = 256  # Standard model input size
LEGACY_IMAGE_SIZE = 224   # Legacy model input size
LARGE_IMAGE_SIZE = 512    # For models needing larger input

# Default overlap and ratios
DEFAULT_PATCH_OVERLAP = 0.1
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_VAL_RATIO = 0.2
DEFAULT_TEST_RATIO = 0.0

# Data split constants
DEFAULT_TRAIN_SPLIT = 0.7
DEFAULT_VAL_SPLIT = 0.2
DEFAULT_TEST_SPLIT = 0.1

# === Data Processing Constants ===

# Expected input image dimensions (based on actual data)
EXPECTED_INPUT_WIDTH = 2448   # Typical width of input images
EXPECTED_INPUT_HEIGHT = 2048  # Typical height of input images

# Patch calculation constants
PATCHES_PER_ROW = EXPECTED_INPUT_WIDTH // DEFAULT_PATCH_SIZE      # 2448 // 256 = 9
PATCHES_PER_COL = EXPECTED_INPUT_HEIGHT // DEFAULT_PATCH_SIZE     # 2048 // 256 = 8
EXPECTED_PATCHES_PER_IMAGE = PATCHES_PER_ROW * PATCHES_PER_COL   # 9 * 8 = 72 patches

# === Model Constants ===
DEFAULT_PREDICTION_THRESHOLD = 0.5

# Mask processing thresholds
DEFAULT_MASK_THRESHOLD = 127  # Your proven threshold for H&E mask conversion

# === Training Constants ===
DEFAULT_BATCH_SIZE = 8  # From pipeline config
DEFAULT_EPOCHS = 50  # From pipeline config
DEFAULT_LEARNING_RATE = 1e-3

# === Data Augmentation Constants ===
DEFAULT_FLIP_VERT = True
DEFAULT_MAX_ROTATE = 45
DEFAULT_MIN_ZOOM = 0.8
DEFAULT_MAX_ZOOM = 1.3
DEFAULT_MAX_WARP = 0.4
DEFAULT_MAX_LIGHTING = 0.2
DEFAULT_RANDOM_ERASING_P = 0.5
DEFAULT_RANDOM_ERASING_SL = 0.01
DEFAULT_RANDOM_ERASING_SH = 0.3
DEFAULT_RANDOM_ERASING_MIN_ASPECT = 0.3
DEFAULT_RANDOM_ERASING_MAX_COUNT = 3

# File extensions
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
MASK_EXTENSIONS = ['.png', '.jpg', '.tif', '.tiff']

# Model loading constants
FASTAI_MODEL_EXTENSION = '.pkl'

# === Output Directory Constants ===
# Standardized output directory structure
DEFAULT_MODELS_DIR = "models"
DEFAULT_SEGMENTATION_DIR = "models/segmentation"
DEFAULT_MITOCHONDRIA_MODEL_DIR = "models/segmentation/mitochondria"
DEFAULT_GLOMERULI_MODEL_DIR = "models/segmentation/glomeruli"
DEFAULT_OUTPUT_DIR = "output"  # For general outputs (plots, logs, etc.)
DEFAULT_RESULTS_DIR = "results"  # For analysis results

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
