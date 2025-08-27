#!/usr/bin/env python3
"""
EQ Core Module

This module provides the core functionality that was previously duplicated 
across multiple files in the codebase. It serves as the single source of truth
for essential functions used throughout the endotheliosis quantifier package.

Key consolidations:
- Data loading functions (get_glom_y, get_glom_mask_file, n_glom_codes)
- Model loading with proper binary mask support
- Preprocessing functions  
- Shared constants and mappings

Uses the correct, principled binary segmentation approach for H&E glomeruli masks.

Import Usage:
    from eq.core import get_glom_y, get_glom_mask_file, BINARY_P2C
    from eq.core import load_model_with_historical_support
    from eq.core import preprocess_image_for_model
"""

# Constants
from .constants import (
                        BINARY_P2C,  # Correct binary approach [0, 1]
                        CACHE_PATTERNS,
                        DEFAULT_IMAGE_SIZE,
                        DEFAULT_MASK_THRESHOLD,  # Proven H&E threshold (127)
                        DEFAULT_P2C,  # Default: BINARY_P2C
                        DEFAULT_PREDICTION_THRESHOLD,
                        FASTAI_MODEL_EXTENSION,
                        IMAGE_EXTENSIONS,
                        LARGE_IMAGE_SIZE,  # For models needing larger input
                        MASK_EXTENSIONS,
                        MPS_FALLBACK_ENV_VAR,
)

# Data loading functions
from .data_loading import (
                        get_glom_mask_file,
                        get_glom_y,
                        get_mask_path_patterns,
                        n_glom_codes,
                        p2c,  # Legacy compatibility - now uses BINARY_P2C
                        setup_global_functions,
)

# Model loading functions
from .model_loading import (
                        get_model_info,
                        load_model_with_historical_support,
                        setup_model_loading_environment,
                        validate_model_compatibility,
)

# Preprocessing functions
from .preprocessing import (
                        normalize_image_array,
                        prepare_image_for_inference,
                        preprocess_image_for_model,
                        resize_image_large,
                        resize_image_standard,
)

# Public API
__all__ = [
    # Constants
    'BINARY_P2C',             # Correct binary approach
    'DEFAULT_P2C',            # Default: BINARY_P2C
    'DEFAULT_IMAGE_SIZE',
    'LARGE_IMAGE_SIZE',
    'DEFAULT_MASK_THRESHOLD', # Proven H&E threshold
    'DEFAULT_PREDICTION_THRESHOLD',
    'IMAGE_EXTENSIONS',
    'MASK_EXTENSIONS',
    'FASTAI_MODEL_EXTENSION',
    'MPS_FALLBACK_ENV_VAR',
    'CACHE_PATTERNS',
    
    # Data loading
    'get_glom_mask_file',
    'get_glom_y',
    'n_glom_codes',
    'get_mask_path_patterns',
    'setup_global_functions',
    'p2c',  # Legacy compatibility
    
    # Model loading
    'setup_model_loading_environment',
    'load_model_with_historical_support',
    'get_model_info',
    'validate_model_compatibility',
    
    # Preprocessing
    'resize_image_standard',
    'resize_image_large',
    'preprocess_image_for_model',
    'normalize_image_array',
    'prepare_image_for_inference',
]

# Version info
__version__ = "1.0.0"
__author__ = "EQ Development Team"
__description__ = "Core functionality consolidation for endotheliosis quantifier"
