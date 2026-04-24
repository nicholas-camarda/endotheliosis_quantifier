#!/usr/bin/env python3
"""
EQ Core Module

This module provides core constants for the endotheliosis quantifier package.

Import Usage:
    from eq.core.constants import BINARY_P2C, DEFAULT_MASK_THRESHOLD
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
]

__description__ = "Core constants for endotheliosis quantifier"
