#!/usr/bin/env python3
"""
EQ Core Module

This module provides core constants, types, and abstract interfaces for the endotheliosis quantifier package.
After reorganization, core only contains:
- Constants and configurations
- Type definitions and abstract interfaces
- Shared type hints

Import Usage:
    from eq.core.constants import BINARY_P2C, DEFAULT_MASK_THRESHOLD
    from eq.core.types import DataLoaderInterface, ModelLoaderInterface
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

# Types and interfaces
from .types import (
                        DataLoaderInterface,
                        ModelLoaderInterface,
                        PreprocessorInterface,
                        TrainingConfig,
                        InferenceConfig,
                        ImageArray,
                        MaskArray,
                        ImagePath,
                        MaskPath,
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
    
    # Types and interfaces
    'DataLoaderInterface',
    'ModelLoaderInterface',
    'PreprocessorInterface',
    'TrainingConfig',
    'InferenceConfig',
    'ImageArray',
    'MaskArray',
    'ImagePath',
    'MaskPath',
]

__description__ = "Core constants, types, and interfaces for endotheliosis quantifier"
