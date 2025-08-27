#!/usr/bin/env python3
"""
Core Preprocessing Functions

This module contains shared preprocessing functions used across the package.
Consolidates preprocessing logic that was previously duplicated in multiple files.
Uses standard, principled approaches for binary glomeruli segmentation.
"""

from pathlib import Path
from typing import Tuple, Union

import numpy as np

# Import PIL/FastAI components (with fallback handling)
try:
    from fastai.vision.all import PILImage
    from PIL import Image
except ImportError:
    Image = None
    PILImage = None

from .constants import DEFAULT_IMAGE_SIZE, LARGE_IMAGE_SIZE


def resize_image_standard(image: Union[str, Path, object], 
                         target_size: int = DEFAULT_IMAGE_SIZE) -> object:
    """
    Resize image using standard approach (224px).
    
    This function implements the standard preprocessing approach used
    throughout your codebase.
    
    Args:
        image: PIL Image, PILImage, or path to image
        target_size: Target size (default: 224px for standard compatibility)
        
    Returns:
        Resized image object
    """
    if PILImage is None:
        raise ImportError("FastAI not available. Please install fastai to use this function.")
    
    # Handle different input types
    if isinstance(image, (str, Path)):
        img = PILImage.create(image)
    elif hasattr(image, 'resize'):
        img = image
    else:
        img = PILImage.create(image)
    
    # Resize to target size
    resized_img = img.resize((target_size, target_size))
    return resized_img


def resize_image_large(image: Union[str, Path, object], 
                      target_size: int = LARGE_IMAGE_SIZE) -> object:
    """
    Resize image using large size approach (512px).
    
    Use this for models that require larger input images.
    
    Args:
        image: PIL Image, PILImage, or path to image
        target_size: Target size (default: 512px)
        
    Returns:
        Resized image object
    """
    return resize_image_standard(image, target_size)


def preprocess_image_for_model(image_path: Union[str, Path], 
                              use_large_size: bool = False,
                              target_size: int = None) -> object:
    """
    Preprocess image for model input using specified approach.
    
    Args:
        image_path: Path to the image file
        use_large_size: Whether to use large size (512px) or standard (224px)
        target_size: Override target size (if None, uses approach default)
        
    Returns:
        Preprocessed image ready for model input
    """
    if target_size is None:
        target_size = LARGE_IMAGE_SIZE if use_large_size else DEFAULT_IMAGE_SIZE
    
    return resize_image_standard(image_path, target_size)


def normalize_image_array(image_array: np.ndarray, 
                         method: str = 'zero_one') -> np.ndarray:
    """
    Normalize image array.
    
    Args:
        image_array: Input image array
        method: Normalization method ('zero_one', 'mean_std')
        
    Returns:
        Normalized image array
    """
    if method == 'zero_one':
        # Normalize to [0, 1]
        return image_array.astype(np.float32) / 255.0
    elif method == 'mean_std':
        # Normalize to mean=0, std=1
        mean = np.mean(image_array)
        std = np.std(image_array)
        return (image_array - mean) / (std + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def prepare_image_for_inference(image_path: Union[str, Path],
                               use_large_preprocessing: bool = False) -> Tuple[object, dict]:
    """
    Prepare image for inference with metadata.
    
    Args:
        image_path: Path to the image file
        use_large_preprocessing: Whether to use large size preprocessing
        
    Returns:
        Tuple of (processed_image, metadata)
    """
    # Load original image to get metadata
    if PILImage is None:
        raise ImportError("FastAI not available. Please install fastai to use this function.")
    
    original_img = PILImage.create(image_path)
    
    # Preprocess image
    processed_img = preprocess_image_for_model(
        image_path, 
        use_large_size=use_large_preprocessing
    )
    
    # Create metadata
    metadata = {
        'original_size': np.array(original_img).shape,
        'processed_size': np.array(processed_img).shape,
        'preprocessing_approach': 'large_512px' if use_large_preprocessing else 'standard_224px',
        'target_size': LARGE_IMAGE_SIZE if use_large_preprocessing else DEFAULT_IMAGE_SIZE,
        'image_path': str(image_path)
    }
    
    return processed_img, metadata
