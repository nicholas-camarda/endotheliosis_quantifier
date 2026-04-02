#!/usr/bin/env python3
"""
Data Configuration

Simple configuration classes for data loading and preprocessing.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    
    # Image configuration
    image_size: int = 224
    image_channels: int = 3
    
    # Data splits
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    test_ratio: float = 0.1
    
    # Caching
    use_cache: bool = True
    cache_dir: str = "cache"
    
    # Binary conversion
    mask_threshold: int = 127
    
    # Random seed
    random_seed: int = 42


@dataclass  
class AugmentationConfig:
    """Configuration for data augmentation."""
    
    # Basic augmentations
    horizontal_flip: bool = True
    vertical_flip: bool = False
    rotation_range: int = 15
    
    # Color augmentations
    brightness_range: Tuple[float, float] = (0.8, 1.2)
    contrast_range: Tuple[float, float] = (0.8, 1.2)
    
    # Noise and blur
    gaussian_noise: bool = False
    gaussian_blur: bool = False
    
    # Probability of applying augmentations
    augmentation_probability: float = 0.5
