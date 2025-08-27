"""
Core Types Module

This module contains shared type definitions and abstract interfaces used across the endotheliosis quantifier package.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from pathlib import Path

# Type aliases for common data structures
ImageArray = np.ndarray
MaskArray = np.ndarray
ImagePath = Union[str, Path]
MaskPath = Union[str, Path]
DataLoader = Any  # Will be more specific when we know the exact type
Model = Any  # Will be more specific when we know the exact type

class DataLoaderInterface(ABC):
    """Abstract interface for data loaders."""
    
    @abstractmethod
    def load_data(self, data_path: ImagePath) -> Tuple[ImageArray, MaskArray]:
        """Load image and mask data from path."""
        pass
    
    @abstractmethod
    def get_data_loader(self, **kwargs) -> DataLoader:
        """Get a data loader instance."""
        pass

class ModelLoaderInterface(ABC):
    """Abstract interface for model loaders."""
    
    @abstractmethod
    def load_model(self, model_path: str) -> Model:
        """Load a model from path."""
        pass
    
    @abstractmethod
    def save_model(self, model: Model, model_path: str) -> None:
        """Save a model to path."""
        pass

class PreprocessorInterface(ABC):
    """Abstract interface for data preprocessors."""
    
    @abstractmethod
    def preprocess_data(self, data: ImageArray, **kwargs) -> ImageArray:
        """Preprocess input data."""
        pass
    
    @abstractmethod
    def apply_transforms(self, data: ImageArray, **kwargs) -> ImageArray:
        """Apply transformations to data."""
        pass

# Concrete type definitions for better type hints
class TrainingConfig:
    """Configuration for training parameters."""
    
    def __init__(self, 
                 epochs: int = 50,
                 batch_size: int = 8,
                 learning_rate: float = 1e-3,
                 **kwargs):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        for key, value in kwargs.items():
            setattr(self, key, value)

class InferenceConfig:
    """Configuration for inference parameters."""
    
    def __init__(self,
                 batch_size: int = 1,
                 threshold: float = 0.5,
                 **kwargs):
        self.batch_size = batch_size
        self.threshold = threshold
        for key, value in kwargs.items():
            setattr(self, key, value)

# Export the interfaces and types
__all__ = [
    'ImageArray',
    'MaskArray', 
    'ImagePath',
    'MaskPath',
    'DataLoader',
    'Model',
    'DataLoaderInterface',
    'ModelLoaderInterface',
    'PreprocessorInterface',
    'TrainingConfig',
    'InferenceConfig'
]
