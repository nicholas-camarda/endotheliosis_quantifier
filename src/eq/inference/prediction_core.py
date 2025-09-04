#!/usr/bin/env python3
"""
Consolidated Prediction Core Module

This module provides unified prediction functionality that eliminates duplication
across different inference and evaluation modules. All prediction logic should
use this module instead of implementing their own versions.

Features:
- Unified image preprocessing
- Standardized prediction workflow
- Consistent tensor handling
- Reusable prediction utilities
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import torch
from PIL import Image

from eq.utils.logger import get_logger
from eq.core.constants import DEFAULT_IMAGE_SIZE


class PredictionCore:
    """
    Core prediction functionality for all segmentation models.
    
    This class consolidates all the duplicate prediction logic that was
    scattered across different modules.
    """

    def __init__(self, expected_size: int = DEFAULT_IMAGE_SIZE):
        """
        Initialize prediction core.
        
        Args:
            expected_size: Expected input size for the model
        """
        self.expected_size = expected_size
        self.logger = get_logger("eq.prediction_core")
    
    def preprocess_image(self, image: Union[Image.Image, np.ndarray]) -> torch.Tensor:
        """
        Preprocess image for model input.
        
        Args:
            image: PIL Image or numpy array to preprocess
            
        Returns:
            Preprocessed tensor ready for model input
        """
        # Convert numpy array to PIL if needed
        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[-1] == 1:
                image = np.repeat(image, 3, axis=-1)
            elif image.ndim == 2:
                image = np.repeat(image[..., None], 3, axis=-1)
            
            # Convert to PIL Image
            image = Image.fromarray((image * 255).astype(np.uint8))
        
        # Ensure image is PIL Image
        if not isinstance(image, Image.Image):
            raise ValueError(f"Expected PIL Image or numpy array, got {type(image)}")
        
        # Resize to expected input size
        img_resized = image.resize(
            (self.expected_size, self.expected_size),
            Image.Resampling.BILINEAR,
        )
        
        # Convert to tensor and normalize
        img_tensor = torch.from_numpy(np.array(img_resized)).float() / 255.0
        
        # Convert to channels-first format (B, C, H, W)
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
        
        return img_tensor
    
    def predict_with_model(self, model: torch.nn.Module, image: Union[Image.Image, np.ndarray],
                          threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Make prediction using a PyTorch model.
        
        Args:
            model: PyTorch model to use for prediction
            image: Input image (PIL Image or numpy array)
            threshold: Threshold for binary prediction
            
        Returns:
            Tuple of (raw_output, probabilities, binary_prediction)
        """
        # Preprocess image
        img_tensor = self.preprocess_image(image)
        
        # Make prediction
        with torch.no_grad():
            raw_output = model(img_tensor)
        
        # Apply softmax to get probabilities
        if raw_output.shape[1] == 2:  # Binary segmentation
            probs = torch.softmax(raw_output, dim=1)
            pred_mask = probs[:, 1]  # Take foreground class
        else:
            pred_mask = torch.sigmoid(raw_output)
        
        # Create binary prediction
        pred_binary = (pred_mask > threshold).float()
        
        return raw_output, pred_mask, pred_binary
    
    def predict_with_fastai_learner(self, learn, image: Union[Image.Image, np.ndarray],
                                   threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Make prediction using a FastAI learner.
        
        Args:
            learn: FastAI learner object
            image: Input image (PIL Image or numpy array)
            threshold: Threshold for binary prediction
            
        Returns:
            Tuple of (raw_output, probabilities, binary_prediction)
        """
        # Use FastAI's predict method
        pred_result: Any = learn.predict(image)
        
        # Extract prediction tensor from fastai tuple
        if isinstance(pred_result, tuple) and len(pred_result) >= 2:
            pred_tensor_any: Any = pred_result[1]
        else:
            pred_tensor_any = pred_result
        
        # Convert to tensor if needed
        pred_tensor: torch.Tensor
        if isinstance(pred_tensor_any, torch.Tensor):
            pred_tensor = pred_tensor_any
        elif hasattr(pred_tensor_any, 'cpu'):
            pred_tensor = cast(Any, pred_tensor_any).cpu()
            if not isinstance(pred_tensor, torch.Tensor):
                pred_tensor = torch.from_numpy(np.asarray(pred_tensor))
        elif hasattr(pred_tensor_any, 'numpy'):
            pred_tensor = torch.from_numpy(cast(Any, pred_tensor_any).numpy())
        else:
            pred_tensor = torch.from_numpy(np.asarray(pred_tensor_any))
        
        # Apply threshold to get binary prediction
        pred_binary = (pred_tensor > threshold).float()
        
        # For compatibility, return similar structure
        return pred_tensor, pred_tensor, pred_binary
    
    def extract_prediction_from_result(self, pred_result: Union[torch.Tensor, Tuple[Any, ...], np.ndarray]) -> torch.Tensor:
        """
        Extract prediction tensor from various result formats.
        
        Args:
            pred_result: Prediction result (tensor, tuple, or numpy array)
            
        Returns:
            Extracted prediction tensor
        """
        if isinstance(pred_result, torch.Tensor):
            return pred_result
        
        elif isinstance(pred_result, tuple) and len(pred_result) >= 2:
            # FastAI format: (prediction, target, loss)
            return self._to_tensor(pred_result[1])
        
        elif isinstance(pred_result, np.ndarray):
            return torch.from_numpy(pred_result)
        
        else:
            # Try to convert
            try:
                return self._to_tensor(pred_result)  # type: ignore[arg-type]
            except Exception as e:
                raise ValueError(f"Cannot extract prediction from {type(pred_result)}: {e}")

    def _to_tensor(self, obj: Any) -> torch.Tensor:
        """Best-effort conversion to torch.Tensor."""
        if isinstance(obj, torch.Tensor):
            return obj
        if hasattr(obj, 'cpu'):
            maybe = cast(Any, obj).cpu()
            if isinstance(maybe, torch.Tensor):
                return maybe
        if hasattr(obj, 'numpy'):
            try:
                return torch.from_numpy(cast(Any, obj).numpy())
            except Exception:
                pass
        return torch.from_numpy(np.asarray(obj))
    
    def convert_tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Convert tensor to numpy array safely.
        
        Args:
            tensor: PyTorch tensor
            
        Returns:
            Numpy array
        """
        if hasattr(tensor, 'numpy'):
            return tensor.numpy()
        elif hasattr(tensor, 'cpu'):
            return tensor.cpu().numpy()
        else:
            return np.asarray(tensor)
    
    def resize_prediction_to_match(self, pred_mask: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """
        Resize prediction mask to match target shape.
        
        Args:
            pred_mask: Prediction mask to resize
            target_shape: Target shape
            
        Returns:
            Resized prediction mask
        """
        if pred_mask.shape == target_shape:
            return pred_mask
        
        try:
            from scipy.ndimage import zoom
            
            # Calculate scale factors
            scale_factors = [target_shape[j] / pred_mask.shape[j] for j in range(len(target_shape))]
            pred_mask = zoom(pred_mask, scale_factors, order=1)
            
            return pred_mask
        except ImportError:
            self.logger.warning("scipy not available, using basic resize")
            # Fallback to basic resize
            from PIL import Image
            pred_img = Image.fromarray(pred_mask)
            # Ensure size is a 2-tuple (width, height)
            size_wh = (int(target_shape[1]), int(target_shape[0]))
            target_img = pred_img.resize(size_wh, Image.Resampling.NEAREST)  # PIL uses (W, H)
            return np.array(target_img)
    
    def ensure_binary_mask(self, mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Ensure mask is binary.
        
        Args:
            mask: Input mask
            threshold: Threshold for binarization
            
        Returns:
            Binary mask
        """
        return (mask > threshold).astype(np.float32)
    
    def prepare_image_for_display(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """
        Prepare image for display.
        
        Args:
            image: Input image
            
        Returns:
            Image array ready for display
        """
        if isinstance(image, Image.Image):
            return np.array(image)
        elif isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[-1] == 1:
                return image.squeeze()
            elif image.ndim == 2:
                return image
            else:
                return image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")


def create_prediction_core(expected_size: int = DEFAULT_IMAGE_SIZE) -> PredictionCore:
    """
    Factory function to create a prediction core instance.
    
    Args:
        expected_size: Expected input size for the model
        
    Returns:
        PredictionCore instance
    """
    return PredictionCore(expected_size)


# Convenience functions for common operations
def preprocess_image_for_prediction(image: Union[Image.Image, np.ndarray], 
                                  expected_size: int = DEFAULT_IMAGE_SIZE) -> torch.Tensor:
    """
    Convenience function to preprocess image for prediction.
    
    Args:
        image: Input image
        expected_size: Expected input size
        
    Returns:
        Preprocessed tensor
    """
    core = create_prediction_core(expected_size)
    return core.preprocess_image(image)


def extract_prediction_tensor(pred_result: Union[torch.Tensor, Tuple, np.ndarray]) -> torch.Tensor:
    """
    Convenience function to extract prediction tensor.
    
    Args:
        pred_result: Prediction result
        
    Returns:
        Extracted prediction tensor
    """
    core = create_prediction_core()
    return core.extract_prediction_from_result(pred_result)


def convert_prediction_to_numpy(pred_tensor: torch.Tensor) -> np.ndarray:
    """
    Convenience function to convert prediction tensor to numpy.
    
    Args:
        pred_tensor: Prediction tensor
        
    Returns:
        Numpy array
    """
    core = create_prediction_core()
    return core.convert_tensor_to_numpy(pred_tensor)



