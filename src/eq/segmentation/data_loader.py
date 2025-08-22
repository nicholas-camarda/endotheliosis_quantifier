"""
Data loading and preprocessing utilities for segmentation tasks.

This module provides utilities for loading, preprocessing, and validating
segmentation data for both mitochondria and glomeruli tasks.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    
    # Image configuration
    image_size: int = 224
    image_channels: int = 3
    
    # Augmentation configuration
    enable_augmentation: bool = True
    rotation_limit: int = 30
    scale_limit: Tuple[float, float] = (0.8, 1.2)
    brightness_limit: float = 0.2
    contrast_limit: float = 0.2
    horizontal_flip_prob: float = 0.5
    vertical_flip_prob: float = 0.5
    
    # Data validation
    validate_data: bool = True
    min_image_size: int = 64
    max_image_size: int = 2048
    
    # File extensions
    image_extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.tiff', '.tif')
    mask_extensions: Tuple[str, ...] = ('.png', '.tiff', '.tif')


class SegmentationDataLoader:
    """
    Data loader for segmentation tasks.
    
    Handles loading, preprocessing, and validation of image-mask pairs
    for both mitochondria and glomeruli segmentation.
    """
    
    def __init__(self, config: DataConfig):
        """
        Initialize the data loader.
        
        Args:
            config: Configuration for data loading and preprocessing
        """
        self.config = config
        self.transforms = self._create_transforms()
        
    def _create_transforms(self) -> Dict[str, A.Compose]:
        """Create augmentation transforms for training and validation."""
        
        # Training transforms with augmentation
        train_transforms = A.Compose([
            A.Resize(self.config.image_size, self.config.image_size),
            A.HorizontalFlip(p=self.config.horizontal_flip_prob),
            A.VerticalFlip(p=self.config.vertical_flip_prob),
            A.Rotate(limit=self.config.rotation_limit, p=0.5),
            A.RandomScale(scale_limit=self.config.scale_limit, p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=self.config.brightness_limit,
                contrast_limit=self.config.contrast_limit,
                p=0.5
            ),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # Validation transforms (no augmentation)
        val_transforms = A.Compose([
            A.Resize(self.config.image_size, self.config.image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        return {
            'train': train_transforms,
            'val': val_transforms
        }
    
    def find_data_files(self, data_path: Path, task_type: str) -> Tuple[List[Path], List[Path]]:
        """
        Find image and mask files in the data directory.
        
        Args:
            data_path: Path to the data directory
            task_type: Type of segmentation task ("mitochondria" or "glomeruli")
            
        Returns:
            Tuple of (image_files, mask_files)
        """
        # Find image files
        image_dir = data_path / "images"
        if not image_dir.exists():
            raise ValueError(f"Image directory not found: {image_dir}")
        
        image_files = []
        for ext in self.config.image_extensions:
            image_files.extend(image_dir.glob(f"*{ext}"))
            image_files.extend(image_dir.glob(f"*{ext.upper()}"))
        
        if not image_files:
            raise ValueError(f"No image files found in {image_dir}")
        
        # Find corresponding mask files
        mask_dir = data_path / "masks"
        if not mask_dir.exists():
            raise ValueError(f"Mask directory not found: {mask_dir}")
        
        mask_files = []
        valid_image_files = []
        
        for img_path in image_files:
            mask_path = self._get_mask_path(img_path, task_type)
            if mask_path.exists():
                mask_files.append(mask_path)
                valid_image_files.append(img_path)
            else:
                logger.warning(f"Mask not found for {img_path}: {mask_path}")
        
        if not mask_files:
            raise ValueError("No valid image-mask pairs found")
        
        logger.info(f"Found {len(valid_image_files)} valid image-mask pairs")
        return valid_image_files, mask_files
    
    def _get_mask_path(self, image_path: Path, task_type: str) -> Path:
        """Get corresponding mask path for an image."""
        mask_dir = image_path.parent.parent / "masks"
        
        # Try different mask naming conventions
        base_name = image_path.stem
        
        # Remove common suffixes
        for suffix in ['_img', '_image', '_orig']:
            if base_name.endswith(suffix):
                base_name = base_name[:-len(suffix)]
        
        # Try different mask extensions
        for ext in self.config.mask_extensions:
            mask_path = mask_dir / f"{base_name}_mask{ext}"
            if mask_path.exists():
                return mask_path
            
            # Try without _mask suffix
            mask_path = mask_dir / f"{base_name}{ext}"
            if mask_path.exists():
                return mask_path
        
        # If no mask found, return expected path
        return mask_dir / f"{base_name}_mask.png"
    
    def validate_image(self, image_path: Path) -> bool:
        """
        Validate an image file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            True if image is valid, False otherwise
        """
        try:
            # Check file exists and is readable
            if not image_path.exists():
                logger.warning(f"Image file does not exist: {image_path}")
                return False
            
            # Load image to check format
            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning(f"Could not load image: {image_path}")
                return False
            
            # Check image dimensions
            height, width = image.shape[:2]
            if height < self.config.min_image_size or width < self.config.min_image_size:
                logger.warning(f"Image too small ({width}x{height}): {image_path}")
                return False
            
            if height > self.config.max_image_size or width > self.config.max_image_size:
                logger.warning(f"Image too large ({width}x{height}): {image_path}")
                return False
            
            # Check number of channels
            if len(image.shape) == 3 and image.shape[2] != self.config.image_channels:
                logger.warning(f"Unexpected number of channels ({image.shape[2]}): {image_path}")
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error validating image {image_path}: {e}")
            return False
    
    def validate_mask(self, mask_path: Path) -> bool:
        """
        Validate a mask file.
        
        Args:
            mask_path: Path to the mask file
            
        Returns:
            True if mask is valid, False otherwise
        """
        try:
            # Check file exists and is readable
            if not mask_path.exists():
                logger.warning(f"Mask file does not exist: {mask_path}")
                return False
            
            # Load mask to check format
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                logger.warning(f"Could not load mask: {mask_path}")
                return False
            
            # Check mask values (should be binary or categorical)
            unique_values = np.unique(mask)
            if len(unique_values) > 10:  # Too many unique values for segmentation
                logger.warning(f"Mask has too many unique values ({len(unique_values)}): {mask_path}")
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error validating mask {mask_path}: {e}")
            return False
    
    def load_image(self, image_path: Path) -> np.ndarray:
        """
        Load an image file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Loaded image as numpy array
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image
    
    def load_mask(self, mask_path: Path) -> np.ndarray:
        """
        Load a mask file.
        
        Args:
            mask_path: Path to the mask file
            
        Returns:
            Loaded mask as numpy array
        """
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not load mask: {mask_path}")
        
        return mask
    
    def preprocess_image(self, image: np.ndarray, split: str = 'val') -> torch.Tensor:
        """
        Preprocess an image for model input.
        
        Args:
            image: Input image as numpy array
            split: Data split ('train' or 'val')
            
        Returns:
            Preprocessed image as torch tensor
        """
        transforms = self.transforms[split]
        transformed = transforms(image=image)
        return transformed['image']
    
    def preprocess_mask(self, mask: np.ndarray, num_classes: int = 2) -> torch.Tensor:
        """
        Preprocess a mask for model training.
        
        Args:
            mask: Input mask as numpy array
            num_classes: Number of classes in the mask
            
        Returns:
            Preprocessed mask as torch tensor
        """
        # Normalize mask values to [0, num_classes-1]
        if num_classes == 2:
            # Binary segmentation
            mask = (mask > 0).astype(np.uint8)
        else:
            # Multi-class segmentation
            mask = np.clip(mask, 0, num_classes - 1).astype(np.uint8)
        
        # Convert to tensor
        mask_tensor = torch.from_numpy(mask).long()
        
        return mask_tensor
    
    def create_dataset(self, data_path: Path, task_type: str, 
                      split_ratio: float = 0.8, seed: int = 42) -> Dict[str, List[Tuple[Path, Path]]]:
        """
        Create train/validation dataset splits.
        
        Args:
            data_path: Path to the data directory
            task_type: Type of segmentation task
            split_ratio: Ratio of training data (0.0 to 1.0)
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary with 'train' and 'val' splits
        """
        # Find all valid image-mask pairs
        image_files, mask_files = self.find_data_files(data_path, task_type)
        
        # Validate all pairs
        valid_pairs = []
        for img_path, mask_path in zip(image_files, mask_files):
            if self.validate_image(img_path) and self.validate_mask(mask_path):
                valid_pairs.append((img_path, mask_path))
        
        if not valid_pairs:
            raise ValueError("No valid image-mask pairs found after validation")
        
        # Shuffle and split
        np.random.seed(seed)
        np.random.shuffle(valid_pairs)
        
        split_idx = int(len(valid_pairs) * split_ratio)
        train_pairs = valid_pairs[:split_idx]
        val_pairs = valid_pairs[split_idx:]
        
        logger.info(f"Created dataset with {len(train_pairs)} training and {len(val_pairs)} validation pairs")
        
        return {
            'train': train_pairs,
            'val': val_pairs
        }
    
    def get_data_info(self, data_path: Path, task_type: str) -> Dict[str, Any]:
        """
        Get information about the dataset.
        
        Args:
            data_path: Path to the data directory
            task_type: Type of segmentation task
            
        Returns:
            Dictionary with dataset information
        """
        try:
            image_files, mask_files = self.find_data_files(data_path, task_type)
            
            # Sample some images to get statistics
            sample_images = []
            sample_masks = []
            
            for i in range(min(10, len(image_files))):
                try:
                    image = self.load_image(image_files[i])
                    mask = self.load_mask(mask_files[i])
                    sample_images.append(image)
                    sample_masks.append(mask)
                except Exception as e:
                    logger.warning(f"Error loading sample {i}: {e}")
            
            if not sample_images:
                return {"error": "Could not load any sample images"}
            
            # Calculate statistics
            image_sizes = [(img.shape[1], img.shape[0]) for img in sample_images]
            mask_sizes = [(mask.shape[1], mask.shape[0]) for mask in sample_masks]
            
            mask_values = []
            for mask in sample_masks:
                mask_values.extend(np.unique(mask).tolist())
            unique_mask_values = list(set(mask_values))
            
            return {
                'total_pairs': len(image_files),
                'sample_image_sizes': image_sizes,
                'sample_mask_sizes': mask_sizes,
                'unique_mask_values': unique_mask_values,
                'image_channels': sample_images[0].shape[2] if len(sample_images[0].shape) > 2 else 1,
                'task_type': task_type
            }
            
        except Exception as e:
            return {"error": str(e)}


def create_data_loader(config: Optional[DataConfig] = None) -> SegmentationDataLoader:
    """
    Create a data loader instance.
    
    Args:
        config: Configuration object (uses default if None)
        
    Returns:
        Configured SegmentationDataLoader
    """
    if config is None:
        config = DataConfig()
    
    return SegmentationDataLoader(config)


def validate_dataset(data_path: Path, task_type: str) -> Dict[str, Any]:
    """
    Validate a complete dataset.
    
    Args:
        data_path: Path to the data directory
        task_type: Type of segmentation task
        
    Returns:
        Validation results
    """
    loader = create_data_loader()
    
    try:
        # Get dataset info
        info = loader.get_data_info(data_path, task_type)
        
        if "error" in info:
            return {"valid": False, "error": info["error"]}
        
        # Create dataset splits
        dataset = loader.create_dataset(data_path, task_type)
        
        return {
            "valid": True,
            "info": info,
            "train_samples": len(dataset['train']),
            "val_samples": len(dataset['val']),
            "total_samples": len(dataset['train']) + len(dataset['val'])
        }
        
    except Exception as e:
        return {"valid": False, "error": str(e)}
