#!/usr/bin/env python3
"""
Unified Data Loaders

This module consolidates all data loading functionality from:
- features/data_loader.py
- features/glomeruli_data_loader.py  
- features/mitochondria_data_loader.py
- segmentation/data_loader.py

Uses the correct binary segmentation approach from eq.core.
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Import our core functionality
from eq.core.constants import BINARY_P2C, DEFAULT_MASK_THRESHOLD
from eq.data_management.data_loading import get_glom_mask_file
from eq.utils.logger import get_logger


class Annotation:
    """Annotation class for RLE mask handling (from features/data_loader.py)."""
    
    def __init__(self, image_name: str, rle_mask: Any, score: float = None):
        self.image_name = image_name
        self.rle_mask = rle_mask
        self.score = score

    def __repr__(self):
        return f"Annotation(image_path={self.image_name}, annotations={self.rle_mask}, score={self.score})"


class UnifiedDataLoader:
    """
    Unified data loader that handles both glomeruli and mitochondria data.
    
    Consolidates the functionality from the specialized loaders while using
    the correct binary segmentation approach from eq.core.
    """
    
    def __init__(self, 
                 data_type: str,  # 'glomeruli' or 'mitochondria'
                 data_dir: str,
                 cache_dir: str,
                 mask_dir: str = None,
                 annotations_file: str = None,
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.2,
                 test_ratio: float = 0.1,
                 random_seed: int = 42):
        """
        Initialize unified data loader.
        
        Args:
            data_type: Type of data ('glomeruli' or 'mitochondria')
            data_dir: Directory containing images
            cache_dir: Directory for caching processed data
            mask_dir: Directory containing masks (if separate from data_dir)
            annotations_file: Path to annotations JSON (for glomeruli)
            train_ratio: Ratio for training split
            val_ratio: Ratio for validation split  
            test_ratio: Ratio for test split
            random_seed: Random seed for reproducibility
        """
        self.data_type = data_type.lower()
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.annotations_file = annotations_file
        
        # Split ratios
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio  
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Logger
        self.logger = get_logger(f"eq.data.{self.data_type}_loader")
        
        # Validate split ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")
    
    def load_data(self, use_cache: bool = True) -> Dict[str, Any]:
        """
        Load and split data into train/val/test sets.
        
        Args:
            use_cache: Whether to use cached data if available
            
        Returns:
            Dictionary containing train/val/test data splits
        """
        cache_file = self.cache_dir / f"{self.data_type}_data_cache.pkl"
        
        # Try to load from cache
        if use_cache and cache_file.exists():
            try:
                self.logger.info(f"Loading {self.data_type} data from cache: {cache_file}")
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}. Loading fresh data.")
        
        # Load fresh data
        self.logger.info(f"Loading fresh {self.data_type} data from: {self.data_dir}")
        
        if self.data_type == 'glomeruli':
            data = self._load_glomeruli_data()
        elif self.data_type == 'mitochondria':
            data = self._load_mitochondria_data()
        else:
            raise ValueError(f"Unknown data type: {self.data_type}")
        
        # Cache the results
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            self.logger.info(f"Cached {self.data_type} data to: {cache_file}")
        except Exception as e:
            self.logger.warning(f"Failed to cache data: {e}")
        
        return data
    
    def _load_glomeruli_data(self) -> Dict[str, Any]:
        """Load glomeruli data with proper binary conversion."""
        # Find image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']:
            image_files.extend(list(self.data_dir.glob(f"**/{ext}")))
        
        if not image_files:
            raise ValueError(f"No image files found in {self.data_dir}")
        
        self.logger.info(f"Found {len(image_files)} glomeruli images")
        
        # Load and process data  
        images = []
        masks = []
        valid_files = []
        
        for img_file in image_files:
            try:
                # Load image
                image = cv2.imread(str(img_file))
                if image is None:
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Load mask using core function (applies binary conversion)
                mask = get_glom_mask_file(img_file, BINARY_P2C, DEFAULT_MASK_THRESHOLD)
                if mask is None:
                    continue
                
                # Convert to numpy for processing
                mask_array = np.array(mask)
                
                # Verify binary conversion worked
                unique_vals = np.unique(mask_array)
                if len(unique_vals) > 2:
                    self.logger.warning(f"Mask {img_file} has {len(unique_vals)} values after binary conversion: {unique_vals}")
                
                images.append(image)
                masks.append(mask_array)
                valid_files.append(str(img_file))
                
            except Exception as e:
                self.logger.warning(f"Failed to load {img_file}: {e}")
        
        self.logger.info(f"Successfully loaded {len(images)} glomeruli image-mask pairs")
        
        # Create train/val/test splits
        return self._create_splits(images, masks, valid_files)
    
    def _load_mitochondria_data(self) -> Dict[str, Any]:
        """Load mitochondria patch data."""
        # Find image patch files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']:
            image_files.extend(list(self.data_dir.glob(f"**/{ext}")))
        
        if not image_files:
            raise ValueError(f"No image files found in {self.data_dir}")
        
        self.logger.info(f"Found {len(image_files)} mitochondria patches")
        
        # Load and process data
        images = []
        masks = []
        valid_files = []
        
        for img_file in image_files:
            try:
                # Load image
                image = cv2.imread(str(img_file))
                if image is None:
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Find corresponding mask file
                mask_file = self._find_mask_file(img_file)
                if not mask_file or not mask_file.exists():
                    continue
                
                # Load and process mask with binary conversion
                mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    continue
                
                # Apply binary conversion (same as core approach)
                mask[mask <= DEFAULT_MASK_THRESHOLD] = 0  # Background
                mask[mask > DEFAULT_MASK_THRESHOLD] = 1   # Mitochondria
                
                images.append(image)
                masks.append(mask)
                valid_files.append(str(img_file))
                
            except Exception as e:
                self.logger.warning(f"Failed to load {img_file}: {e}")
        
        self.logger.info(f"Successfully loaded {len(images)} mitochondria patch-mask pairs")
        
        # Create train/val/test splits
        return self._create_splits(images, masks, valid_files)
    
    def _find_mask_file(self, image_file: Path) -> Optional[Path]:
        """Find corresponding mask file for an image."""
        if self.mask_dir:
            # Look in separate mask directory
            mask_name = image_file.stem + '_mask' + image_file.suffix
            return self.mask_dir / mask_name
        else:
            # Look for common mask patterns
            patterns = [
                image_file.parent / (image_file.stem + '_mask' + image_file.suffix),
                image_file.parent / (image_file.stem + '_mask.png'),
                image_file.parent.parent / 'masks' / image_file.name,
            ]
            
            for pattern in patterns:
                if pattern.exists():
                    return pattern
        
        return None
    
    def _create_splits(self, images: List[np.ndarray], 
                      masks: List[np.ndarray], 
                      filenames: List[str]) -> Dict[str, Any]:
        """Create train/val/test splits."""
        # Convert to numpy arrays
        images = np.array(images)
        masks = np.array(masks)
        filenames = np.array(filenames)
        
        # First split: separate test set
        if self.test_ratio > 0:
            train_val_images, test_images, train_val_masks, test_masks, train_val_files, test_files = \
                train_test_split(images, masks, filenames, 
                               test_size=self.test_ratio, 
                               random_state=self.random_seed)
        else:
            train_val_images, train_val_masks, train_val_files = images, masks, filenames
            test_images, test_masks, test_files = np.array([]), np.array([]), np.array([])
        
        # Second split: separate train and validation
        if self.val_ratio > 0:
            val_size = self.val_ratio / (self.train_ratio + self.val_ratio)
            train_images, val_images, train_masks, val_masks, train_files, val_files = \
                train_test_split(train_val_images, train_val_masks, train_val_files,
                               test_size=val_size,
                               random_state=self.random_seed)
        else:
            train_images, train_masks, train_files = train_val_images, train_val_masks, train_val_files
            val_images, val_masks, val_files = np.array([]), np.array([]), np.array([])
        
        result = {
            'train': {
                'images': train_images,
                'masks': train_masks, 
                'filenames': train_files
            },
            'val': {
                'images': val_images,
                'masks': val_masks,
                'filenames': val_files
            },
            'test': {
                'images': test_images,
                'masks': test_masks,
                'filenames': test_files
            },
            'metadata': {
                'data_type': self.data_type,
                'total_samples': len(images),
                'train_samples': len(train_images) if len(train_images.shape) > 0 else 0,
                'val_samples': len(val_images) if len(val_images.shape) > 0 else 0,
                'test_samples': len(test_images) if len(test_images.shape) > 0 else 0,
                'binary_conversion': True,
                'threshold': DEFAULT_MASK_THRESHOLD
            }
        }
        
        self.logger.info(f"Created {self.data_type} splits - Train: {result['metadata']['train_samples']}, "
                        f"Val: {result['metadata']['val_samples']}, Test: {result['metadata']['test_samples']}")
        
        return result


# Legacy compatibility functions (maintain existing API)
def load_glomeruli_data(processed_images_dir: str, cache_dir: str, **kwargs) -> Dict[str, Any]:
    """Legacy compatibility function for glomeruli data loading."""
    loader = UnifiedDataLoader('glomeruli', processed_images_dir, cache_dir, **kwargs)
    return loader.load_data()


def load_mitochondria_patches(image_patches_dir: str, cache_dir: str, mask_patches_dir: str = None, **kwargs) -> Dict[str, Any]:
    """Legacy compatibility function for mitochondria data loading."""
    loader = UnifiedDataLoader('mitochondria', image_patches_dir, cache_dir, 
                             mask_dir=mask_patches_dir, **kwargs)
    return loader.load_data()


# Annotation handling functions (from features/data_loader.py)
def load_annotations_from_json(annotations_path: str) -> List[Annotation]:
    """Load annotations from JSON file."""
    with open(annotations_path, 'r') as f:
        data = json.load(f)
    
    annotations = []
    for item in data:
        if isinstance(item, dict):
            annotation = Annotation(
                image_name=item.get('image_name', ''),
                rle_mask=item.get('rle_mask', ''), 
                score=item.get('score')
            )
            annotations.append(annotation)
    
    return annotations


def get_scores_from_annotations(annotations: List[Annotation]) -> List[float]:
    """Extract scores from annotations."""
    return [ann.score for ann in annotations if ann.score is not None]
