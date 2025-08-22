#!/usr/bin/env python3
"""
Mitochondria Data Loader

This module handles loading mitochondria patches for U-Net training.
It creates train/val splits and loads the patches with their corresponding masks.
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

from eq.utils.logger import get_logger


class MitochondriaDataLoader:
    """Loads mitochondria patches for U-Net training."""
    
    def __init__(self, 
                 image_patches_dir: str,
                 mask_patches_dir: str,
                 cache_dir: str,
                 train_ratio: float = 0.8,
                 val_ratio: float = 0.2,
                 random_seed: int = 42):
        """
        Initialize the mitochondria data loader.
        
        Args:
            image_patches_dir: Directory containing image patches
            mask_patches_dir: Directory containing mask patches
            cache_dir: Directory to cache processed data
            train_ratio: Ratio of data for training
            val_ratio: Ratio of data for validation
            random_seed: Random seed for reproducibility
        """
        self.logger = get_logger("eq.mitochondria_data_loader")
        self.image_patches_dir = Path(image_patches_dir)
        self.mask_patches_dir = Path(mask_patches_dir)
        self.cache_dir = Path(cache_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.random_seed = random_seed
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache file paths
        self.cache_file = self.cache_dir / "mitochondria_patches_cache.pkl"
        
    def _get_patch_files(self) -> Tuple[List[str], List[str]]:
        """
        Get lists of image and mask patch files.
        
        Returns:
            Tuple of (image_files, mask_files) lists
        """
        # Get all image patch files (support both .tif and .jpg)
        image_files = []
        for ext in ["*.tif", "*.jpg"]:
            image_files.extend([f.name for f in self.image_patches_dir.glob(ext)])
        image_files = sorted(image_files)
        
        # Get corresponding mask files
        mask_files = []
        for img_file in image_files:
            # For patch files, the mask filename is the same as the image filename
            # e.g., training_60_patch_3.jpg -> training_60_patch_3.jpg
            mask_file = img_file
            
            mask_path = self.mask_patches_dir / mask_file
            if mask_path.exists():
                mask_files.append(mask_file)
            else:
                self.logger.warning(f"Mask file not found for {img_file}: {mask_file}")
                mask_files.append(None)
        
        # Filter out None values
        valid_pairs = [(img, mask) for img, mask in zip(image_files, mask_files) if mask is not None]
        if valid_pairs:
            image_files, mask_files = zip(*valid_pairs)
        else:
            image_files, mask_files = [], []
            
        self.logger.info(f"Found {len(image_files)} valid image-mask pairs")
        return list(image_files), list(mask_files)
    
    def _load_patch(self, file_path: Path) -> np.ndarray:
        """
        Load a single patch file.
        
        Args:
            file_path: Path to the patch file
            
        Returns:
            Loaded image as numpy array
        """
        img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image: {file_path}")
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Add channel dimension for U-Net
        img = np.expand_dims(img, axis=-1)
        
        return img
    
    def _load_patch_pair(self, image_file: str, mask_file: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load an image-mask patch pair.
        
        Args:
            image_file: Image patch filename
            mask_file: Mask patch filename
            
        Returns:
            Tuple of (image, mask) as numpy arrays
        """
        image_path = self.image_patches_dir / image_file
        mask_path = self.mask_patches_dir / mask_file
        
        # Load image and mask
        image = self._load_patch(image_path)
        mask = self._load_patch(mask_path)
        
        # Convert mask to binary (0 or 1)
        mask = (mask > 0.5).astype(np.float32)
        
        return image, mask
    
    def _create_train_val_split(self, image_files: List[str], mask_files: List[str]) -> Dict[str, Any]:
        """
        Create train/validation split of the data.
        
        Args:
            image_files: List of image filenames
            mask_files: List of mask filenames
            
        Returns:
            Dictionary with train/val splits
        """
        # Create train/val split
        train_img, val_img, train_mask, val_mask = train_test_split(
            image_files, mask_files,
            test_size=self.val_ratio,
            random_state=self.random_seed,
            shuffle=True
        )
        
        split_info = {
            'train': {
                'image_files': train_img,
                'mask_files': train_mask,
                'count': len(train_img)
            },
            'val': {
                'image_files': val_img,
                'mask_files': val_mask,
                'count': len(val_img)
            },
            'total': len(image_files),
            'train_ratio': self.train_ratio,
            'val_ratio': self.val_ratio,
            'random_seed': self.random_seed
        }
        
        self.logger.info(f"Created train/val split: {split_info['train']['count']} train, {split_info['val']['count']} val")
        return split_info
    
    def _load_split_data(self, split_info: Dict[str, Any], split_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data for a specific split (train or val).
        
        Args:
            split_info: Split information dictionary
            split_name: Name of split ('train' or 'val')
            
        Returns:
            Tuple of (images, masks) as numpy arrays
        """
        split_data = split_info[split_name]
        image_files = split_data['image_files']
        mask_files = split_data['mask_files']
        
        self.logger.info(f"Loading {split_name} data: {len(image_files)} patches")
        
        images = []
        masks = []
        
        for img_file, mask_file in zip(image_files, mask_files):
            try:
                image, mask = self._load_patch_pair(img_file, mask_file)
                images.append(image)
                masks.append(mask)
            except Exception as e:
                self.logger.warning(f"Failed to load {img_file}: {e}")
                continue
        
        # Convert to numpy arrays
        images = np.array(images)
        masks = np.array(masks)
        
        self.logger.info(f"Loaded {split_name} data: {images.shape}, {masks.shape}")
        return images, masks
    
    def load_data(self, use_cache: bool = True) -> Dict[str, Any]:
        """
        Load mitochondria patch data with train/val split.
        
        Args:
            use_cache: Whether to use cached data if available
            
        Returns:
            Dictionary containing train/val data and metadata
        """
        # Check if we're in quick test mode
        import os
        is_quick_test = os.getenv('QUICK_TEST', 'false').lower() == 'true'
        
        # Try to load from cache first (but not in quick test mode)
        if use_cache and not is_quick_test and self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                self.logger.info("Loaded data from cache")
                return cached_data
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}")
        
        # Load data from scratch
        if is_quick_test:
            self.logger.info("QUICK_TEST mode: Loading limited data from scratch (ignoring cache)")
        else:
            self.logger.info("Loading mitochondria patch data from scratch")
        
        # Get patch files
        image_files, mask_files = self._get_patch_files()
        
        if not image_files:
            raise ValueError("No valid patch files found")
        
        # Check if we're in quick test mode and limit samples BEFORE loading
        import os
        is_quick_test = os.getenv('QUICK_TEST', 'false').lower() == 'true'
        
        if is_quick_test:
            # Use only first 20 samples for quick testing
            max_samples = 20
            if len(image_files) > max_samples:
                image_files = image_files[:max_samples]
                mask_files = mask_files[:max_samples]
                print(f"🔬 QUICK_TEST: Limited to {max_samples} samples (from {len(image_files) + len(mask_files)})")
                self.logger.info(f"QUICK_TEST mode: Limited to {max_samples} samples")
        
        # Create train/val split
        split_info = self._create_train_val_split(image_files, mask_files)
        
        # Load train data
        train_images, train_masks = self._load_split_data(split_info, 'train')
        
        # Load validation data
        val_images, val_masks = self._load_split_data(split_info, 'val')
        
        # Prepare data dictionary
        data = {
            'train': {
                'images': train_images,
                'masks': train_masks
            },
            'val': {
                'images': val_images,
                'masks': val_masks
            },
            'split_info': split_info,
            'patch_size': train_images.shape[1:3] if len(train_images) > 0 else (224, 224),
            'num_classes': 2
        }
        
        # Cache the data
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(data, f)
            self.logger.info("Cached data for future use")
        except Exception as e:
            self.logger.warning(f"Failed to cache data: {e}")
        
        return data
    
    def get_data_info(self) -> Dict[str, Any]:
        """
        Get information about the available data without loading it.
        
        Returns:
            Dictionary with data information
        """
        image_files, mask_files = self._get_patch_files()
        
        return {
            'total_patches': len(image_files),
            'image_patches_dir': str(self.image_patches_dir),
            'mask_patches_dir': str(self.mask_patches_dir),
            'patch_size': 256,
            'num_classes': 2,
            'sample_image_files': image_files[:5] if image_files else [],
            'sample_mask_files': mask_files[:5] if mask_files else []
        }


def load_mitochondria_patches(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to load mitochondria patches using configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Loaded data dictionary
    """
    data_config = config.get('data', {})
    processed_config = data_config.get('processed', {})
    
    loader = MitochondriaDataLoader(
        image_patches_dir=processed_config.get('train_dir'),
        mask_patches_dir=processed_config.get('train_mask_dir'),
        cache_dir=processed_config.get('cache_dir'),
        train_ratio=processed_config.get('train_ratio', 0.8),
        val_ratio=processed_config.get('val_ratio', 0.2),
        random_seed=processed_config.get('random_seed', 42)
    )
    
    return loader.load_data()
