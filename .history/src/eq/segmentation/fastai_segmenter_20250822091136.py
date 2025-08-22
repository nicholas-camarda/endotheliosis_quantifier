"""
FastAI-based segmentation module for endotheliosis quantification.

This module provides a unified interface for training and inference using fastai
for both mitochondria and glomeruli segmentation tasks. It supports dual-environment
architecture with MPS (Apple Silicon) and CUDA (NVIDIA) backends.
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from fastai import *
from fastai.vision.all import *
from fastai.vision.augment import *

from eq.utils.hardware_detection import get_device_recommendation, get_optimal_batch_size


# Functions needed for loading pre-trained models
def n_glom_codes(mask_files):
    """Get unique codes from mask files."""
    codes = set()
    for mask_file in mask_files:
        mask = np.array(PILMask.create(mask_file))
        codes.update(np.unique(mask))
    return sorted(list(codes))


def get_glom_mask_file(image_file, p2c, thresh=127):
    """Get mask file path for a given image file."""
    # this is the base path
    base_path = image_file.parent.parent.parent
    first_name = image_file.parent.name
    # get training or testing from here
    full_name = re.findall(string=image_file.name, pattern=r"^[A-Za-z]*[0-9]+[_|-]+[A-Za-z]*[0-9]+")[0]
    
    # put the whole thing together
    str_name = f'{full_name}_mask' + image_file.suffix
    # attach it to the correct path
    mask_path = (base_path / 'masks' / first_name / str_name)
    
    # convert to an array (mask)
    msk = np.array(PILMask.create(mask_path))
    # convert the image to binary if it isn't already (tends to happen when working with .jpg files)
    msk[msk <= thresh] = 0
    msk[msk > thresh] = 1
    
    # find all the possible values in the mask (0,255)
    for i, val in enumerate(p2c):
        msk[msk == p2c[i]] = val
    return PILMask.create(msk)


def get_glom_y(o):
    """Get glomeruli mask for a given image file."""
    # This is a placeholder - p2c should be defined when this function is used
    # For now, we'll use a default value
    p2c = [0, 1]  # Default binary mask codes
    return get_glom_mask_file(o, p2c)


@dataclass
class SegmentationConfig:
    """Configuration for segmentation training and inference."""
    
    # Data configuration
    image_size: int = 224
    batch_size: int = 16
    valid_pct: float = 0.2
    seed: int = 42
    
    # Model configuration
    model_arch: str = "resnet34"
    pretrained: bool = True
    
    # Training configuration
    learning_rate: float = 1e-3
    epochs: int = 10
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.001
    
    # Augmentation configuration
    min_scale: float = 0.3
    max_rotate: float = 30
    min_zoom: float = 0.8
    max_zoom: float = 1.15
    max_warp: float = 0.3
    flip_vert: bool = True
    
    # Hardware configuration
    device_mode: str = "auto"  # "auto", "development", "production"
    
    # Output configuration
    model_save_path: Optional[Path] = None
    results_save_path: Optional[Path] = None


class FastaiSegmenter:
    """
    FastAI-based segmentation model for endotheliosis quantification.
    
    Supports both mitochondria and glomeruli segmentation with dual-environment
    architecture (MPS/CUDA) and comprehensive data augmentation.
    """
    
    def __init__(self, config: SegmentationConfig):
        """
        Initialize the FastAI segmenter.
        
        Args:
            config: Configuration object for training and inference
        """
        self.config = config
        self.device = self._setup_device()
        self.learn = None
        self.dls = None
        
    def _setup_device(self) -> torch.device:
        """Setup device based on configuration and hardware availability."""
        if self.config.device_mode == "production":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        elif self.config.device_mode == "development":
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        else:  # auto
            return get_device_recommendation()
    
    def _preprocess_image(self, image: PILImage) -> torch.Tensor:
        """
        Preprocess image for model input.
        
        Args:
            image: PILImage to preprocess
            
        Returns:
            Preprocessed tensor
        """
        # Resize to model input size
        image = image.resize((self.config.image_size, self.config.image_size))
        
        # Convert to tensor and normalize
        image_tensor = torch.from_numpy(np.array(image)).float()
        
        # Add channel dimension if needed
        if len(image_tensor.shape) == 2:
            image_tensor = image_tensor.unsqueeze(0)
        elif len(image_tensor.shape) == 3:
            # Convert RGB to grayscale if needed
            if image_tensor.shape[2] == 3:
                image_tensor = image_tensor.mean(dim=2, keepdim=True)
            # Move channel dimension to front
            image_tensor = image_tensor.permute(2, 0, 1)
        
        # Normalize to [0, 1]
        image_tensor = image_tensor / 255.0
        
        return image_tensor.to(self.device)
    
    def _get_optimal_batch_size(self) -> int:
        """Get optimal batch size based on hardware capabilities."""
        if self.config.batch_size == 0:  # Auto-detect
            return get_optimal_batch_size(self.config.device_mode)
        return self.config.batch_size
    
    def _create_data_block(self, image_files: List[Path], mask_files: List[Path], 
                          task_type: str) -> DataBlock:
        """
        Create DataBlock for segmentation task.
        
        Args:
            image_files: List of image file paths
            mask_files: List of mask file paths
            task_type: Type of segmentation task ("mitochondria" or "glomeruli")
            
        Returns:
            Configured DataBlock for the task
        """
        # Define class codes based on task type
        if task_type == "mitochondria":
            codes = np.array(['not_mito', 'mito'])
            get_y_func = self._get_mitochondria_mask
        elif task_type == "glomeruli":
            codes = np.array(['not_glom', 'glom'])
            get_y_func = self._get_glomeruli_mask
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
        # Create augmentation transforms
        item_tfms = [RandomResizedCrop(self.config.image_size, min_scale=self.config.min_scale)]
        
        batch_tfms = [
            *aug_transforms(
                size=self.config.image_size,
                flip_vert=self.config.flip_vert,
                max_rotate=self.config.max_rotate,
                min_zoom=self.config.min_zoom,
                max_zoom=self.config.max_zoom,
                max_warp=self.config.max_warp
            )
        ]
        
        # Create DataBlock
        data_block = DataBlock(
            blocks=(ImageBlock, MaskBlock(codes=codes)),
            splitter=RandomSplitter(valid_pct=self.config.valid_pct, seed=self.config.seed),
            get_items=lambda x: image_files,
            get_y=get_y_func,
            item_tfms=item_tfms,
            batch_tfms=batch_tfms,
            n_inp=1
        )
        
        return data_block
    
    def _get_mitochondria_mask(self, image_path: Path) -> Path:
        """Get corresponding mask path for mitochondria segmentation."""
        # Convert image path to mask path
        mask_path = image_path.parent.parent / "masks" / image_path.name.replace(".jpg", "_mask.png")
        return mask_path
    
    def _get_glomeruli_mask(self, image_path: Path) -> Path:
        """Get corresponding mask path for glomeruli segmentation."""
        # Convert image path to mask path
        mask_path = image_path.parent.parent / "masks" / image_path.name.replace(".jpg", "_mask.png")
        return mask_path
    
    def prepare_data(self, data_path: Path, task_type: str) -> None:
        """
        Prepare data for training.
        
        Args:
            data_path: Path to the data directory
            task_type: Type of segmentation task ("mitochondria" or "glomeruli")
        """
        # Find image files
        image_dir = data_path / "images"
        image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
        
        if not image_files:
            raise ValueError(f"No image files found in {image_dir}")
        
        # Find corresponding mask files
        mask_dir = data_path / "masks"
        mask_files = []
        for img_path in image_files:
            if task_type == "mitochondria":
                mask_path = self._get_mitochondria_mask(img_path)
            else:
                mask_path = self._get_glomeruli_mask(img_path)
            
            if mask_path.exists():
                mask_files.append(mask_path)
            else:
                print(f"Warning: Mask not found for {img_path}")
        
        if not mask_files:
            raise ValueError(f"No mask files found in {mask_dir}")
        
        # Create DataBlock
        data_block = self._create_data_block(image_files, mask_files, task_type)
        
        # Create DataLoaders
        batch_size = self._get_optimal_batch_size()
        self.dls = data_block.dataloaders(image_files, bs=batch_size)
        
        print(f"Prepared {len(image_files)} images and {len(mask_files)} masks")
        print(f"Batch size: {batch_size}")
        print(f"Training samples: {len(self.dls.train_ds)}")
        print(f"Validation samples: {len(self.dls.valid_ds)}")
    
    def prepare_data_from_cache(self, cache_dir: Path, task_type: str) -> None:
        """
        Prepare data from cached pickle files.
        
        Args:
            cache_dir: Directory containing cached data files
            task_type: Type of segmentation task ("mitochondria" or "glomeruli")
        """
        from eq.utils.common import load_pickled_data

        # Load cached data
        train_images = load_pickled_data(cache_dir / "train_images.pickle")
        train_masks = load_pickled_data(cache_dir / "train_masks.pickle")
        val_images = load_pickled_data(cache_dir / "val_images.pickle")
        val_masks = load_pickled_data(cache_dir / "val_masks.pickle")
        
        print("Loaded cached data:")
        print(f"  Training images: {train_images.shape}")
        print(f"  Training masks: {train_masks.shape}")
        print(f"  Validation images: {val_images.shape}")
        print(f"  Validation masks: {val_masks.shape}")
        
        # Convert numpy arrays to temporary image files for fastai
        # This is a workaround since fastai expects file paths
        # In a production system, you might want to implement a custom data loader
        temp_dir = Path("/tmp/fastai_temp_data")
        temp_dir.mkdir(exist_ok=True)
        
        # Save training images and masks
        train_image_files = []
        for i, (img, mask) in enumerate(zip(train_images, train_masks)):
            img_path = temp_dir / f"train_img_{i}.jpg"
            mask_path = temp_dir / f"train_img_{i}_mask.png"  # Fixed naming
            
            # Save image
            import cv2
            if img.ndim == 3 and img.shape[2] == 1:
                img_3ch = np.repeat(img, 3, axis=2)
            else:
                img_3ch = img
            cv2.imwrite(str(img_path), (img_3ch * 255).astype(np.uint8))
            
            # Save mask
            cv2.imwrite(str(mask_path), (mask * 255).astype(np.uint8))
            
            train_image_files.append(img_path)
        
        # Save validation images and masks
        val_image_files = []
        for i, (img, mask) in enumerate(zip(val_images, val_masks)):
            img_path = temp_dir / f"val_img_{i}.jpg"
            mask_path = temp_dir / f"val_img_{i}_mask.png"  # Fixed naming
            
            # Save image
            import cv2
            if img.ndim == 3 and img.shape[2] == 1:
                img_3ch = np.repeat(img, 3, axis=2)
            else:
                img_3ch = img
            cv2.imwrite(str(img_path), (img_3ch * 255).astype(np.uint8))
            
            # Save mask
            cv2.imwrite(str(mask_path), (mask * 255).astype(np.uint8))
            
            val_image_files.append(img_path)
        
        # Create DataBlock with custom splitter
        # DataBlock is already imported from fastai.vision.all
        
        def custom_splitter(items):
            """Custom splitter that uses our predefined train/val split."""
            # Convert to indices based on the combined list
            all_items = train_image_files + val_image_files
            train_indices = list(range(len(train_image_files)))
            val_indices = list(range(len(train_image_files), len(all_items)))
            return (train_indices, val_indices)
        
        data_block = DataBlock(
            blocks=(ImageBlock, MaskBlock(codes=np.array(['not_glom', 'glom']))),
            splitter=custom_splitter,
            get_items=lambda x: x,
            get_y=lambda x: str(x).replace('.jpg', '_mask.png'),
            item_tfms=[RandomResizedCrop(self.config.image_size, min_scale=self.config.min_scale)],
            batch_tfms=self._get_advanced_augmentations()
        )
        
        # Create DataLoaders
        batch_size = self._get_optimal_batch_size()
        self.dls = data_block.dataloaders(train_image_files + val_image_files, bs=batch_size)
        
        print(f"Prepared data from cache with {len(train_image_files)} training and {len(val_image_files)} validation samples")
        print(f"Batch size: {batch_size}")
        print(f"Training samples: {len(self.dls.train_ds)}")
        print(f"Validation samples: {len(self.dls.valid_ds)}")
    
    def _get_advanced_augmentations(self):
        """Get advanced augmentation transforms for training."""
        return [
            *aug_transforms(
                size=self.config.image_size,
                flip_vert=self.config.flip_vert,
                max_rotate=self.config.max_rotate,
                min_zoom=self.config.min_zoom,
                max_zoom=self.config.max_zoom,
                max_warp=self.config.max_warp
            ),
            # Add additional augmentations that are available in fastai
            Brightness(max_lighting=0.2, p=0.5),
            Contrast(max_lighting=0.2, p=0.5),
            RandomErasing(p=0.1, max_count=2)
        ]
    
    def create_model(self, task_type: str) -> None:
        """
        Create the segmentation model.
        
        Args:
            task_type: Type of segmentation task ("mitochondria" or "glomeruli")
        """
        if self.dls is None:
            raise ValueError("Data must be prepared before creating model. Call prepare_data() first.")
        
        # Create U-Net learner with different architecture options
        if self.config.model_arch == "resnet34":
            arch = resnet34
        elif self.config.model_arch == "resnet50":
            arch = resnet50
        elif self.config.model_arch == "resnet18":
            arch = resnet18
        elif self.config.model_arch == "resnet101":
            arch = resnet101
        elif self.config.model_arch == "custom_unet":
            # Use custom U-Net architecture
            arch = self._create_custom_unet_architecture()
        else:
            raise ValueError(f"Unsupported architecture: {self.config.model_arch}")
        
        # Create learner with Dice metric
        self.learn = unet_learner(
            self.dls, 
            arch, 
            metrics=Dice,
            opt_func=Adam,
            pretrained=self.config.pretrained
        )
        
        # Move to device
        self.learn.to(self.device)
        
        print(f"Created {self.config.model_arch} U-Net model for {task_type} segmentation")
        print(f"Model parameters: {sum(p.numel() for p in self.learn.model.parameters()):,}")
    
    def _create_custom_unet_architecture(self):
        """
        Create a custom U-Net architecture with configurable parameters.
        
        Returns:
            Custom U-Net model architecture
        """
        # This is a placeholder for custom U-Net implementation
        # In a full implementation, you would define the U-Net architecture here
        # For now, we'll fall back to resnet34
        print("Custom U-Net architecture not yet implemented, using ResNet34 as backbone")
        return resnet34
    
    def find_learning_rate(self, start_lr: float = 1e-7, end_lr: float = 10, 
                          num_iter: int = 100) -> Tuple[float, float, float, float]:
        """
        Find optimal learning rate.
        
        Args:
            start_lr: Starting learning rate
            end_lr: Ending learning rate
            num_iter: Number of iterations
            
        Returns:
            Tuple of (lr_min, lr_steep, lr_valley, lr_slide)
        """
        if self.learn is None:
            raise ValueError("Model must be created before finding learning rate. Call create_model() first.")
        
        print("Finding optimal learning rate...")
        lr_min, lr_steep, lr_valley, lr_slide = self.learn.lr_find(
            suggest_funcs=(minimum, steep, valley, slide),
            start_lr=start_lr,
            end_lr=end_lr,
            num_iter=num_iter
        )
        
        print("Learning rate suggestions:")
        print(f"  Minimum: {lr_min:.2e}")
        print(f"  Steep: {lr_steep:.2e}")
        print(f"  Valley: {lr_valley:.2e}")
        print(f"  Slide: {lr_slide:.2e}")
        
        return lr_min, lr_steep, lr_valley, lr_slide
    
    def train(self, epochs: Optional[int] = None, 
              learning_rate: Optional[float] = None) -> Dict[str, Any]:
        """
        Train the segmentation model.
        
        Args:
            epochs: Number of training epochs (uses config if None)
            learning_rate: Learning rate (uses config if None)
            
        Returns:
            Training history and metrics
        """
        if self.learn is None:
            raise ValueError("Model must be created before training. Call create_model() first.")
        
        epochs = epochs or self.config.epochs
        learning_rate = learning_rate or self.config.learning_rate
        
        print(f"Training for {epochs} epochs with learning rate {learning_rate:.2e}")
        print(f"Device: {self.device}")
        print(f"Mode: {self.config.device_mode}")
        
        # Setup callbacks based on mode
        callbacks = self._get_mode_specific_callbacks()
        
        # Training strategy based on mode
        if self.config.device_mode == "production":
            # Production mode: use conservative training with early stopping
            print("Production mode: Using conservative training strategy")
            training_result = self._train_production_mode(epochs, learning_rate, callbacks)
        elif self.config.device_mode == "development":
            # Development mode: use aggressive training with learning rate finding
            print("Development mode: Using aggressive training strategy")
            training_result = self._train_development_mode(epochs, learning_rate, callbacks)
        else:
            # Auto mode: use balanced training
            print("Auto mode: Using balanced training strategy")
            training_result = self._train_auto_mode(epochs, learning_rate, callbacks)
        
        # Get final metrics
        final_metrics = self.learn.recorder.values[-1]
        
        print("Training completed. Final metrics:")
        print(f"  Training loss: {final_metrics[0]:.4f}")
        print(f"  Validation loss: {final_metrics[1]:.4f}")
        print(f"  Dice score: {final_metrics[2]:.4f}")
        
        return training_result
    
    def _get_mode_specific_callbacks(self) -> List:
        """Get callbacks based on the current mode."""
        callbacks = [
            EarlyStoppingCallback(
                monitor='valid_loss',
                min_delta=self.config.early_stopping_min_delta,
                patience=self.config.early_stopping_patience
            )
        ]
        
        if self.config.device_mode == "production":
            # Add production-specific callbacks
            callbacks.extend([
                ReduceLROnPlateau(monitor='valid_loss', patience=3, factor=0.5),
                SaveModelCallback(monitor='valid_loss', every_epoch=False)  # Save best only
            ])
        elif self.config.device_mode == "development":
            # Add development-specific callbacks
            callbacks.extend([
                ReduceLROnPlateau(monitor='valid_loss', patience=2, factor=0.7),
                SaveModelCallback(monitor='valid_loss', every_epoch=True)
            ])
        
        return callbacks
    
    def _train_production_mode(self, epochs: int, learning_rate: float, callbacks: List) -> Dict[str, Any]:
        """Train in production mode with conservative settings."""
        # Use fine_tune with conservative settings
        self.learn.fine_tune(
            epochs, 
            learning_rate,
            cbs=callbacks,
            freeze_epochs=2  # Freeze early layers for 2 epochs
        )
        
        return self._extract_training_results()
    
    def _train_development_mode(self, epochs: int, learning_rate: float, callbacks: List) -> Dict[str, Any]:
        """Train in development mode with aggressive settings."""
        # Use fit_one_cycle for aggressive training
        self.learn.fit_one_cycle(
            epochs,
            learning_rate,
            cbs=callbacks
        )
        
        return self._extract_training_results()
    
    def _train_auto_mode(self, epochs: int, learning_rate: float, callbacks: List) -> Dict[str, Any]:
        """Train in auto mode with balanced settings."""
        # Use fine_tune with balanced settings
        self.learn.fine_tune(
            epochs, 
            learning_rate,
            cbs=callbacks
        )
        
        return self._extract_training_results()
    
    def _extract_training_results(self) -> Dict[str, Any]:
        """Extract training results from the learner."""
        final_metrics = self.learn.recorder.values[-1]
        
        return {
            'training_loss': final_metrics[0],
            'validation_loss': final_metrics[1],
            'dice_score': final_metrics[2],
            'history': self.learn.recorder.values,
            'learning_rates': self.learn.recorder.lrs if hasattr(self.learn.recorder, 'lrs') else [],
            'device_used': str(self.device),
            'mode_used': self.config.device_mode
        }
    
    def predict(self, image_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Predict segmentation masks for a single image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of prediction dictionaries with masks and confidence scores
        """
        if self.learn is None:
            raise ValueError("No model loaded. Call load_model() first.")
        
        # Load and preprocess image
        image = PILImage.create(image_path)
        image_tensor = self._preprocess_image(image)
        
        # Run prediction
        with torch.no_grad():
            prediction = self.learn.model(image_tensor.unsqueeze(0))
            prediction = torch.softmax(prediction, dim=1)
        
        # Convert to numpy
        pred_np = prediction.cpu().numpy()[0]
        
        # Get class predictions
        class_predictions = np.argmax(pred_np, axis=0)
        
        # Create mask for each class (excluding background)
        masks = []
        for class_id in range(1, pred_np.shape[0]):  # Skip background class 0
            mask = (class_predictions == class_id).astype(np.uint8)
            confidence = np.mean(pred_np[class_id][mask > 0]) if np.any(mask) else 0.0
            
            masks.append({
                'class_id': class_id,
                'mask': mask,
                'confidence': float(confidence)
            })
        
        return masks
    
    def extract_rois(self, image_path: Union[str, Path], masks: List[Dict[str, Any]]) -> List[np.ndarray]:
        """
        Extract ROIs (Regions of Interest) from an image based on segmentation masks.
        
        Args:
            image_path: Path to the image file
            masks: List of mask dictionaries from predict()
            
        Returns:
            List of ROI arrays
        """
        # Load the original image
        image = PILImage.create(image_path)
        image_array = np.array(image)
        
        rois = []
        for mask_info in masks:
            mask = mask_info['mask']
            
            # Find bounding box of the mask
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            
            if not np.any(rows) or not np.any(cols):
                continue  # Skip empty masks
                
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            
            # Add padding
            padding = 10
            rmin = max(0, rmin - padding)
            rmax = min(image_array.shape[0], rmax + padding)
            cmin = max(0, cmin - padding)
            cmax = min(image_array.shape[1], cmax + padding)
            
            # Extract ROI
            roi = image_array[rmin:rmax, cmin:cmax]
            rois.append(roi)
        
        return rois
    
    def predict_batch(self, image_paths: List[Union[str, Path]]) -> List[np.ndarray]:
        """
        Predict segmentation masks for multiple images.
        
        Args:
            image_paths: List of paths to input images
            
        Returns:
            List of predicted segmentation masks
        """
        if self.learn is None:
            raise ValueError("Model must be trained before prediction. Call train() first.")
        
        masks = []
        for image_path in image_paths:
            mask = self.predict(image_path)
            masks.append(mask)
        
        return masks
    
    def save_model(self, path: Optional[Path] = None) -> None:
        """
        Save the trained model.
        
        Args:
            path: Path to save the model (uses config if None)
        """
        if self.learn is None:
            raise ValueError("No model to save. Train a model first.")
        
        path = path or self.config.model_save_path
        if path is None:
            raise ValueError("No save path specified in config or method call.")
        
        # Create directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the model
        self.learn.export(path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: Path) -> None:
        """
        Load a trained model.
        
        Args:
            path: Path to the saved model
        """
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        # Load the model
        self.learn = load_learner(path)
        
        # Setup device
        self.device = self._setup_device()
        self.learn.to(self.device)
        
        print(f"Model loaded from {path}")
    
    def show_results(self, max_n: int = 3, figsize: Tuple[int, int] = (12, 4)) -> None:
        """
        Show training results.
        
        Args:
            max_n: Maximum number of examples to show
            figsize: Figure size for plotting
        """
        if self.learn is None:
            raise ValueError("No model to show results for. Train a model first.")
        
        self.learn.show_results(max_n=max_n, figsize=figsize)
    
    def plot_training_history(self) -> None:
        """Plot training history."""
        if self.learn is None:
            raise ValueError("No model to plot history for. Train a model first.")
        
        self.learn.recorder.plot_loss()


def create_mitochondria_segmenter(config: Optional[SegmentationConfig] = None) -> FastaiSegmenter:
    """
    Create a mitochondria segmentation model.
    
    Args:
        config: Configuration object (uses default if None)
        
    Returns:
        Configured FastaiSegmenter for mitochondria segmentation
    """
    if config is None:
        config = SegmentationConfig()
    
    return FastaiSegmenter(config)


def create_glomeruli_segmenter(config: Optional[SegmentationConfig] = None) -> FastaiSegmenter:
    """
    Create a glomeruli segmentation model.
    
    Args:
        config: Configuration object (uses default if None)
        
    Returns:
        Configured FastaiSegmenter for glomeruli segmentation
    """
    if config is None:
        config = SegmentationConfig()
    
    return FastaiSegmenter(config)
