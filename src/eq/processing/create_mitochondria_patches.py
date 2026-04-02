#!/usr/bin/env python3
"""
Mitochondria Patch Creation Script

This script creates patches from the organized mitochondria data and organizes them
into the correct directory structure for training.

IMPORTANT: This script follows the HISTORICAL mitochondria training approach:
- Output size: 224x224 (not 256x256 like glomeruli)
- Lighter preprocessing: min_scale=0.3 (not 0.45 like glomeruli)
- Designed for transfer learning foundation

Input Structure (from organize_lucchi_dataset.py):
data/mitochondria_data/
├── training/
│   ├── images/          # Training images (TIF)
│   └── masks/           # Training masks (TIF)
└── testing/
    ├── images/          # Test images (TIF)
    └── masks/           # Test masks (TIF)

Output Structure (matching historical training):
data/mitochondria_data/
├── training/
│   ├── images/          # Original training images
│   ├── masks/           # Original training masks
│   ├── image_patches/   # 224x224 image patches (historical size)
│   ├── mask_patches/    # 224x224 mask patches (historical size)
│   ├── image_patch_validation/  # Validation patches
│   └── mask_patch_validation/   # Validation mask patches
├── testing/
│   ├── images/          # Original test images
│   ├── masks/           # Original test masks
│   ├── image_patches/   # Test image patches
│   ├── mask_patches/    # Test mask patches
│   ├── image_patch_validation/  # Test validation patches
│   └── mask_patch_validation/   # Test validation mask patches
└── cache/                # Dataset-specific cache

Usage:
    python -m eq.processing.create_mitochondria_patches --input-dir data/mitochondria_data
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_patches_from_image(image_path: Path, mask_path: Path, patch_size: int = 224, overlap: float = 0.1):
    """
    Create patches from a single image and its corresponding mask.
    
    Args:
        image_path: Path to the image file
        mask_path: Path to the mask file
        patch_size: Size of patches (default: 224 for historical mitochondria training)
        overlap: Overlap between patches (default: 0.1)
    
    Returns:
        Tuple of (image_patches, mask_patches)
    """
    # Load image and mask
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    
    if image is None or mask is None:
        logger.warning(f"Could not load image or mask: {image_path}, {mask_path}")
        return [], []
    
    # Calculate step size based on overlap
    step = int(patch_size * (1 - overlap))
    
    # Calculate number of patches
    h, w = image.shape
    num_patches_h = (h - patch_size) // step + 1
    num_patches_w = (w - patch_size) // step + 1
    
    image_patches = []
    mask_patches = []
    
    # Create patches
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            y = i * step
            x = j * step
            
            # Extract patch
            image_patch = image[y:y + patch_size, x:x + patch_size]
            mask_patch = mask[y:y + patch_size, x:x + patch_size]
            
            # Filter out patches that are mostly empty (optional)
            # Use lower threshold for mitochondria (more sensitive to small structures)
            if np.mean(mask_patch) > 0.005:  # At least 0.5% of pixels are foreground
                image_patches.append(image_patch)
                mask_patches.append(mask_patch)
    
    return image_patches, mask_patches


def save_patches(
    patches: List[np.ndarray], 
    output_dir: Path, 
    base_name: str, 
    patch_type: str
) -> None:
    """
    Save patches to the specified output directory.
    
    Args:
        patches: List of patch arrays
        output_dir: Directory to save patches
        base_name: Base name for the patches
        patch_type: Type of patches ('image' or 'mask')
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, patch in enumerate(patches):
        if patch_type == 'image':
            output_path = output_dir / f"{base_name}_patch_{i:03d}.jpg"
            cv2.imwrite(str(output_path), patch)
        else:  # mask
            output_path = output_dir / f"{base_name}_mask_patch_{i:03d}.jpg"
            cv2.imwrite(str(output_path), patch)


def create_validation_patches(
    image_patches: List[np.ndarray], 
    mask_patches: List[np.ndarray],
    val_ratio: float = 0.2
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Split patches into training and validation sets.
    
    Args:
        image_patches: List of image patches
        mask_patches: List of mask patches
        val_ratio: Ratio of patches to use for validation
    
    Returns:
        Tuple of (train_images, train_masks, val_images, val_masks)
    """
    n_patches = len(image_patches)
    n_val = int(n_patches * val_ratio)
    
    # Randomly shuffle indices
    indices = np.random.permutation(n_patches)
    
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    
    train_images = [image_patches[i] for i in train_indices]
    train_masks = [mask_patches[i] for i in train_indices]
    val_images = [image_patches[i] for i in val_indices]
    val_masks = [mask_patches[i] for i in val_indices]
    
    return train_images, train_masks, val_images, val_masks


def process_dataset_split(images_dir: Path, masks_dir: Path, output_dir: Path, split_name: str, 
                         patch_size: int = 224, overlap: float = 0.1, val_ratio: float = 0.2):
    """
    Process a dataset split (training or testing) to create patches.
    
    Args:
        images_dir: Directory containing images
        masks_dir: Directory containing masks
        output_dir: Base output directory
        split_name: Name of the split ('training' or 'testing')
        patch_size: Size of patches (default: 224 for historical mitochondria training)
        overlap: Overlap between patches
        val_ratio: Ratio of patches for validation
    """
    try:
        from alive_progress import alive_bar
        has_alive_progress = True
    except ImportError:
        has_alive_progress = False
        # Create a simple progress bar fallback
        class SimpleProgressBar:
            def __init__(self, total, title="", **kwargs):
                self.total = total
                self.current = 0
                self.title = title
                print(f"{title}: Starting...")
            
            def __enter__(self):
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                print(f"{self.title}: Complete!")
            
            def __call__(self):
                self.current += 1
                if self.current % max(1, self.total // 10) == 0:  # Print every 10%
                    print(f"{self.title}: {self.current}/{self.total}")
        
        alive_bar = SimpleProgressBar
    
    logger.info(f"🔄 Processing {split_name} split...")
    logger.info(f"   📁 Images directory: {images_dir}")
    logger.info(f"   📁 Masks directory: {masks_dir}")
    logger.info(f"   🎯 Patch size: {patch_size}x{patch_size} (historical mitochondria size)")
    logger.info(f"   🔗 Overlap: {overlap*100:.0f}%")
    logger.info(f"   📊 Validation ratio: {val_ratio*100:.0f}%")
    
    # Get all image files
    image_files = list(images_dir.glob("*.tif"))
    if not image_files:
        logger.warning(f"❌ No image files found in {images_dir}")
        return
    
    logger.info(f"   📊 Found {len(image_files)} image files")
    
    # Create output directories
    patches_output_dir = output_dir / split_name / "image_patches"
    mask_patches_output_dir = output_dir / split_name / "mask_patches"
    val_patches_output_dir = output_dir / split_name / "image_patch_validation"
    val_mask_patches_output_dir = output_dir / split_name / "mask_patch_validation"
    
    for dir_path in [patches_output_dir, mask_patches_output_dir, val_patches_output_dir, val_mask_patches_output_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("   📂 Created output directories")
    
    all_image_patches = []
    all_mask_patches = []
    patch_names = []
    
    # Process each image with progress bar
    logger.info(f"   🔍 Processing {len(image_files)} images...")
    with alive_bar(len(image_files), title=f"📸 Processing {split_name} images", 
                  bar='smooth', spinner='dots', 
                  elapsed=True, stats=True, 
                  title_length=30) as bar:
        
        for image_file in image_files:
            # Find corresponding mask file
            # Handle both naming conventions: training_0.tif -> training_groundtruth_0.tif
            # and testing_0.tif -> testing_groundtruth_0.tif
            image_name = image_file.stem  # e.g., "training_0"
            mask_name = f"{image_name.replace('training_', 'training_groundtruth_').replace('testing_', 'testing_groundtruth_')}.tif"
            mask_file = masks_dir / mask_name
            
            if not mask_file.exists():
                logger.warning(f"      ⚠️  Mask file not found for {image_file.name}")
                bar()
                continue
            
            # Create patches
            image_patches, mask_patches = create_patches_from_image(image_file, mask_file, patch_size, overlap)
            
            if image_patches:
                all_image_patches.extend(image_patches)
                all_mask_patches.extend(mask_patches)
                
                # Generate patch names
                for k in range(len(image_patches)):
                    patch_names.append(f"{image_name}_patch_{k}")
            
            # Update progress bar
            bar()
    
    if not all_image_patches:
        logger.warning(f"❌ No patches created for {split_name}")
        return
    
    logger.info(f"   🎯 Total patches created: {len(all_image_patches)}")
    
    # Split into training and validation
    num_patches = len(all_image_patches)
    num_val = int(num_patches * val_ratio)
    num_train = num_patches - num_val
    
    logger.info(f"   📊 Splitting patches: {num_train} training, {num_val} validation")
    
    # Shuffle indices
    indices = np.random.permutation(num_patches)
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]
    
    logger.info(f"   💾 Saving {num_train} training patches...")
    
    # Save training patches with progress bar
    with alive_bar(num_train, title="💾 Saving training patches", 
                  bar='smooth', spinner='dots', 
                  elapsed=True, stats=True, 
                  title_length=30) as bar:
        
        for i, idx in enumerate(train_indices):
            patch_name = patch_names[idx]
            
            # Save image patch
            image_patch_path = patches_output_dir / f"{patch_name}.jpg"
            cv2.imwrite(str(image_patch_path), all_image_patches[idx])
            
            # Save mask patch
            mask_patch_path = mask_patches_output_dir / f"{patch_name}.jpg"
            cv2.imwrite(str(mask_patch_path), all_mask_patches[idx])
            
            # Update progress bar
            bar()
    
    logger.info(f"   💾 Saving {num_val} validation patches...")
    
    # Save validation patches with progress bar
    with alive_bar(num_val, title="💾 Saving validation patches", 
                  bar='smooth', spinner='dots', 
                  elapsed=True, stats=True, 
                  title_length=30) as bar:
        
        for i, idx in enumerate(val_indices):
            patch_name = patch_names[idx]
            
            # Save image patch
            image_patch_path = val_patches_output_dir / f"{patch_name}.jpg"
            cv2.imwrite(str(image_patch_path), all_image_patches[idx])
            
            # Save mask patch
            mask_patch_path = val_mask_patches_output_dir / f"{patch_name}.jpg"
            cv2.imwrite(str(mask_patch_path), all_mask_patches[idx])
            
            # Update progress bar
            bar()
    
    logger.info(f"✅ {split_name} processing complete!")
    logger.info(f"   📁 Training patches: {patches_output_dir}")
    logger.info(f"   📁 Training masks: {mask_patches_output_dir}")
    logger.info(f"   📁 Validation patches: {val_patches_output_dir}")
    logger.info(f"   📁 Validation masks: {val_mask_patches_output_dir}")


def create_mitochondria_patches(input_dir: str, patch_size: int = 224, overlap: float = 0.1, val_ratio: float = 0.2):
    """
    Create patches from organized mitochondria data.
    
    IMPORTANT: This function follows the HISTORICAL mitochondria training approach:
    - Default patch size: 224x224 (not 256x256 like glomeruli)
    - Designed as foundation for transfer learning to glomeruli model
    
    Args:
        input_dir: Path to organized mitochondria data directory
        patch_size: Size of patches (default: 224 for historical mitochondria training)
        overlap: Overlap between patches (default: 0.1)
        val_ratio: Ratio of patches for validation (default: 0.2)
    """
    import time
    start_time = time.time()
    
    input_path = Path(input_dir)
    
    logger.info("🚀 Starting mitochondria patch creation...")
    logger.info(f"📁 Input directory: {input_path}")
    logger.info(f"🎯 Patch size: {patch_size}x{patch_size} (historical mitochondria size)")
    logger.info(f"🔗 Overlap: {overlap*100:.0f}%")
    logger.info(f"📊 Validation ratio: {val_ratio*100:.0f}%")
    logger.info("💡 Note: Using 224x224 patches for historical mitochondria training compatibility")
    
    # Validate input structure
    required_dirs = [
        'training/images',
        'training/masks', 
        'testing/images',
        'testing/masks'
    ]
    
    logger.info("🔍 Validating input structure...")
    for req_dir in required_dirs:
        if not (input_path / req_dir).exists():
            raise FileNotFoundError(f"❌ Required directory not found: {req_dir}")
        logger.info(f"   ✅ Found: {req_dir}")
    
    # Create cache directory
    cache_dir = input_path / "cache"
    cache_dir.mkdir(exist_ok=True)
    logger.info(f"📂 Created cache directory: {cache_dir}")
    
    # Process training split
    logger.info("\n" + "="*60)
    process_dataset_split(
        input_path / "training" / "images",
        input_path / "training" / "masks", 
        input_path,
        "training",
        patch_size,
        overlap,
        val_ratio
    )
    
    # Process testing split
    logger.info("\n" + "="*60)
    process_dataset_split(
        input_path / "testing" / "images",
        input_path / "testing" / "masks",
        input_path, 
        "testing",
        patch_size,
        overlap,
        val_ratio
    )
    
    elapsed_time = time.time() - start_time
    logger.info("\n" + "="*60)
    logger.info("🎉 Mitochondria patch creation complete!")
    logger.info(f"⏱️  Total time: {elapsed_time:.1f} seconds")
    logger.info(f"📁 Output structure created at: {input_path}")
    logger.info("💡 These 224x224 patches are optimized for mitochondria training")
    logger.info("💡 They will serve as the foundation for glomeruli transfer learning")
    logger.info("="*60)


def main():
    parser = argparse.ArgumentParser(description="Create patches from organized mitochondria data")
    parser.add_argument(
        "--input-dir", 
        type=str, 
        default="data/mitochondria_data",
        help="Path to organized mitochondria data directory"
    )
    parser.add_argument(
        "--patch-size", 
        type=int, 
        default=224,  # Changed from 256 to 224 for historical compatibility
        help="Size of patches (default: 224 for historical mitochondria training)"
    )
    parser.add_argument(
        "--overlap", 
        type=float, 
        default=0.1,
        help="Overlap between patches (default: 0.1 = 10%)"
    )
    parser.add_argument(
        "--val-ratio", 
        type=float, 
        default=0.2,
        help="Ratio of patches for validation (default: 0.2)"
    )
    
    args = parser.parse_args()
    
    try:
        create_mitochondria_patches(
            args.input_dir,
            args.patch_size,
            args.overlap,
            args.val_ratio
        )
    except Exception as e:
        logger.error(f"Error creating patches: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
