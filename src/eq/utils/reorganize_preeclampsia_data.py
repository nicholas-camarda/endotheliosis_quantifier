#!/usr/bin/env python3
"""
Reorganize the messy preeclampsia data structure into a clean format.

Current mess:
- Duplicate images in train/T30/ and train/images/T30/
- Masks separated in train/masks/T30/
- Annotations at root level

Target structure:
raw_data/preeclampsia_data/
├── train/
│   ├── T30/
│   │   ├── images/
│   │   │   ├── T30_Image4.jpg
│   │   │   └── ...
│   │   └── masks/
│   │       ├── T30_Image4_mask.jpg
│   │       └── ...
│   ├── T31/
│   └── annotations.json
└── test/
"""

import logging
import shutil
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def reorganize_preeclampsia_data():
    """Reorganize the preeclampsia data into a clean structure."""
    
    base_dir = Path("raw_data/preeclampsia_data")
    train_dir = base_dir / "train"
    
    # Create backup
    backup_dir = base_dir / "backup_before_reorganization"
    if not backup_dir.exists():
        logger.info(f"Creating backup at {backup_dir}")
        shutil.copytree(train_dir, backup_dir)
    
    # Get all subject directories
    subject_dirs = []
    
    # Check both possible locations for subject directories
    for possible_dir in [train_dir, train_dir / "images"]:
        if possible_dir.exists():
            for item in possible_dir.iterdir():
                if item.is_dir() and item.name.startswith("T"):
                    subject_dirs.append(item.name)
    
    subject_dirs = sorted(list(set(subject_dirs)))  # Remove duplicates
    logger.info(f"Found subject directories: {subject_dirs}")
    
    # Process each subject
    for subject in subject_dirs:
        logger.info(f"Processing subject {subject}")
        
        # Create new structure
        new_subject_dir = train_dir / subject
        new_images_dir = new_subject_dir / "images"
        new_masks_dir = new_subject_dir / "masks"
        
        # Create directories
        new_images_dir.mkdir(parents=True, exist_ok=True)
        new_masks_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy images from both possible locations
        image_sources = [
            train_dir / subject,
            train_dir / "images" / subject
        ]
        
        images_copied = 0
        for source in image_sources:
            if source.exists():
                for image_file in source.glob("*.jpg"):
                    dest = new_images_dir / image_file.name
                    if not dest.exists():
                        shutil.copy2(image_file, dest)
                        images_copied += 1
                        logger.info(f"Copied image: {image_file.name}")
        
        # Copy masks
        mask_source = train_dir / "masks" / subject
        masks_copied = 0
        if mask_source.exists():
            for mask_file in mask_source.glob("*_mask.jpg"):
                dest = new_masks_dir / mask_file.name
                if not dest.exists():
                    shutil.copy2(mask_file, dest)
                    masks_copied += 1
                    logger.info(f"Copied mask: {mask_file.name}")
        
        logger.info(f"Subject {subject}: {images_copied} images, {masks_copied} masks")
    
    # Copy annotations
    annotation_files = list(train_dir.glob("*.json"))
    for ann_file in annotation_files:
        dest = train_dir / ann_file.name
        if not dest.exists():
            shutil.copy2(ann_file, dest)
            logger.info(f"Copied annotation: {ann_file.name}")
    
    # Clean up old structure (but keep backup)
    logger.info("Cleaning up old structure...")
    
    # Remove duplicate subject directories at root level
    for subject in subject_dirs:
        old_dir = train_dir / subject
        if old_dir.exists() and old_dir.is_dir():
            # Check if it only contains images (not the new structure)
            if not (old_dir / "images").exists():
                shutil.rmtree(old_dir)
                logger.info(f"Removed old subject directory: {subject}")
    
    # Remove old images directory
    old_images_dir = train_dir / "images"
    if old_images_dir.exists():
        shutil.rmtree(old_images_dir)
        logger.info("Removed old images directory")
    
    # Remove old masks directory
    old_masks_dir = train_dir / "masks"
    if old_masks_dir.exists():
        shutil.rmtree(old_masks_dir)
        logger.info("Removed old masks directory")
    
    logger.info("Reorganization complete!")
    logger.info(f"Backup saved at: {backup_dir}")

if __name__ == "__main__":
    reorganize_preeclampsia_data()
