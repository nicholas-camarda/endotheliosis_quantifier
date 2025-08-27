"""
Raw Data Organization Utility

This module provides functions to programmatically organize raw data from
Lauren_PreEclampsia_Raw_TIF_Images/ into the proper train/test structure
with images/masks organization.
"""

import logging
import random
import shutil
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)


def organize_raw_data(
    raw_images_dir: str = "raw_data/preeclampsia_data/Lauren_PreEclampsia_Raw_TIF_Images",
    output_dir: str = "raw_data/preeclampsia_data",
    train_split: float = 0.8,
    random_seed: int = 42,
    convert_tif_to_jpg: bool = True
) -> Dict[str, Path]:
    """
    Programmatically organize raw data from Lauren_PreEclampsia_Raw_TIF_Images/
    into the proper train/test structure.
    
    Args:
        raw_images_dir: Path to the raw TIF images directory
        output_dir: Path to the output directory for organized data
        train_split: Fraction of data to use for training (0.0-1.0)
        random_seed: Random seed for reproducible train/test splits
        convert_tif_to_jpg: Whether to convert TIF files to JPG during organization
        
    Returns:
        Dictionary with paths to organized data directories
    """
    raw_images_path = Path(raw_images_dir)
    output_path = Path(output_dir)
    
    if not raw_images_path.exists():
        raise FileNotFoundError(f"Raw images directory not found: {raw_images_path}")
    
    # Set random seed for reproducible splits
    random.seed(random_seed)
    
    # Create output directory structure
    train_images_dir = output_path / "train" / "images"
    train_masks_dir = output_path / "train" / "masks"
    test_images_dir = output_path / "test" / "images"
    test_masks_dir = output_path / "test" / "masks"
    annotations_dir = output_path / "annotations"
    
    for dir_path in [train_images_dir, train_masks_dir, test_images_dir, test_masks_dir, annotations_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Find all subject directories (T19, T20, etc.)
    subject_dirs = [d for d in raw_images_path.iterdir() if d.is_dir() and d.name.startswith('T')]
    subject_dirs.sort()
    
    logger.info(f"Found {len(subject_dirs)} subject directories: {[d.name for d in subject_dirs]}")
    
    # Split subjects into train/test
    random.shuffle(subject_dirs)
    split_idx = int(len(subject_dirs) * train_split)
    train_subjects = subject_dirs[:split_idx]
    test_subjects = subject_dirs[split_idx:]
    
    logger.info(f"Train subjects: {[s.name for s in train_subjects]}")
    logger.info(f"Test subjects: {[s.name for s in test_subjects]}")
    
    # Process train subjects
    for subject_dir in train_subjects:
        _process_subject_directory(
            subject_dir, train_images_dir, train_masks_dir, convert_tif_to_jpg
        )
    
    # Process test subjects
    for subject_dir in test_subjects:
        _process_subject_directory(
            subject_dir, test_images_dir, test_masks_dir, convert_tif_to_jpg
        )
    
    # Copy annotations if they exist
    _copy_annotations(raw_images_path.parent, annotations_dir)
    
    logger.info("Raw data organization completed successfully!")
    
    return {
        "train_images": train_images_dir,
        "train_masks": train_masks_dir,
        "test_images": test_images_dir,
        "test_masks": test_masks_dir,
        "annotations": annotations_dir
    }


def _process_subject_directory(
    subject_dir: Path,
    images_output_dir: Path,
    masks_output_dir: Path,
    convert_tif_to_jpg: bool = True
) -> None:
    """
    Process a single subject directory, organizing images and finding corresponding masks.
    
    Args:
        subject_dir: Path to the subject directory (e.g., T30/)
        images_output_dir: Output directory for images
        masks_output_dir: Output directory for masks
        convert_tif_to_jpg: Whether to convert TIF files to JPG
    """
    subject_name = subject_dir.name
    logger.info(f"Processing subject {subject_name}...")
    
    # Create subject subdirectories
    subject_images_dir = images_output_dir / subject_name
    subject_masks_dir = masks_output_dir / subject_name
    subject_images_dir.mkdir(exist_ok=True)
    subject_masks_dir.mkdir(exist_ok=True)
    
    # Find all TIF files in the subject directory
    tif_files = list(subject_dir.glob("*.tif"))
    
    for tif_file in tif_files:
        image_name = tif_file.stem  # e.g., "Image0"
        
        # Determine output image path
        if convert_tif_to_jpg:
            output_image_path = subject_images_dir / f"{subject_name}_{image_name}.jpg"
        else:
            output_image_path = subject_images_dir / f"{subject_name}_{image_name}.tif"
        
        # Copy/convert image
        if convert_tif_to_jpg:
            _convert_tif_to_jpg(tif_file, output_image_path)
        else:
            shutil.copy2(tif_file, output_image_path)
        
        # Look for corresponding mask
        mask_found = _find_and_copy_mask(
            subject_name, image_name, subject_masks_dir
        )
        
        if not mask_found:
            logger.warning(f"No mask found for {subject_name}_{image_name}")


def _convert_tif_to_jpg(tif_path: Path, jpg_path: Path) -> None:
    """
    Convert a TIF file to JPG format.
    
    Args:
        tif_path: Path to the input TIF file
        jpg_path: Path to the output JPG file
    """
    try:
        from PIL import Image
        with Image.open(tif_path) as img:
            # Convert to RGB if necessary
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGB')
            img.save(jpg_path, 'JPEG', quality=95)
    except ImportError:
        logger.warning("PIL not available, copying TIF file instead")
        shutil.copy2(tif_path, jpg_path)
    except Exception as e:
        logger.error(f"Error converting {tif_path}: {e}")
        # Fallback to copying
        shutil.copy2(tif_path, jpg_path)


def _find_and_copy_mask(
    subject_name: str,
    image_name: str,
    masks_output_dir: Path
) -> bool:
    """
    Find and copy the corresponding mask file.
    
    Args:
        subject_name: Name of the subject (e.g., "T30")
        image_name: Name of the image (e.g., "Image0")
        masks_output_dir: Output directory for masks
        
    Returns:
        True if mask was found and copied, False otherwise
    """
    # Try multiple possible mask locations
    possible_mask_locations = [
        # Check if masks exist in the backup
        Path("raw_data/preeclampsia_data/backup_before_reorganization/masks") / subject_name / f"{image_name}_mask.jpg",
        # Check if masks exist in the current train structure
        Path("raw_data/preeclampsia_data/train/masks") / subject_name / f"{image_name}_mask.jpg",
        # Check if masks exist in the backup train structure
        Path("raw_data/preeclampsia_data/backup_before_reorganization/train/masks") / subject_name / f"{image_name}_mask.jpg",
    ]
    
    for mask_location in possible_mask_locations:
        if mask_location.exists():
            output_mask_path = masks_output_dir / f"{subject_name}_{image_name}_mask.jpg"
            shutil.copy2(mask_location, output_mask_path)
            logger.debug(f"Found mask: {mask_location} -> {output_mask_path}")
            return True
    
    return False


def _copy_annotations(source_dir: Path, annotations_dir: Path) -> None:
    """
    Copy annotation files to the annotations directory.
    
    Args:
        source_dir: Source directory to search for annotations
        annotations_dir: Output directory for annotations
    """
    # Look for annotation files
    annotation_patterns = ["*.json", "*.xlsx", "*.csv"]
    
    for pattern in annotation_patterns:
        for annotation_file in source_dir.rglob(pattern):
            if "annotation" in annotation_file.name.lower():
                output_path = annotations_dir / annotation_file.name
                shutil.copy2(annotation_file, output_path)
                logger.info(f"Copied annotation: {annotation_file} -> {output_path}")


def create_clean_raw_data_structure(
    raw_images_dir: str = "raw_data/preeclampsia_data/Lauren_PreEclampsia_Raw_TIF_Images",
    output_dir: str = "raw_data/preeclampsia_data",
    train_split: float = 0.8,
    random_seed: int = 42
) -> Dict[str, Path]:
    """
    Create a clean raw data structure from the messy Lauren_PreEclampsia_Raw_TIF_Images/.
    
    This function will:
    1. Read from Lauren_PreEclampsia_Raw_TIF_Images/
    2. Organize into proper train/test splits
    3. Create the clean directory structure
    4. Handle annotations properly
    5. Convert TIF to JPG if needed
    
    Args:
        raw_images_dir: Path to the raw TIF images directory
        output_dir: Path to the output directory for organized data
        train_split: Fraction of data to use for training (0.0-1.0)
        random_seed: Random seed for reproducible train/test splits
        
    Returns:
        Dictionary with paths to organized data directories
    """
    logger.info("Starting raw data organization...")
    logger.info(f"Raw images directory: {raw_images_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Train split: {train_split}")
    
    return organize_raw_data(
        raw_images_dir=raw_images_dir,
        output_dir=output_dir,
        train_split=train_split,
        random_seed=random_seed,
        convert_tif_to_jpg=True
    )


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run the organization
    result = create_clean_raw_data_structure()
    print("Organization completed!")
    print("Organized directories:")
    for key, path in result.items():
        print(f"  {key}: {path}")
