#!/usr/bin/env python3
"""
Organize Lucchi mitochondria dataset into the pipeline's expected structure.

This script extracts individual images from TIF stacks and organizes them to match
the exact structure of the my_valid_structure directory.
"""

import argparse
import logging
from pathlib import Path

import tifffile

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_tif_stack(tif_path: Path, output_dir: Path, prefix: str):
    """
    Extract individual images from a TIF stack.
    
    Args:
        tif_path: Path to the TIF stack file
        output_dir: Directory to save individual images
        prefix: Prefix for the output filenames (e.g., 'training', 'testing')
    """
    import time

    from alive_progress import alive_bar
    
    start_time = time.time()
    
    logger.info(f"📦 Extracting TIF stack: {tif_path}")
    
    # Read the TIF stack
    with tifffile.TiffFile(tif_path) as tif:
        # Get the number of images in the stack
        num_images = len(tif.pages)
        logger.info(f"   📊 Found {num_images} images in TIF stack")
        
        # Extract each image with progress bar
        with alive_bar(num_images, title=f"📸 Extracting {prefix} images", 
                      bar='smooth', spinner='dots', 
                      elapsed=True, stats=True, 
                      title_length=30) as bar:
            
            for i in range(num_images):
                # Read the image
                img = tif.pages[i].asarray()
                
                # Create output filename
                output_filename = f"{prefix}_{i}.tif"
                output_path = output_dir / output_filename
                
                # Save individual image
                tifffile.imwrite(output_path, img)
                
                # Update progress bar
                bar()
    
    elapsed_time = time.time() - start_time
    logger.info(f"✅ Extracted {num_images} images to {output_dir}")
    logger.info(f"   ⏱️  Time: {elapsed_time:.1f}s, Rate: {num_images/elapsed_time:.1f} img/s")

def organize_lucchi_dataset(input_dir: str, output_dir: str = "data/mitochondria_data"):
    """
    Organize the Lucchi dataset into the pipeline's expected structure.
    
    Args:
        input_dir: Path to the input directory containing img/ and label/ folders
        output_dir: Path to the output directory
    """
    import time
    start_time = time.time()
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    logger.info("🚀 Starting Lucchi dataset organization...")
    logger.info(f"📁 Input directory: {input_path}")
    logger.info(f"📁 Output directory: {output_path}")
    
    # Validate input structure
    img_dir = input_path / "img"
    label_dir = input_path / "label"
    
    logger.info("🔍 Validating input structure...")
    if not img_dir.exists():
        raise FileNotFoundError(f"❌ Image directory not found: {img_dir}")
    logger.info(f"   ✅ Found: {img_dir}")
    
    if not label_dir.exists():
        raise FileNotFoundError(f"❌ Label directory not found: {label_dir}")
    logger.info(f"   ✅ Found: {label_dir}")
    
    # Find the TIF files
    train_img_file = img_dir / "train_im.tif"
    train_label_file = label_dir / "train_label.tif"
    test_img_file = img_dir / "test_im.tif"
    test_label_file = label_dir / "test_label.tif"
    
    logger.info("🔍 Validating TIF files...")
    if not train_img_file.exists():
        raise FileNotFoundError(f"❌ Training image file not found: {train_img_file}")
    logger.info(f"   ✅ Found: {train_img_file}")
    
    if not train_label_file.exists():
        raise FileNotFoundError(f"❌ Training label file not found: {train_label_file}")
    logger.info(f"   ✅ Found: {train_label_file}")
    
    if not test_img_file.exists():
        raise FileNotFoundError(f"❌ Testing image file not found: {test_img_file}")
    logger.info(f"   ✅ Found: {test_img_file}")
    
    if not test_label_file.exists():
        raise FileNotFoundError(f"❌ Testing label file not found: {test_label_file}")
    logger.info(f"   ✅ Found: {test_label_file}")
    
    # Create output directory structure
    training_images_dir = output_path / "training" / "images"
    training_masks_dir = output_path / "training" / "masks"
    testing_images_dir = output_path / "testing" / "images"
    testing_masks_dir = output_path / "testing" / "masks"
    
    logger.info("📂 Creating output directory structure...")
    for dir_path in [training_images_dir, training_masks_dir, testing_images_dir, testing_masks_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"   ✅ Created: {dir_path}")
    
    logger.info(f"📁 Output directory structure created at: {output_path}")
    
    # Extract training images and masks
    logger.info("\n" + "="*60)
    logger.info("🔄 Processing training data...")
    extract_tif_stack(train_img_file, training_images_dir, "training")
    extract_tif_stack(train_label_file, training_masks_dir, "training_groundtruth")
    
    # Extract testing images and masks
    logger.info("\n" + "="*60)
    logger.info("🔄 Processing testing data...")
    extract_tif_stack(test_img_file, testing_images_dir, "testing")
    extract_tif_stack(test_label_file, testing_masks_dir, "testing_groundtruth")
    
    # Create cache directory
    cache_dir = output_path / "cache"
    cache_dir.mkdir(exist_ok=True)
    logger.info(f"📂 Created cache directory: {cache_dir}")
    
    # Create README
    readme_path = output_path / "README.md"
    readme_content = """# Mitochondria Dataset Organization

This directory contains the Lucchi et al. 2012 mitochondria dataset reorganized for the endotheliosis quantifier pipeline.

## Original Source
- Dataset: Lucchi et al. 2012 benchmark dataset
- Source: http://rhoana.rc.fas.harvard.edu/dataset/lucchi.zip
- Reference: https://connectomics.readthedocs.io/en/latest/tutorials/mito.html

## Organization
```
mitochondria_data/
├── training/
│   ├── images/          # 165 individual training images (training_0.tif to training_164.tif)
│   └── masks/           # 165 individual training masks (training_groundtruth_0.tif to training_groundtruth_164.tif)
└── testing/
    ├── images/          # 165 individual test images (testing_0.tif to testing_164.tif)
    └── masks/           # 165 individual test masks (testing_groundtruth_0.tif to testing_groundtruth_164.tif)
```

## File Descriptions
- **Images**: Individual electron microscopy images extracted from TIF stacks (.tif format)
- **Masks**: Individual ground truth segmentation masks extracted from TIF stacks (.tif format)
- **Training**: 165 images used for model training
- **Testing**: 165 images used for model evaluation

## Usage in Pipeline
The pipeline expects this exact structure for:
1. Data loading and preprocessing
2. Patch generation
3. Model training and evaluation

## Reproducibility
To recreate this structure:
1. Download: wget http://rhoana.rc.fas.harvard.edu/dataset/lucchi.zip
2. Extract: unzip lucchi.zip
3. Run: python -m eq.utils.organize_lucchi_dataset --input-dir mnt/coxfs01/vcg_connectomics/mitochondria/Lucchi
"""
    
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    elapsed_time = time.time() - start_time
    logger.info("\n" + "="*60)
    logger.info("🎉 Dataset organized successfully!")
    logger.info(f"⏱️  Total time: {elapsed_time:.1f} seconds")
    logger.info(f"📁 Output location: {output_path}")
    logger.info("✅ Ready for pipeline processing!")
    logger.info("="*60)

def main():
    parser = argparse.ArgumentParser(description="Organize Lucchi mitochondria dataset")
    parser.add_argument("--input-dir", required=True, help="Input directory containing img/ and label/ folders")
    parser.add_argument("--output-dir", default="data/mitochondria_data", help="Output directory (default: data/mitochondria_data)")
    
    args = parser.parse_args()
    
    try:
        organize_lucchi_dataset(args.input_dir, args.output_dir)
    except Exception as e:
        logger.error(f"Failed to organize dataset: {e}")
        raise

if __name__ == "__main__":
    main()
