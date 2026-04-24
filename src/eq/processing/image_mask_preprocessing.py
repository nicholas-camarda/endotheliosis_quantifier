from pathlib import Path

import tifffile


def _extract_tif_stack(tif_path: Path, output_dir: Path, prefix: str):
    """Extract individual images from a TIF stack."""
    print(f"📦 Extracting TIF stack: {tif_path}")
    
    # Read the TIF stack
    with tifffile.TiffFile(tif_path) as tif:
        # Get the number of images in the stack
        num_images = len(tif.pages)
        print(f"   📊 Found {num_images} images in TIF stack")
        
        # Extract each image
        for i in range(num_images):
            # Read the image
            img = tif.pages[i].asarray()
            
            # Create output filename
            output_filename = f"{prefix}_{i}.tif"
            output_path = output_dir / output_filename
            
            # Save individual image
            tifffile.imwrite(output_path, img)
    
    print(f"✅ Extracted {num_images} images to {output_dir}")
    return num_images


def extract_large_images(
    input_root: str,
    output_root: str,
) -> dict:
    """
    Extract large images from TIF files without patchifying them.
    
    This function:
    1. Finds TIF files in the input directory
    2. Extracts individual images from TIF stacks
    3. Saves them as large images in images/ and masks/ directories
    4. Does NOT create patches - preserves full image size
    
    Args:
        input_root: Input directory with TIF files
        output_root: Output directory for extracted images
        
    Returns:
        dict: {"images": int, "masks": int}
    """
    input_path = Path(input_root)
    output_path = Path(output_root)
    
    # Create output directory structure (alongside existing patches)
    images_dir = output_path / "images"
    masks_dir = output_path / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Creating large images in: {images_dir}")
    print(f"📁 Creating large masks in: {masks_dir}")
    
    images_count = 0
    masks_count = 0
    
    # Detect input structure - try multiple common naming conventions
    possible_image_dirs = ["img", "images", "data"]
    possible_mask_dirs = ["label", "labels", "masks", "groundtruth"]
    
    input_images_dir = None
    input_masks_dir = None
    
    # Find input image directory
    for dir_name in possible_image_dirs:
        candidate = input_path / dir_name
        if candidate.exists():
            input_images_dir = candidate
            break
    
    # Find input mask directory
    for dir_name in possible_mask_dirs:
        candidate = input_path / dir_name
        if candidate.exists():
            input_masks_dir = candidate
            break
    
    if not input_images_dir:
        raise FileNotFoundError(f"No image directory found in {input_path}. Expected one of: {possible_image_dirs}")
    
    print(f"📂 Processing images from: {input_images_dir}")
    if input_masks_dir:
        print(f"📂 Processing masks from: {input_masks_dir}")
    
    # Process image files
    image_files = list(input_images_dir.glob("*.tif")) + list(input_images_dir.glob("*.tiff"))
    
    for image_file in image_files:
        print(f"📦 Processing: {image_file.name}")
        
        # Extract images from TIF stack
        base_stem = image_file.stem
        if base_stem.endswith("_im"):
            # Mitochondria naming: train_im.tif -> train_label.tif
            base_stem = base_stem.replace("_im", "")
        
        # Extract individual images from TIF stack
        images_extracted = _extract_tif_stack(image_file, images_dir, base_stem)
        images_count += images_extracted
        
        # Process corresponding mask file if it exists
        if input_masks_dir:
            # Try different mask naming patterns
            mask_candidates = [
                input_masks_dir / f"{base_stem}_label.tif",
                input_masks_dir / f"{base_stem}_label.tiff",
                input_masks_dir / f"{base_stem}.tif",
                input_masks_dir / f"{base_stem}.tiff",
                input_masks_dir / image_file.name,  # Same filename
            ]
            
            mask_file = None
            for candidate in mask_candidates:
                if candidate.exists():
                    mask_file = candidate
                    break
            
            if mask_file:
                print(f"📦 Processing mask: {mask_file.name}")
                masks_extracted = _extract_tif_stack(mask_file, masks_dir, f"{base_stem}_mask")
                masks_count += masks_extracted
            else:
                print(f"⚠️  No corresponding mask found for {image_file.name}")
    
    print(f"✅ Extraction completed: {images_count} images, {masks_count} masks")
    return {"images": images_count, "masks": masks_count}
