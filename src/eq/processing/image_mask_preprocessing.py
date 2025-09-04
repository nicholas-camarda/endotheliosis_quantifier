# patchify_images.py

import os
from pathlib import Path
import shutil

import cv2
import numpy as np
import tifffile
from PIL import Image

from eq.core.constants import DEFAULT_MASK_THRESHOLD, DEFAULT_PATCH_SIZE


def _should_process_file(image_path: Path) -> bool:
    """Avoid duplicate processing when both JPG and PNG exist for same stem.

    Preference: If a .png exists for the same stem, skip the .jpg/.jpeg.
    Always allow .png. Allow .tif/.tiff.
    """
    suffix = image_path.suffix.lower()
    if suffix in {'.jpg', '.jpeg'}:
        png_peer = image_path.with_suffix('.png')
        if png_peer.exists():
            return False
    return True

def _convert_jpeg_to_png(input_path, output_path):
    """Convert JPEG file to PNG to avoid compression artifacts."""
    try:
        with Image.open(input_path) as img:
            # Convert to RGB if necessary (JPEG might be in different mode)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.save(output_path, 'PNG')
        return True
    except Exception as e:
        print(f"Warning: Could not convert {input_path} to PNG: {e}")
        return False


def _patchify_single_image(square_size, image_path, output_dir, base_stem):
    """Helper function to patchify a single image."""
    img = cv2.imread(str(image_path))
    if img is None:
        return 0
    
    h, w = img.shape[:2]
    patch_count = 0
    
    for i in range(0, h, square_size):
        for j in range(0, w, square_size):
            patch = img[i:i+square_size, j:j+square_size]
            if patch.shape[0] != square_size or patch.shape[1] != square_size:
                continue
            output_filename = f"{base_stem}_{i//square_size}_{j//square_size}.png"
            output_path = Path(output_dir) / output_filename
            cv2.imwrite(str(output_path), patch)
            patch_count += 1
    
    return patch_count


def _patchify_image_with_mask(square_size, image_path, mask_path, output_dir, base_stem):
    """Helper function to patchify a single image with its corresponding mask."""
    img = cv2.imread(str(image_path))
    if img is None:
        return 0, 0
    
    h, w = img.shape[:2]
    image_patch_count = 0
    mask_patch_count = 0
    
    # Load and process mask
    mask_img = None
    if mask_path and mask_path.exists():
        mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_img is not None:
            # Ensure mask is binary (0 or 255) to prevent corruption
            mask_img = np.where(mask_img > DEFAULT_MASK_THRESHOLD, 255, 0).astype(np.uint8)
            # If mask size mismatches, skip mask writing but still write image patches
            if mask_img.shape[0] != h or mask_img.shape[1] != w:
                mask_img = None
    
    for i in range(0, h, square_size):
        for j in range(0, w, square_size):
            patch = img[i:i+square_size, j:j+square_size]
            if patch.shape[0] != square_size or patch.shape[1] != square_size:
                continue
            
            # Save image patch
            out_name = f"{base_stem}_{i//square_size}_{j//square_size}.png"
            cv2.imwrite(str(Path(output_dir) / out_name), patch)
            image_patch_count += 1

            # Save mask patch if available
            if mask_img is not None:
                mask_patch = mask_img[i:i+square_size, j:j+square_size]
                if mask_patch.shape[0] != square_size or mask_patch.shape[1] != square_size:
                    continue
                # Ensure mask patch is binary before saving (0 or 1 for FastAI)
                mask_patch = np.where(mask_patch > DEFAULT_MASK_THRESHOLD, 1, 0).astype(np.uint8)
                out_mask_name = f"{base_stem}_{i//square_size}_{j//square_size}_mask.png"
                # Use PNG format to preserve exact binary values
                cv2.imwrite(str(Path(output_dir) / out_mask_name), mask_patch)
                mask_patch_count += 1
    
    return image_patch_count, mask_patch_count


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


def _process_directory_recursive(square_size, input_dir, output_dir):
    """Helper function to process directories recursively."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    total_patches = 0
    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        if os.path.isdir(input_path):
            # Skip if this is the output directory to prevent infinite recursion
            if os.path.abspath(input_path) == os.path.abspath(output_dir):
                continue
            # recursively process subdirectories
            output_subdir = os.path.join(output_dir, filename)
            if not os.path.exists(output_subdir):
                os.mkdir(output_subdir)
            total_patches += _process_directory_recursive(square_size, input_path, output_subdir)
        elif filename.lower().endswith(('.png', '.tif', '.tiff', '.jpg', '.jpeg')):
            if not _should_process_file(Path(input_path)):
                continue
            base_stem = os.path.splitext(filename)[0]
            total_patches += _patchify_single_image(square_size, input_path, output_dir, base_stem)
    
    return total_patches


def patchify_dataset(
    input_root: str,
    output_root: str,
    patch_size: int = DEFAULT_PATCH_SIZE,
) -> dict:
    """
    Unified patchification entrypoint with auto structure and mask detection.
    - Supports nested structures with images/ and masks/ subject subfolders
    - Supports images-only datasets (recurses)
    - Automatically converts JPEG files to PNG to avoid compression artifacts
    - Enforces binary masks and PNG format for data integrity

    Returns counts: {"images": int, "masks": int, "subjects": int}
    """
    input_path = Path(input_root)
    output_path = Path(output_root)
    image_patches_dir = output_path / "image_patches"
    mask_patches_dir = output_path / "mask_patches"
    image_patches_dir.mkdir(parents=True, exist_ok=True)
    mask_patches_dir.mkdir(parents=True, exist_ok=True)

    images_count = 0
    masks_count = 0
    subjects_count = 0

    # Note: We do NOT modify files under input_root (raw data remains untouched).
    # JPEGs and mislabeled files will be handled during patching reads; outputs are written under output_root only.

    # Detect nested structure - try multiple common naming conventions
    possible_image_dirs = ["images", "image_patches", "img", "imgs", "data"]
    possible_mask_dirs = ["masks", "mask_patches", "labels", "label", "groundtruth"]
    
    images_dir = None
    masks_dir = None
    has_images_dir = False
    has_masks_dir = False
    
    # Find image directory
    for dir_name in possible_image_dirs:
        candidate = input_path / dir_name
        if candidate.exists():
            images_dir = candidate
            has_images_dir = True
            break
    
    # Find mask directory
    for dir_name in possible_mask_dirs:
        candidate = input_path / dir_name
        if candidate.exists():
            masks_dir = candidate
            has_masks_dir = True
            break

    if has_images_dir and images_dir is not None:
        # Check if this is a flat structure (files directly in images/ and masks/)
        image_files = list(images_dir.glob("*"))
        has_subdirectories = any(f.is_dir() for f in image_files)
        
        if has_subdirectories:
            # Nested structure with subjects
            for subject_dir in images_dir.iterdir():
                if not subject_dir.is_dir():
                    continue
                subjects_count += 1
                subject_name = subject_dir.name
                subject_masks_dir = (masks_dir / subject_name) if has_masks_dir and masks_dir is not None else None

                if has_masks_dir and subject_masks_dir is not None and subject_masks_dir.exists():
                    # Patchify paired data
                    tmp_dir = image_patches_dir / f".__tmp_{subject_name}"
                    tmp_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Process each image file in the subject directory
                    for image_file in subject_dir.glob("*"):
                        if not image_file.is_file() or not image_file.suffix.lower() in ['.png', '.tif', '.tiff', '.jpg', '.jpeg']:
                            continue
                        if not _should_process_file(image_file):
                            continue
                        
                        base_stem = image_file.stem
                        # Look for corresponding mask
                        mask_file = None
                        for ext in ['.png', '.tif', '.tiff', '.jpg', '.jpeg']:
                            candidate = subject_masks_dir / f"{base_stem}_mask{ext}"
                            if candidate.exists():
                                mask_file = candidate
                                break
                        
                        if mask_file:
                            img_count, mask_count = _patchify_image_with_mask(
                                patch_size, image_file, mask_file, tmp_dir, base_stem
                            )
                            images_count += img_count
                            masks_count += mask_count
                        else:
                            img_count = _patchify_single_image(
                                patch_size, image_file, tmp_dir, base_stem
                            )
                            images_count += img_count
                    
                    # Move patches to appropriate directories
                    for p in tmp_dir.glob("*_mask.png"):
                        shutil.move(str(p), str(mask_patches_dir / p.name))
                    for p in tmp_dir.glob("*.png"):
                        if not p.name.endswith("_mask.png"):
                            shutil.move(str(p), str(image_patches_dir / p.name))
                    
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                else:
                    # Images only, use recursive processing
                    before = len(list(image_patches_dir.rglob("*.png")))
                    _process_directory_recursive(patch_size, str(subject_dir), str(image_patches_dir))
                    after = len(list(image_patches_dir.rglob("*.png")))
                    images_count += max(0, after - before)
        else:
            # Flat structure - files directly in images/ and masks/
            subjects_count = 1  # Treat as single subject
            tmp_dir = image_patches_dir / ".__tmp_flat"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            
            # First, extract any TIF stacks to individual images
            extraction_dir = tmp_dir / "extracted"
            extraction_dir.mkdir(exist_ok=True)
            
            # Extract image TIF stacks
            for image_file in images_dir.glob("*"):
                if not image_file.is_file() or not image_file.suffix.lower() in ['.tif', '.tiff']:
                    continue
                
                # Check if this is a TIF stack (multiple pages)
                try:
                    with tifffile.TiffFile(image_file) as tif:
                        if len(tif.pages) > 1:
                            print(f"🔍 Detected image TIF stack: {image_file} ({len(tif.pages)} images)")
                            # Extract the stack
                            base_stem = image_file.stem
                            _extract_tif_stack(image_file, extraction_dir, base_stem)
                        else:
                            # Single image TIF, just copy it
                            shutil.copy2(image_file, extraction_dir / image_file.name)
                except Exception as e:
                    print(f"⚠️  Could not process {image_file}: {e}")
                    continue
            
            # Extract mask TIF stacks if available
            if has_masks_dir and masks_dir is not None:
                mask_extraction_dir = tmp_dir / "extracted_masks"
                mask_extraction_dir.mkdir(exist_ok=True)
                
                for mask_file in masks_dir.glob("*"):
                    if not mask_file.is_file() or not mask_file.suffix.lower() in ['.tif', '.tiff']:
                        continue
                    
                    # Check if this is a TIF stack (multiple pages)
                    try:
                        with tifffile.TiffFile(mask_file) as tif:
                            if len(tif.pages) > 1:
                                print(f"🔍 Detected mask TIF stack: {mask_file} ({len(tif.pages)} masks)")
                                # Extract the stack
                                base_stem = mask_file.stem
                                _extract_tif_stack(mask_file, mask_extraction_dir, base_stem)
                            else:
                                # Single image TIF, just copy it
                                shutil.copy2(mask_file, mask_extraction_dir / mask_file.name)
                    except Exception as e:
                        print(f"⚠️  Could not process {mask_file}: {e}")
                        continue
            
            # Copy any non-TIF files
            for image_file in images_dir.glob("*"):
                if not image_file.is_file() or image_file.suffix.lower() in ['.tif', '.tiff']:
                    continue
                shutil.copy2(image_file, extraction_dir / image_file.name)
            
            # Now process the extracted individual images
            for image_file in extraction_dir.glob("*"):
                if not image_file.is_file() or not image_file.suffix.lower() in ['.png', '.tif', '.tiff', '.jpg', '.jpeg']:
                    continue
                if not _should_process_file(image_file):
                    continue
                
                base_stem = image_file.stem
                # Look for corresponding mask in extracted masks directory
                mask_file = None
                if has_masks_dir and masks_dir is not None:
                    # First try the extracted masks directory
                    mask_extraction_dir = tmp_dir / "extracted_masks"
                    if mask_extraction_dir.exists():
                        # Try to find matching mask by index
                        # base_stem format: "train_im_0", "test_im_1", etc.
                        # We need to find: "train_label_0", "test_label_1", etc.
                        if "_im_" in base_stem:
                            # Extract the prefix and index
                            parts = base_stem.split("_im_")
                            if len(parts) == 2:
                                prefix = parts[0]  # "train" or "test"
                                index = parts[1]  # "0", "1", etc.
                                # Look for corresponding mask
                                mask_candidates = [
                                    mask_extraction_dir / f"{prefix}_label_{index}.tif",
                                    mask_extraction_dir / f"{prefix}_label_{index}.tiff"
                                ]
                                for candidate in mask_candidates:
                                    if candidate.exists():
                                        mask_file = candidate
                                        break
                    
                    # Fallback to original mask directory if no extracted mask found
                    if mask_file is None:
                        for ext in ['.png', '.tif', '.tiff', '.jpg', '.jpeg']:
                            # Try different mask naming patterns
                            candidates = [
                                masks_dir / f"{base_stem}_mask{ext}",
                                masks_dir / f"{base_stem.replace('_im', '_label')}{ext}",
                                masks_dir / f"{base_stem.replace('train_im', 'train_label')}{ext}",
                                masks_dir / f"{base_stem.replace('test_im', 'test_label')}{ext}",
                                masks_dir / f"{base_stem}{ext}"
                            ]
                            for candidate in candidates:
                                if candidate.exists():
                                    mask_file = candidate
                                    break
                            if mask_file:
                                break
                
                if mask_file:
                    img_count, mask_count = _patchify_image_with_mask(
                        patch_size, image_file, mask_file, tmp_dir, base_stem
                    )
                    images_count += img_count
                    masks_count += mask_count
                else:
                    img_count = _patchify_single_image(
                        patch_size, image_file, tmp_dir, base_stem
                    )
                    images_count += img_count
            
            # Move patches to appropriate directories
            for p in tmp_dir.glob("*_mask.png"):
                shutil.move(str(p), str(mask_patches_dir / p.name))
            for p in tmp_dir.glob("*.png"):
                if not p.name.endswith("_mask.png"):
                    shutil.move(str(p), str(image_patches_dir / p.name))
            
            shutil.rmtree(tmp_dir, ignore_errors=True)
    else:
        # Flat or arbitrary tree; images-only fallback
        before = len(list(image_patches_dir.rglob("*.png")))
        _process_directory_recursive(patch_size, str(input_path), str(image_patches_dir))
        after = len(list(image_patches_dir.rglob("*.png")))
        images_count += max(0, after - before)

    return {
        "images": images_count,
        "masks": masks_count,
        "subjects": subjects_count,
    }