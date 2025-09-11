#!/usr/bin/env python3
"""
Comprehensive debugging script for dynamic patching datablock pipeline.

This script investigates potential issues with the datablock that could cause
low dice scores during transfer learning.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import torch
from fastai.vision.all import PILMask, get_image_files
from fastai.data.transforms import RandomSplitter

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from eq.data_management.datablock_loader import (
    build_segmentation_datablock_dynamic_patching,
    build_segmentation_dls_dynamic_patching,
    CropTransform,
    get_items_full_images
)
from eq.data_management.standard_getters import get_y
from eq.core.constants import (
    DEFAULT_IMAGE_SIZE, DEFAULT_MASK_THRESHOLD, DEFAULT_VAL_RATIO,
    DEFAULT_MAX_ROTATE, DEFAULT_FLIP_VERT, DEFAULT_MIN_ZOOM, 
    DEFAULT_MAX_ZOOM, DEFAULT_MAX_WARP, DEFAULT_MAX_LIGHTING
)
from eq.utils.logger import get_logger

def debug_data_structure(data_root):
    """Debug the data directory structure and file organization."""
    print("=" * 80)
    print("DEBUGGING DATA STRUCTURE")
    print("=" * 80)
    
    root = Path(data_root)
    print(f"Data root: {root}")
    print(f"Data root exists: {root.exists()}")
    
    if not root.exists():
        print("❌ Data root does not exist!")
        return False
    
    # Check for different directory structures
    possible_dirs = ["images", "image_patches", "img", "masks", "mask_patches", "label"]
    found_dirs = []
    
    for dir_name in possible_dirs:
        dir_path = root / dir_name
        if dir_path.exists():
            found_dirs.append(dir_name)
            file_count = len(list(dir_path.rglob("*")))
            print(f"✅ Found {dir_name}/ with {file_count} files")
        else:
            print(f"❌ Missing {dir_name}/")
    
    print(f"\nFound directories: {found_dirs}")
    return len(found_dirs) > 0

def debug_get_items_function(data_root):
    """Debug the get_items function behavior."""
    print("\n" + "=" * 80)
    print("DEBUGGING GET_ITEMS FUNCTION")
    print("=" * 80)
    
    try:
        items = get_items_full_images(Path(data_root))
        print(f"✅ get_items_full_images returned {len(items)} items")
        
        if len(items) > 0:
            print(f"First 5 items:")
            for i, item in enumerate(items[:5]):
                print(f"  {i+1}. {item}")
            
            # Check if items are valid paths
            valid_count = 0
            for item in items[:10]:  # Check first 10
                if Path(item).exists():
                    valid_count += 1
                else:
                    print(f"❌ Invalid path: {item}")
            
            print(f"Valid paths in first 10: {valid_count}/10")
        
        return items
    except Exception as e:
        print(f"❌ Error in get_items_full_images: {e}")
        return []

def debug_get_y_function(items):
    """Debug the get_y function behavior."""
    print("\n" + "=" * 80)
    print("DEBUGGING GET_Y FUNCTION")
    print("=" * 80)
    
    if not items:
        print("❌ No items to test get_y function")
        return []
    
    successful_masks = []
    failed_masks = []
    
    for i, item in enumerate(items[:10]):  # Test first 10 items
        try:
            mask = get_y(item)
            successful_masks.append((item, mask))
            print(f"✅ {i+1}. {Path(item).name} -> mask loaded successfully")
            
            # Check mask properties
            mask_array = np.array(mask)
            print(f"   Mask shape: {mask_array.shape}")
            print(f"   Mask dtype: {mask_array.dtype}")
            print(f"   Mask min/max: {mask_array.min():.3f}/{mask_array.max():.3f}")
            print(f"   Unique values: {np.unique(mask_array)}")
            print(f"   Positive pixels: {(mask_array > 0).sum()}/{mask_array.size} ({(mask_array > 0).sum()/mask_array.size:.2%})")
            
        except Exception as e:
            failed_masks.append((item, str(e)))
            print(f"❌ {i+1}. {Path(item).name} -> Error: {e}")
    
    print(f"\nSummary: {len(successful_masks)} successful, {len(failed_masks)} failed")
    return successful_masks

def debug_crop_transform(image_path, mask_path):
    """Debug the CropTransform implementation."""
    print("\n" + "=" * 80)
    print("DEBUGGING CROP TRANSFORM")
    print("=" * 80)
    
    try:
        # Load original image and mask
        from fastai.vision.all import PILImage
        original_img = PILImage.create(image_path)
        original_mask = PILMask.create(mask_path)
        
        print(f"Original image shape: {original_img.size}")
        print(f"Original mask shape: {original_mask.size}")
        
        # Test CropTransform
        crop_transform = CropTransform(size=DEFAULT_IMAGE_SIZE)
        
        # Apply transform multiple times to test randomness
        for i in range(3):
            result = crop_transform.encodes((original_img, original_mask))
            if isinstance(result, tuple) and len(result) == 2:
                cropped_img, cropped_mask = result
                print(f"Crop {i+1}:")
                print(f"  Cropped image shape: {cropped_img.size if hasattr(cropped_img, 'size') else cropped_img.shape}")
                print(f"  Cropped mask shape: {cropped_mask.size if hasattr(cropped_mask, 'size') else cropped_mask.shape}")
                
                # Check mask content after cropping
                mask_array = np.array(cropped_mask)
                positive_pixels = (mask_array > 0).sum()
                print(f"  Positive pixels after crop: {positive_pixels}/{mask_array.size} ({positive_pixels/mask_array.size:.2%})")
            else:
                print(f"❌ Crop {i+1} failed: unexpected result type")
        
        return True
    except Exception as e:
        print(f"❌ Error in CropTransform: {e}")
        import traceback
        traceback.print_exc()
        return False

def debug_datablock_creation(data_root):
    """Debug the DataBlock creation process."""
    print("\n" + "=" * 80)
    print("DEBUGGING DATABLOCK CREATION")
    print("=" * 80)
    
    try:
        # Create DataBlock
        db = build_segmentation_datablock_dynamic_patching()
        print("✅ DataBlock created successfully")
        
        # Create DataLoaders
        dls = db.dataloaders(Path(data_root), bs=2, num_workers=0)
        print("✅ DataLoaders created successfully")
        
        print(f"Train dataset size: {len(dls.train_ds)}")
        print(f"Valid dataset size: {len(dls.valid_ds)}")
        
        return dls
    except Exception as e:
        print(f"❌ Error creating DataBlock/DataLoaders: {e}")
        import traceback
        traceback.print_exc()
        return None

def debug_batch_processing(dls):
    """Debug batch processing and augmentation pipeline."""
    print("\n" + "=" * 80)
    print("DEBUGGING BATCH PROCESSING")
    print("=" * 80)
    
    if dls is None:
        print("❌ No DataLoaders to test")
        return
    
    try:
        # Get a batch from training data
        batch = dls.train.one_batch()
        print(f"✅ Batch loaded successfully")
        print(f"Batch type: {type(batch)}")
        print(f"Batch length: {len(batch)}")
        
        if len(batch) == 2:
            images, masks = batch
            print(f"Images shape: {images.shape}")
            print(f"Masks shape: {masks.shape}")
            print(f"Images dtype: {images.dtype}")
            print(f"Masks dtype: {masks.dtype}")
            print(f"Images min/max: {images.min():.3f}/{images.max():.3f}")
            print(f"Masks min/max: {masks.min():.3f}/{masks.max():.3f}")
            
            # Check mask content
            for i in range(min(4, masks.shape[0])):
                mask = masks[i]
                positive_pixels = (mask > 0).sum().item()
                total_pixels = mask.numel()
                print(f"Sample {i+1}: {positive_pixels}/{total_pixels} positive pixels ({positive_pixels/total_pixels:.2%})")
            
            return images, masks
        else:
            print(f"❌ Unexpected batch structure: {batch}")
            return None, None
            
    except Exception as e:
        print(f"❌ Error in batch processing: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def visualize_batch(images, masks, save_path=None):
    """Visualize a batch of images and masks."""
    print("\n" + "=" * 80)
    print("VISUALIZING BATCH")
    print("=" * 80)
    
    if images is None or masks is None:
        print("❌ No batch data to visualize")
        return
    
    batch_size = min(4, images.shape[0])
    fig, axes = plt.subplots(2, batch_size, figsize=(4*batch_size, 8))
    
    if batch_size == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(batch_size):
        # Convert tensors to numpy for visualization
        img = images[i].permute(1, 2, 0).cpu().numpy()
        mask = masks[i].squeeze().cpu().numpy()
        
        # Normalize image for display (assuming it's in [0,1] or needs normalization)
        if img.max() <= 1.0:
            img_display = img
        else:
            img_display = img / 255.0
        
        # Display image
        axes[0, i].imshow(img_display)
        axes[0, i].set_title(f'Image {i+1}')
        axes[0, i].axis('off')
        
        # Display mask
        axes[1, i].imshow(mask, cmap='gray')
        axes[1, i].set_title(f'Mask {i+1}\nPositive: {(mask > 0).sum()}/{mask.size} ({(mask > 0).sum()/mask.size:.1%})')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def debug_normalization_impact(images, masks):
    """Debug the impact of normalization on the data."""
    print("\n" + "=" * 80)
    print("DEBUGGING NORMALIZATION IMPACT")
    print("=" * 80)
    
    if images is None:
        print("❌ No images to analyze")
        return
    
    # Check if ImageNet normalization is being applied
    from fastai.vision.all import imagenet_stats
    
    print(f"ImageNet stats: mean={imagenet_stats[0]}, std={imagenet_stats[1]}")
    print(f"Current images mean: {images.mean():.3f}")
    print(f"Current images std: {images.std():.3f}")
    
    # Check if images are in expected range
    if images.min() < -2 or images.max() > 2:
        print("⚠️  Images appear to be normalized (range outside [0,1])")
        print("   This could be problematic for medical images")
    else:
        print("✅ Images appear to be in [0,1] range")
    
    # Check mask values
    if masks is not None:
        print(f"Masks mean: {masks.mean():.3f}")
        print(f"Masks std: {masks.std():.3f}")
        print(f"Masks unique values: {torch.unique(masks)}")

def main():
    """Main debugging function."""
    print("🔍 COMPREHENSIVE DATABLOCK PIPELINE DEBUGGING")
    print("=" * 80)
    
    # You'll need to set this to your actual data path
    data_root = input("Enter path to your data directory: ").strip()
    if not data_root:
        print("❌ No data path provided")
        return
    
    # Step 1: Debug data structure
    if not debug_data_structure(data_root):
        print("❌ Data structure issues found. Please fix before continuing.")
        return
    
    # Step 2: Debug get_items function
    items = debug_get_items_function(data_root)
    if not items:
        print("❌ get_items function issues found. Please fix before continuing.")
        return
    
    # Step 3: Debug get_y function
    successful_masks = debug_get_y_function(items)
    if not successful_masks:
        print("❌ get_y function issues found. Please fix before continuing.")
        return
    
    # Step 4: Debug CropTransform
    if successful_masks:
        first_item, first_mask = successful_masks[0]
        debug_crop_transform(first_item, first_mask)
    
    # Step 5: Debug DataBlock creation
    dls = debug_datablock_creation(data_root)
    if dls is None:
        print("❌ DataBlock creation issues found. Please fix before continuing.")
        return
    
    # Step 6: Debug batch processing
    images, masks = debug_batch_processing(dls)
    
    # Step 7: Visualize results
    if images is not None and masks is not None:
        visualize_batch(images, masks, "debug_batch_visualization.png")
        
        # Step 8: Debug normalization
        debug_normalization_impact(images, masks)
    
    print("\n" + "=" * 80)
    print("DEBUGGING COMPLETE")
    print("=" * 80)
    print("Check the output above for any issues that might cause low dice scores.")
    print("Common issues to look for:")
    print("1. Mask loading failures")
    print("2. Incorrect mask values (not 0/1)")
    print("3. CropTransform issues")
    print("4. Normalization problems")
    print("5. Augmentation synchronization issues")

if __name__ == "__main__":
    main()
