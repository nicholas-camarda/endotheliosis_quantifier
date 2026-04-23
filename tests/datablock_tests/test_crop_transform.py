#!/usr/bin/env python3
"""
Test script specifically for CropTransform issues.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from eq.data_management.datablock_loader import CropTransform
from eq.data_management.standard_getters import get_y
from eq.core.constants import DEFAULT_IMAGE_SIZE
from fastai.vision.all import PILImage, PILMask

def check_crop_transform_with_real_data(data_root, num_tests=5):
    """Test CropTransform with real data to identify issues."""
    print("Testing CropTransform with real data...")
    
    # Get some real image-mask pairs
    from eq.data_management.datablock_loader import get_items_full_images
    items = get_items_full_images(Path(data_root))
    
    if not items:
        print("❌ No items found")
        return
    
    crop_transform = CropTransform(size=DEFAULT_IMAGE_SIZE)
    
    for i, item in enumerate(items[:num_tests]):
        print(f"\n--- Test {i+1}: {Path(item).name} ---")
        
        try:
            # Load image and mask
            img = PILImage.create(item)
            mask = get_y(item)
            
            print(f"Original image size: {img.size}")
            print(f"Original mask size: {mask.size}")
            
            # Check if sizes match
            if img.size != mask.size:
                print(f"⚠️  Size mismatch: image {img.size} vs mask {mask.size}")
            
            # Test crop transform
            result = crop_transform.encodes((img, mask))
            
            if isinstance(result, tuple) and len(result) == 2:
                cropped_img, cropped_mask = result
                img_size = cropped_img.size if hasattr(cropped_img, 'size') else cropped_img.shape
                mask_size = cropped_mask.size if hasattr(cropped_mask, 'size') else cropped_mask.shape
                print(f"Cropped image size: {img_size}")
                print(f"Cropped mask size: {mask_size}")
                
                # Check if sizes still match after cropping
                if hasattr(cropped_img, 'size') and hasattr(cropped_mask, 'size'):
                    if cropped_img.size != cropped_mask.size:
                        print(f"❌ Size mismatch after crop: image {cropped_img.size} vs mask {cropped_mask.size}")
                    else:
                        print("✅ Sizes match after crop")
                
                # Check mask content
                mask_array = np.array(cropped_mask)
                positive_pixels = (mask_array > 0).sum()
                print(f"Positive pixels: {positive_pixels}/{mask_array.size} ({positive_pixels/mask_array.size:.2%})")
                
                # Visualize if this is the first test
                if i == 0:
                    visualize_crop_test(img, mask, cropped_img, cropped_mask, f"crop_test_{i+1}.png")
                
            else:
                print(f"❌ Unexpected result type: {type(result)}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

def visualize_crop_test(orig_img, orig_mask, crop_img, crop_mask, save_path):
    """Visualize the crop transform result."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original image
    axes[0, 0].imshow(orig_img)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Original mask
    orig_mask_array = np.array(orig_mask)
    axes[0, 1].imshow(orig_mask_array, cmap='gray')
    axes[0, 1].set_title(f'Original Mask\nPositive: {(orig_mask_array > 0).sum()}/{orig_mask_array.size} ({(orig_mask_array > 0).sum()/orig_mask_array.size:.1%})')
    axes[0, 1].axis('off')
    
    # Cropped image
    axes[1, 0].imshow(crop_img)
    axes[1, 0].set_title('Cropped Image')
    axes[1, 0].axis('off')
    
    # Cropped mask
    crop_mask_array = np.array(crop_mask)
    axes[1, 1].imshow(crop_mask_array, cmap='gray')
    axes[1, 1].set_title(f'Cropped Mask\nPositive: {(crop_mask_array > 0).sum()}/{crop_mask_array.size} ({(crop_mask_array > 0).sum()/crop_mask_array.size:.1%})')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Visualization saved to {save_path}")
    plt.close()

def test_crop_transform_edge_cases():
    """Test CropTransform with edge cases."""
    print("\nTesting CropTransform edge cases...")
    
    crop_transform = CropTransform(size=256)
    
    # Test 1: Image smaller than crop size
    print("\n--- Test: Image smaller than crop size ---")
    small_img = PILImage.create(np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8))
    small_mask = PILMask.create(np.random.randint(0, 2, (128, 128), dtype=np.uint8))
    
    result = crop_transform.encodes((small_img, small_mask))
    if isinstance(result, tuple):
        crop_img, crop_mask = result
        img_size = crop_img.size if hasattr(crop_img, 'size') else crop_img.shape
        mask_size = crop_mask.size if hasattr(crop_mask, 'size') else crop_mask.shape
        print(f"Small image crop result: {img_size}")
        print(f"Small mask crop result: {mask_size}")
    
    # Test 2: Very large image
    print("\n--- Test: Very large image ---")
    large_img = PILImage.create(np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8))
    large_mask = PILMask.create(np.random.randint(0, 2, (1024, 1024), dtype=np.uint8))
    
    result = crop_transform.encodes((large_img, large_mask))
    if isinstance(result, tuple):
        crop_img, crop_mask = result
        img_size = crop_img.size if hasattr(crop_img, 'size') else crop_img.shape
        mask_size = crop_mask.size if hasattr(crop_mask, 'size') else crop_mask.shape
        print(f"Large image crop result: {img_size}")
        print(f"Large mask crop result: {mask_size}")

def main():
    """Main test function."""
    print("🔍 CROP TRANSFORM TESTING")
    print("=" * 50)
    
    # Test edge cases first
    test_crop_transform_edge_cases()
    
    # Test with real data if path provided
    data_root = input("\nEnter path to your data directory (or press Enter to skip): ").strip()
    if data_root and Path(data_root).exists():
        check_crop_transform_with_real_data(data_root)
    else:
        print("Skipping real data tests")

if __name__ == "__main__":
    main()
