#!/usr/bin/env python3
"""
Comprehensive analysis script to identify issues causing low dice scores.

This script performs detailed analysis of the datablock pipeline to identify
potential issues that could cause low dice scores during transfer learning.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from collections import defaultdict

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from eq.data_management.datablock_loader import (
    build_segmentation_dls_dynamic_patching,
    build_segmentation_dls
)
from eq.data_management.standard_getters import get_y
from eq.core.constants import DEFAULT_IMAGE_SIZE, DEFAULT_MASK_THRESHOLD

def analyze_mask_distribution(data_root, sample_size=100):
    """Analyze the distribution of mask values and positive pixel ratios."""
    print("=" * 80)
    print("ANALYZING MASK DISTRIBUTION")
    print("=" * 80)
    
    from eq.data_management.datablock_loader import get_items_full_images
    items = get_items_full_images(Path(data_root))
    
    if not items:
        print("❌ No items found")
        return
    
    # Sample items for analysis
    sample_items = items[:min(sample_size, len(items))]
    
    mask_stats = {
        'total_masks': 0,
        'empty_masks': 0,
        'positive_pixel_ratios': [],
        'mask_values': [],
        'mask_shapes': [],
        'mask_dtypes': []
    }
    
    for i, item in enumerate(sample_items):
        try:
            mask = get_y(item)
            mask_array = np.array(mask)
            
            mask_stats['total_masks'] += 1
            mask_stats['mask_shapes'].append(mask_array.shape)
            mask_stats['mask_dtypes'].append(mask_array.dtype)
            mask_stats['mask_values'].extend(np.unique(mask_array).tolist())
            
            positive_pixels = (mask_array > 0).sum()
            total_pixels = mask_array.size
            positive_ratio = positive_pixels / total_pixels
            
            mask_stats['positive_pixel_ratios'].append(positive_ratio)
            
            if positive_ratio == 0:
                mask_stats['empty_masks'] += 1
            
            if i < 10:  # Print details for first 10
                print(f"Mask {i+1}: {Path(item).name}")
                print(f"  Shape: {mask_array.shape}, Dtype: {mask_array.dtype}")
                print(f"  Unique values: {np.unique(mask_array)}")
                print(f"  Positive ratio: {positive_ratio:.3f}")
                
        except Exception as e:
            print(f"❌ Error processing {Path(item).name}: {e}")
    
    # Summary statistics
    print(f"\n--- SUMMARY STATISTICS ---")
    print(f"Total masks analyzed: {mask_stats['total_masks']}")
    print(f"Empty masks: {mask_stats['empty_masks']} ({mask_stats['empty_masks']/mask_stats['total_masks']:.1%})")
    
    if mask_stats['positive_pixel_ratios']:
        ratios = np.array(mask_stats['positive_pixel_ratios'])
        print(f"Positive pixel ratio - Mean: {ratios.mean():.3f}, Std: {ratios.std():.3f}")
        print(f"Positive pixel ratio - Min: {ratios.min():.3f}, Max: {ratios.max():.3f}")
        print(f"Positive pixel ratio - Median: {np.median(ratios):.3f}")
        
        # Distribution analysis
        print(f"\nPositive pixel ratio distribution:")
        print(f"  0% (empty): {(ratios == 0).sum()} masks")
        print(f"  0-1%: {((ratios > 0) & (ratios <= 0.01)).sum()} masks")
        print(f"  1-5%: {((ratios > 0.01) & (ratios <= 0.05)).sum()} masks")
        print(f"  5-10%: {((ratios > 0.05) & (ratios <= 0.1)).sum()} masks")
        print(f"  10%+: {(ratios > 0.1).sum()} masks")
    
    # Check for data quality issues
    print(f"\n--- DATA QUALITY CHECKS ---")
    
    # Check for consistent shapes
    unique_shapes = set(mask_stats['mask_shapes'])
    if len(unique_shapes) > 1:
        print(f"⚠️  Multiple mask shapes found: {unique_shapes}")
    else:
        print(f"✅ Consistent mask shapes: {unique_shapes}")
    
    # Check for consistent dtypes
    unique_dtypes = set(mask_stats['mask_dtypes'])
    if len(unique_dtypes) > 1:
        print(f"⚠️  Multiple mask dtypes found: {unique_dtypes}")
    else:
        print(f"✅ Consistent mask dtypes: {unique_dtypes}")
    
    # Check mask values
    unique_values = set(mask_stats['mask_values'])
    print(f"All unique mask values found: {sorted(unique_values)}")
    
    if not all(v in [0.0, 1.0] for v in unique_values):
        print(f"⚠️  Non-binary mask values found! This could cause issues.")
    else:
        print(f"✅ All masks are binary (0.0, 1.0)")
    
    return mask_stats

def compare_static_vs_dynamic_patching(data_root):
    """Compare static vs dynamic patching approaches."""
    print("\n" + "=" * 80)
    print("COMPARING STATIC VS DYNAMIC PATCHING")
    print("=" * 80)
    
    try:
        # Test static patching
        print("Testing static patching...")
        dls_static, _ = build_segmentation_dls(data_root, bs=2, num_workers=0)
        
        # Test dynamic patching
        print("Testing dynamic patching...")
        dls_dynamic, _ = build_segmentation_dls_dynamic_patching(data_root, bs=2, num_workers=0)
        
        # Compare batch processing
        print("\n--- STATIC PATCHING BATCH ---")
        try:
            batch_static = dls_static.train.one_batch()
            if len(batch_static) == 2:
                img_static, mask_static = batch_static
                print(f"Images shape: {img_static.shape}, dtype: {img_static.dtype}")
                print(f"Masks shape: {mask_static.shape}, dtype: {mask_static.dtype}")
                print(f"Images range: {img_static.min():.3f} to {img_static.max():.3f}")
                print(f"Masks range: {mask_static.min():.3f} to {mask_static.max():.3f}")
                
                # Check mask content
                for i in range(min(2, mask_static.shape[0])):
                    mask = mask_static[i]
                    positive = (mask > 0).sum().item()
                    total = mask.numel()
                    print(f"Sample {i+1}: {positive}/{total} positive pixels ({positive/total:.2%})")
        except Exception as e:
            print(f"❌ Error with static patching: {e}")
        
        print("\n--- DYNAMIC PATCHING BATCH ---")
        try:
            batch_dynamic = dls_dynamic.train.one_batch()
            if len(batch_dynamic) == 2:
                img_dynamic, mask_dynamic = batch_dynamic
                print(f"Images shape: {img_dynamic.shape}, dtype: {img_dynamic.dtype}")
                print(f"Masks shape: {mask_dynamic.shape}, dtype: {mask_dynamic.dtype}")
                print(f"Images range: {img_dynamic.min():.3f} to {img_dynamic.max():.3f}")
                print(f"Masks range: {mask_dynamic.min():.3f} to {mask_dynamic.max():.3f}")
                
                # Check mask content
                for i in range(min(2, mask_dynamic.shape[0])):
                    mask = mask_dynamic[i]
                    positive = (mask > 0).sum().item()
                    total = mask.numel()
                    print(f"Sample {i+1}: {positive}/{total} positive pixels ({positive/total:.2%})")
        except Exception as e:
            print(f"❌ Error with dynamic patching: {e}")
        
        return dls_static, dls_dynamic
        
    except Exception as e:
        print(f"❌ Error comparing approaches: {e}")
        return None, None

def analyze_augmentation_impact(dls):
    """Analyze the impact of augmentations on mask integrity."""
    print("\n" + "=" * 80)
    print("ANALYZING AUGMENTATION IMPACT")
    print("=" * 80)
    
    if dls is None:
        print("❌ No DataLoaders provided")
        return
    
    try:
        # Get multiple batches to see augmentation effects
        print("Analyzing augmentation effects across multiple batches...")
        
        batch_stats = []
        for batch_idx in range(5):
            batch = dls.train.one_batch()
            if len(batch) == 2:
                images, masks = batch
                
                batch_stat = {
                    'batch_idx': batch_idx,
                    'images_shape': images.shape,
                    'masks_shape': masks.shape,
                    'images_range': (images.min().item(), images.max().item()),
                    'masks_range': (masks.min().item(), masks.max().item()),
                    'positive_pixel_ratios': []
                }
                
                for i in range(masks.shape[0]):
                    mask = masks[i]
                    positive = (mask > 0).sum().item()
                    total = mask.numel()
                    ratio = positive / total
                    batch_stat['positive_pixel_ratios'].append(ratio)
                
                batch_stats.append(batch_stat)
                
                print(f"Batch {batch_idx + 1}:")
                print(f"  Images: {images.shape}, range {batch_stat['images_range']}")
                print(f"  Masks: {masks.shape}, range {batch_stat['masks_range']}")
                print(f"  Positive ratios: {[f'{r:.3f}' for r in batch_stat['positive_pixel_ratios']]}")
        
        # Check for consistency
        print(f"\n--- AUGMENTATION CONSISTENCY CHECKS ---")
        
        # Check if mask ranges are consistent
        mask_ranges = [stat['masks_range'] for stat in batch_stats]
        if all(r == mask_ranges[0] for r in mask_ranges):
            print(f"✅ Consistent mask ranges: {mask_ranges[0]}")
        else:
            print(f"⚠️  Inconsistent mask ranges: {mask_ranges}")
        
        # Check if positive pixel ratios are reasonable
        all_ratios = []
        for stat in batch_stats:
            all_ratios.extend(stat['positive_pixel_ratios'])
        
        if all_ratios:
            ratios_array = np.array(all_ratios)
            print(f"Positive pixel ratio statistics across batches:")
            print(f"  Mean: {ratios_array.mean():.3f}")
            print(f"  Std: {ratios_array.std():.3f}")
            print(f"  Min: {ratios_array.min():.3f}")
            print(f"  Max: {ratios_array.max():.3f}")
            
            # Check for problematic ratios
            empty_ratio = (ratios_array == 0).sum() / len(ratios_array)
            if empty_ratio > 0.5:
                print(f"⚠️  High empty mask ratio: {empty_ratio:.1%}")
            else:
                print(f"✅ Reasonable empty mask ratio: {empty_ratio:.1%}")
        
    except Exception as e:
        print(f"❌ Error analyzing augmentations: {e}")
        import traceback
        traceback.print_exc()

def check_normalization_issues(dls):
    """Check for normalization issues that could affect transfer learning."""
    print("\n" + "=" * 80)
    print("CHECKING NORMALIZATION ISSUES")
    print("=" * 80)
    
    if dls is None:
        print("❌ No DataLoaders provided")
        return
    
    try:
        batch = dls.train.one_batch()
        if len(batch) == 2:
            images, masks = batch
            
            print(f"Image statistics:")
            print(f"  Shape: {images.shape}")
            print(f"  Dtype: {images.dtype}")
            print(f"  Min/Max: {images.min():.3f} / {images.max():.3f}")
            print(f"  Mean/Std: {images.mean():.3f} / {images.std():.3f}")
            
            # Check if images are normalized to ImageNet stats
            from fastai.vision.all import imagenet_stats
            imagenet_mean, imagenet_std = imagenet_stats
            
            print(f"\nImageNet normalization stats:")
            print(f"  Mean: {imagenet_mean}")
            print(f"  Std: {imagenet_std}")
            
            # Check if current images match ImageNet normalization
            current_mean = images.mean(dim=[0, 2, 3])  # Mean per channel
            current_std = images.std(dim=[0, 2, 3])   # Std per channel
            
            print(f"\nCurrent image statistics (per channel):")
            print(f"  Mean: {current_mean}")
            print(f"  Std: {current_std}")
            
            # Check if close to ImageNet stats
            mean_diff = torch.abs(current_mean - torch.tensor(imagenet_mean)).mean()
            std_diff = torch.abs(current_std - torch.tensor(imagenet_std)).mean()
            
            print(f"\nDifference from ImageNet stats:")
            print(f"  Mean difference: {mean_diff:.3f}")
            print(f"  Std difference: {std_diff:.3f}")
            
            if mean_diff < 0.1 and std_diff < 0.1:
                print("✅ Images appear to be normalized to ImageNet stats")
                print("⚠️  This could be problematic for medical images!")
            else:
                print("✅ Images are NOT normalized to ImageNet stats")
                print("✅ This is correct for medical images")
            
            # Check mask statistics
            print(f"\nMask statistics:")
            print(f"  Shape: {masks.shape}")
            print(f"  Dtype: {masks.dtype}")
            print(f"  Min/Max: {masks.min():.3f} / {masks.max():.3f}")
            print(f"  Mean/Std: {masks.mean():.3f} / {masks.std():.3f}")
            
            # Check if masks are binary
            unique_values = torch.unique(masks)
            print(f"  Unique values: {unique_values}")
            
            if torch.all((unique_values == 0.0) | (unique_values == 1.0)):
                print("✅ Masks are properly binary")
            else:
                print("⚠️  Masks contain non-binary values!")
                
    except Exception as e:
        print(f"❌ Error checking normalization: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main analysis function."""
    print("🔍 COMPREHENSIVE DICE SCORE ISSUE ANALYSIS")
    print("=" * 80)
    
    data_root = input("Enter path to your data directory: ").strip()
    if not data_root or not Path(data_root).exists():
        print("❌ Invalid data path")
        return
    
    # Step 1: Analyze mask distribution
    mask_stats = analyze_mask_distribution(data_root)
    
    # Step 2: Compare static vs dynamic patching
    dls_static, dls_dynamic = compare_static_vs_dynamic_patching(data_root)
    
    # Step 3: Analyze augmentation impact
    if dls_dynamic is not None:
        analyze_augmentation_impact(dls_dynamic)
    
    # Step 4: Check normalization issues
    if dls_dynamic is not None:
        check_normalization_issues(dls_dynamic)
    
    # Step 5: Provide recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    print("Based on the analysis above, here are potential issues and recommendations:")
    print()
    print("1. MASK DISTRIBUTION ISSUES:")
    if mask_stats and mask_stats['empty_masks'] / mask_stats['total_masks'] > 0.3:
        print("   ⚠️  High empty mask ratio detected - consider balanced sampling")
    print("   ✅ Check that masks are properly binary (0.0, 1.0)")
    print()
    print("2. NORMALIZATION ISSUES:")
    print("   ⚠️  If using ImageNet normalization, this could hurt medical image performance")
    print("   ✅ Consider removing ImageNet normalization for medical images")
    print()
    print("3. CROP TRANSFORM ISSUES:")
    print("   ⚠️  Check that CropTransform maintains image-mask synchronization")
    print("   ✅ Verify that cropped images and masks have matching sizes")
    print()
    print("4. AUGMENTATION ISSUES:")
    print("   ⚠️  Ensure augmentations don't break mask integrity")
    print("   ✅ Check that positive pixel ratios remain reasonable after augmentation")
    print()
    print("5. TRANSFER LEARNING ISSUES:")
    print("   ⚠️  If training from scratch worked but transfer learning doesn't:")
    print("   - Check that the model architecture is compatible")
    print("   - Verify that input preprocessing is identical")
    print("   - Consider using a different learning rate for transfer learning")
    print("   - Check that the pretrained model was saved/loaded correctly")

if __name__ == "__main__":
    main()
