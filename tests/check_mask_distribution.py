#!/usr/bin/env python3
"""
Check the distribution of empty vs non-empty masks in the dataset.
"""

import sys
from pathlib import Path

import numpy as np

from eq.utils.paths import get_runtime_cohort_path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

print("Checking mask distribution in glomerulus (preeclampsia) dataset...")

try:
    from fastai.vision.all import get_image_files

    from eq.data_management.standard_getters import get_y_full
    
    # Check the raw data for full images
    data_root = get_runtime_cohort_path("lauren_preeclampsia")
    images_dir = Path(data_root) / "images"
    
    if not images_dir.exists():
        print(f"❌ Images directory not found: {images_dir}")
        exit(1)
    
    print(f"Checking full images in: {images_dir}")
    items = list(get_image_files(images_dir))
    
    print(f"Total items: {len(items)}")
    
    empty_count = 0
    non_empty_count = 0
    positive_ratios = []
    
    for i, item in enumerate(items):
        try:
            mask = get_y_full(item)
            mask_array = np.array(mask)
            
            positive_pixels = (mask_array > 0).sum()
            total_pixels = mask_array.size
            positive_ratio = positive_pixels / total_pixels
            
            positive_ratios.append(positive_ratio)
            
            if positive_ratio == 0:
                empty_count += 1
            else:
                non_empty_count += 1
                
            if i < 10:  # Print details for first 10
                print(f"Item {i+1}: {Path(item).name} - positive ratio: {positive_ratio:.3f}")
                
        except Exception as e:
            print(f"Error processing {Path(item).name}: {e}")
    
    total_samples = empty_count + non_empty_count
    
    print("\n--- DISTRIBUTION SUMMARY ---")
    print(f"Empty masks: {empty_count} ({empty_count/total_samples:.1%})")
    print(f"Non-empty masks: {non_empty_count} ({non_empty_count/total_samples:.1%})")
    
    if positive_ratios:
        ratios_array = np.array(positive_ratios)
        print("\nPositive pixel ratio statistics:")
        print(f"  Mean: {ratios_array.mean():.3f}")
        print(f"  Median: {np.median(ratios_array):.3f}")
        print(f"  Std: {ratios_array.std():.3f}")
        print(f"  Min: {ratios_array.min():.3f}")
        print(f"  Max: {ratios_array.max():.3f}")
        
        # Distribution analysis
        print("\nDistribution of positive ratios:")
        print(f"  0% (empty): {(ratios_array == 0).sum()} masks")
        print(f"  0-1%: {((ratios_array > 0) & (ratios_array <= 0.01)).sum()} masks")
        print(f"  1-5%: {((ratios_array > 0.01) & (ratios_array <= 0.05)).sum()} masks")
        print(f"  5-10%: {((ratios_array > 0.05) & (ratios_array <= 0.1)).sum()} masks")
        print(f"  10%+: {(ratios_array > 0.1).sum()} masks")
    
    # Determine if balanced sampling is needed
    print("\n--- BALANCED SAMPLING ASSESSMENT ---")
    empty_ratio = empty_count / total_samples
    
    if empty_ratio > 0.8:
        print("❌ HEAVY IMBALANCE: >80% empty masks")
        print("   Balanced sampling would be beneficial")
        print("   Consider implementing it for dynamic patching")
    elif empty_ratio > 0.6:
        print("⚠️  MODERATE IMBALANCE: >60% empty masks")
        print("   Balanced sampling might help")
        print("   Monitor training performance")
    else:
        print("✅ REASONABLE BALANCE: <60% empty masks")
        print("   Balanced sampling probably not critical")
        print("   Standard training should work fine")
    
    # Calculate what the class weights would be
    if non_empty_count > 0 and empty_count > 0:
        n_classes = 2
        class_counts = [empty_count, non_empty_count]
        weights = [total_samples / (n_classes * count) for count in class_counts]
        print("\nIf using balanced sampling, class weights would be:")
        print(f"  Empty masks: {weights[0]:.3f}")
        print(f"  Non-empty masks: {weights[1]:.3f}")
        print(f"  Weight ratio (non-empty/empty): {weights[1]/weights[0]:.3f}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\nAnalysis complete.")
