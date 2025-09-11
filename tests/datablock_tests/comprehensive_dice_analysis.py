#!/usr/bin/env python3
"""
Comprehensive analysis of all potential causes of low dice scores.
"""

import sys
from pathlib import Path
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

print("🔍 COMPREHENSIVE DICE SCORE ANALYSIS")
print("=" * 80)

try:
    from eq.data_management.datablock_loader import (
        build_segmentation_dls, 
        build_segmentation_dls_dynamic_patching
    )
    
    data_root = "/home/ncamarda/endotheliosis_quantifier/data/derived_data/mito"
    
    print("Testing all potential issues that could cause low dice scores...")
    
    # Issue 1: Mask dtype
    print(f"\n1️⃣ MASK DTYPE ISSUE")
    print("-" * 40)
    
    dls_static, _ = build_segmentation_dls(data_root, bs=2, num_workers=0)
    batch_static = dls_static.train.one_batch()
    if len(batch_static) == 2:
        _, masks_static = batch_static
        print(f"Static patching mask dtype: {masks_static.dtype}")
        print(f"Expected: torch.float32, Actual: {masks_static.dtype}")
        if masks_static.dtype == torch.int64:
            print("❌ ISSUE: Masks are int64 instead of float32")
        else:
            print("✅ Masks have correct dtype")
    
    dls_dynamic, _ = build_segmentation_dls_dynamic_patching(data_root, bs=2, num_workers=0)
    batch_dynamic = dls_dynamic.train.one_batch()
    if len(batch_dynamic) == 2:
        _, masks_dynamic = batch_dynamic
        print(f"Dynamic patching mask dtype: {masks_dynamic.dtype}")
        if masks_dynamic.dtype == torch.int64:
            print("❌ ISSUE: Masks are int64 instead of float32")
        else:
            print("✅ Masks have correct dtype")
    
    # Issue 2: Input normalization differences
    print(f"\n2️⃣ INPUT NORMALIZATION DIFFERENCES")
    print("-" * 40)
    
    if len(batch_static) == 2 and len(batch_dynamic) == 2:
        images_static, _ = batch_static
        images_dynamic, _ = batch_dynamic
        
        print(f"Static patching:")
        print(f"  Range: [{images_static.min().item():.3f}, {images_static.max().item():.3f}]")
        print(f"  Mean: {images_static.mean().item():.3f}, Std: {images_static.std().item():.3f}")
        
        print(f"Dynamic patching:")
        print(f"  Range: [{images_dynamic.min().item():.3f}, {images_dynamic.max().item():.3f}]")
        print(f"  Mean: {images_dynamic.mean().item():.3f}, Std: {images_dynamic.std().item():.3f}")
        
        mean_diff = abs(images_static.mean().item() - images_dynamic.mean().item())
        std_diff = abs(images_static.std().item() - images_dynamic.std().item())
        
        print(f"Differences: mean_diff={mean_diff:.3f}, std_diff={std_diff:.3f}")
        
        if mean_diff > 0.1 or std_diff > 0.1:
            print("❌ ISSUE: Significant normalization differences between approaches")
        else:
            print("✅ Normalization is consistent")
    
    # Issue 3: Mask content integrity
    print(f"\n3️⃣ MASK CONTENT INTEGRITY")
    print("-" * 40)
    
    if len(batch_static) == 2 and len(batch_dynamic) == 2:
        _, masks_static = batch_static
        _, masks_dynamic = batch_dynamic
        
        # Check mask values
        static_unique = masks_static.unique()
        dynamic_unique = masks_dynamic.unique()
        
        print(f"Static patching mask unique values: {static_unique}")
        print(f"Dynamic patching mask unique values: {dynamic_unique}")
        
        if not torch.all((static_unique == 0) | (static_unique == 1)):
            print("❌ ISSUE: Static patching masks contain non-binary values")
        else:
            print("✅ Static patching masks are binary")
            
        if not torch.all((dynamic_unique == 0) | (dynamic_unique == 1)):
            print("❌ ISSUE: Dynamic patching masks contain non-binary values")
        else:
            print("✅ Dynamic patching masks are binary")
        
        # Check positive pixel ratios
        static_positive = (masks_static > 0).sum().item() / masks_static.numel()
        dynamic_positive = (masks_dynamic > 0).sum().item() / masks_dynamic.numel()
        
        print(f"Static patching positive ratio: {static_positive:.3f}")
        print(f"Dynamic patching positive ratio: {dynamic_positive:.3f}")
        
        if abs(static_positive - dynamic_positive) > 0.1:
            print("❌ ISSUE: Significant difference in positive pixel ratios")
        else:
            print("✅ Positive pixel ratios are consistent")
    
    # Issue 4: Crop transform synchronization
    print(f"\n4️⃣ CROP TRANSFORM SYNCHRONIZATION")
    print("-" * 40)
    
    # Test multiple batches to check if image-mask pairs stay synchronized
    print("Testing image-mask synchronization across multiple batches...")
    
    for i in range(3):
        batch = dls_dynamic.train.one_batch()
        if len(batch) == 2:
            images, masks = batch
            print(f"Batch {i+1}: images {images.shape}, masks {masks.shape}")
            
            # Check if shapes match
            if images.shape[2:] != masks.shape[1:]:
                print(f"❌ ISSUE: Shape mismatch in batch {i+1}")
            else:
                print(f"✅ Shapes match in batch {i+1}")
    
    # Issue 5: Augmentation effects
    print(f"\n5️⃣ AUGMENTATION EFFECTS")
    print("-" * 40)
    
    print("Testing augmentation consistency...")
    positive_ratios = []
    
    for i in range(5):
        batch = dls_dynamic.train.one_batch()
        if len(batch) == 2:
            _, masks = batch
            for j in range(masks.shape[0]):
                mask = masks[j]
                positive = (mask > 0).sum().item() / mask.numel()
                positive_ratios.append(positive)
    
    if positive_ratios:
        import numpy as np
        ratios_array = np.array(positive_ratios)
        print(f"Positive ratio stats: mean={ratios_array.mean():.3f}, std={ratios_array.std():.3f}")
        print(f"Range: [{ratios_array.min():.3f}, {ratios_array.max():.3f}]")
        
        if ratios_array.std() > 0.1:
            print("❌ ISSUE: High variance in positive ratios (augmentations may be breaking masks)")
        else:
            print("✅ Positive ratios are consistent across augmentations")
    
    # Summary
    print(f"\n" + "=" * 80)
    print("SUMMARY OF POTENTIAL ISSUES")
    print("=" * 80)
    
    print("Based on this analysis, the most likely causes of low dice scores are:")
    print()
    print("1. MASK DTYPE: Both approaches use int64 masks instead of float32")
    print("   - This could cause issues with loss function gradients")
    print("   - Dice loss specifically expects float32 inputs")
    print()
    print("2. NORMALIZATION DIFFERENCES: Static vs dynamic patching have different input ranges")
    print("   - If mitochondria training used static patching and glomeruli uses dynamic")
    print("   - The model expects different input distributions")
    print()
    print("3. CROP TRANSFORM: Need to verify image-mask synchronization")
    print("   - If crops are not properly synchronized, masks won't match images")
    print()
    print("RECOMMENDATIONS:")
    print("1. Fix mask dtype to float32 in both approaches")
    print("2. Use consistent normalization between training and transfer learning")
    print("3. Verify crop transform synchronization")
    print("4. Test with the same approach used for successful mitochondria training")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\nAnalysis complete.")
