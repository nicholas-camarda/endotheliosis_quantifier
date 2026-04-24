#!/usr/bin/env python3
"""
Test dynamic patching datablock specifically.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

print("Testing dynamic patching datablock...")

try:
    from eq.data_management.datablock_loader import build_segmentation_dls_dynamic_patching
    print("✅ Imported build_segmentation_dls_dynamic_patching")
    
    data_root = str(Path.home() / "ProjectsRuntime" / "endotheliosis_quantifier" / "raw_data" / "mitochondria_data" / "training")
    print(f"Creating DataLoaders with data root: {data_root}")
    
    # Create DataLoaders
    dls, weighted_loss = build_segmentation_dls_dynamic_patching(
        data_root, 
        bs=2, 
        num_workers=0
    )
    print("✅ Created DataLoaders")
    
    print(f"Train dataset size: {len(dls.train_ds)}")
    print(f"Valid dataset size: {len(dls.valid_ds)}")
    
    # Test batch processing
    print("\nTesting batch processing...")
    batch = dls.train.one_batch()
    print(f"✅ Got batch: {type(batch)}")
    
    if len(batch) == 2:
        images, masks = batch
        print(f"Images shape: {images.shape}, dtype: {images.dtype}")
        print(f"Masks shape: {masks.shape}, dtype: {masks.dtype}")
        print(f"Images range: {images.min().item():.3f} to {images.max().item():.3f}")
        print(f"Masks range: {masks.min().item():.3f} to {masks.max().item():.3f}")
        
        # Check mask content
        for i in range(min(2, masks.shape[0])):
            mask = masks[i]
            positive = (mask > 0).sum().item()
            total = mask.numel()
            print(f"Sample {i+1}: {positive}/{total} positive pixels ({positive/total:.2%})")
        
        # Check for normalization issues
        print(f"\nChecking normalization...")
        from fastai.vision.all import imagenet_stats
        imagenet_mean, imagenet_std = imagenet_stats
        print(f"ImageNet stats: mean={imagenet_mean}, std={imagenet_std}")
        
        current_mean = images.mean(dim=[0, 2, 3])
        current_std = images.std(dim=[0, 2, 3])
        print(f"Current image stats: mean={current_mean}, std={current_std}")
        
        import torch
        mean_diff = torch.abs(current_mean - torch.tensor(imagenet_mean, device=current_mean.device)).mean()
        std_diff = torch.abs(current_std - torch.tensor(imagenet_std, device=current_std.device)).mean()
        
        print(f"Difference from ImageNet: mean_diff={mean_diff.item():.3f}, std_diff={std_diff.item():.3f}")
        
        if mean_diff < 0.1 and std_diff < 0.1:
            print("⚠️  Images appear to be normalized to ImageNet stats!")
            print("   This could be problematic for medical images and transfer learning.")
        else:
            print("✅ Images are NOT normalized to ImageNet stats")
            print("   This is correct for medical images.")
    
    # Test multiple batches to check consistency
    print(f"\nTesting multiple batches for consistency...")
    for i in range(3):
        batch = dls.train.one_batch()
        if len(batch) == 2:
            images, masks = batch
            mask_ratios = []
            for j in range(masks.shape[0]):
                mask = masks[j]
                positive = (mask > 0).sum().item()
                total = mask.numel()
                ratio = positive / total
                mask_ratios.append(ratio)
            print(f"Batch {i+1}: positive ratios = {[f'{r:.3f}' for r in mask_ratios]}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("Dynamic patching test complete.")
