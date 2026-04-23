#!/usr/bin/env python3
"""
Test the normalization mismatch hypothesis.
"""

import sys
from pathlib import Path

import pytest

pytest.skip(
    "Legacy static-vs-dynamic analysis script; static patches are not supported training inputs.",
    allow_module_level=True,
)

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

print("Testing normalization mismatch hypothesis...")

try:
    from eq.data_management.datablock_loader import (
        build_segmentation_dls, 
        build_segmentation_dls_dynamic_patching
    )
    
    data_root = "/home/ncamarda/endotheliosis_quantifier/data/derived_data/mito"
    
    # Test 1: Static patching (with ImageNet normalization)
    print("\n--- STATIC PATCHING (with ImageNet normalization) ---")
    dls_static, _ = build_segmentation_dls(data_root, bs=2, num_workers=0)
    batch_static = dls_static.train.one_batch()
    
    if len(batch_static) == 2:
        images_static, masks_static = batch_static
        print(f"Images shape: {images_static.shape}, dtype: {images_static.dtype}")
        print(f"Images range: {images_static.min().item():.3f} to {images_static.max().item():.3f}")
        print(f"Images mean: {images_static.mean().item():.3f}")
        print(f"Images std: {images_static.std().item():.3f}")
        
        # Per-channel statistics
        mean_per_channel = images_static.mean(dim=[0, 2, 3])
        std_per_channel = images_static.std(dim=[0, 2, 3])
        print(f"Per-channel mean: {mean_per_channel}")
        print(f"Per-channel std: {std_per_channel}")
        
        print(f"Masks dtype: {masks_static.dtype}")
        print(f"Masks range: {masks_static.min().item():.3f} to {masks_static.max().item():.3f}")
    
    # Test 2: Dynamic patching (without ImageNet normalization)
    print("\n--- DYNAMIC PATCHING (without ImageNet normalization) ---")
    dls_dynamic, _ = build_segmentation_dls_dynamic_patching(data_root, bs=2, num_workers=0)
    batch_dynamic = dls_dynamic.train.one_batch()
    
    if len(batch_dynamic) == 2:
        images_dynamic, masks_dynamic = batch_dynamic
        print(f"Images shape: {images_dynamic.shape}, dtype: {images_dynamic.dtype}")
        print(f"Images range: {images_dynamic.min().item():.3f} to {images_dynamic.max().item():.3f}")
        print(f"Images mean: {images_dynamic.mean().item():.3f}")
        print(f"Images std: {images_dynamic.std().item():.3f}")
        
        # Per-channel statistics
        mean_per_channel = images_dynamic.mean(dim=[0, 2, 3])
        std_per_channel = images_dynamic.std(dim=[0, 2, 3])
        print(f"Per-channel mean: {mean_per_channel}")
        print(f"Per-channel std: {std_per_channel}")
        
        print(f"Masks dtype: {masks_dynamic.dtype}")
        print(f"Masks range: {masks_dynamic.min().item():.3f} to {masks_dynamic.max().item():.3f}")
    
    # Compare the differences
    print(f"\n--- COMPARISON ---")
    if len(batch_static) == 2 and len(batch_dynamic) == 2:
        images_static, _ = batch_static
        images_dynamic, _ = batch_dynamic
        
        print(f"Static images mean: {images_static.mean().item():.3f}")
        print(f"Dynamic images mean: {images_dynamic.mean().item():.3f}")
        print(f"Mean difference: {abs(images_static.mean().item() - images_dynamic.mean().item()):.3f}")
        
        print(f"Static images std: {images_static.std().item():.3f}")
        print(f"Dynamic images std: {images_dynamic.std().item():.3f}")
        print(f"Std difference: {abs(images_static.std().item() - images_dynamic.std().item()):.3f}")
        
        # Check if static is close to ImageNet stats
        from fastai.vision.all import imagenet_stats
        imagenet_mean, imagenet_std = imagenet_stats
        print(f"\nImageNet stats: mean={imagenet_mean}, std={imagenet_std}")
        
        static_mean_per_channel = images_static.mean(dim=[0, 2, 3])
        static_std_per_channel = images_static.std(dim=[0, 2, 3])
        
        import torch
        mean_diff = torch.abs(static_mean_per_channel - torch.tensor(imagenet_mean, device=static_mean_per_channel.device)).mean()
        std_diff = torch.abs(static_std_per_channel - torch.tensor(imagenet_std, device=static_std_per_channel.device)).mean()
        
        print(f"Static vs ImageNet - mean diff: {mean_diff.item():.3f}, std diff: {std_diff.item():.3f}")
        
        if mean_diff < 0.1 and std_diff < 0.1:
            print("✅ Static patching IS using ImageNet normalization")
        else:
            print("❌ Static patching is NOT using ImageNet normalization")
        
        # Check if dynamic is in [0,1] range
        if 0.0 <= images_dynamic.min().item() and images_dynamic.max().item() <= 1.0:
            print("✅ Dynamic patching images are in [0,1] range (no normalization)")
        else:
            print("❌ Dynamic patching images are NOT in [0,1] range")
    
    print(f"\n--- HYPOTHESIS ---")
    print("If mitochondria training used static patching (ImageNet normalization) and")
    print("glomeruli transfer learning uses dynamic patching (no normalization),")
    print("then the model expects ImageNet-normalized inputs but receives [0,1] inputs.")
    print("This would cause poor transfer learning performance!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("Normalization mismatch test complete.")
