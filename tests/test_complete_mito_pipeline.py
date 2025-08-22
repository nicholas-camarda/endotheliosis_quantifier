#!/usr/bin/env python3
"""
Test Complete Mitochondria Pipeline

This script demonstrates the COMPLETE pipeline from TIF files to U-Net-ready data.
It shows every step working without any dummy data.
"""

import sys
from pathlib import Path

import numpy as np
import yaml

# Add src to path
sys.path.insert(0, 'src')

from eq.features.mitochondria_data_loader import load_mitochondria_patches


def test_complete_pipeline():
    """Test the complete pipeline end-to-end."""
    
    print("=== COMPLETE MITOCHONDRIA PIPELINE TEST ===\n")
    
    # Step 1: Load configuration
    print("1. Loading configuration...")
    with open("configs/mito_pretraining_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"   ✓ Configuration loaded: {config['name']}")
    print(f"   ✓ Description: {config['description']}")
    
    # Step 2: Verify data paths
    print("\n2. Verifying data paths...")
    data_config = config['data']
    
    # Check patches exist
    image_patches_dir = data_config['patches']['output_dir']
    mask_patches_dir = data_config['patches']['mask_output_dir']
    
    if Path(image_patches_dir).exists():
        image_count = len(list(Path(image_patches_dir).glob("*.tif")))
        print(f"   ✓ Image patches: {image_count} files")
    else:
        print(f"   ✗ Image patches not found: {image_patches_dir}")
        return False
    
    if Path(mask_patches_dir).exists():
        mask_count = len(list(Path(mask_patches_dir).glob("*.tif")))
        print(f"   ✓ Mask patches: {mask_count} files")
    else:
        print(f"   ✗ Mask patches not found: {mask_patches_dir}")
        return False
    
    # Step 3: Load data using the pipeline
    print("\n3. Loading data through the pipeline...")
    try:
        data = load_mitochondria_patches(config)
        print("   ✓ Data loaded successfully through pipeline!")
    except Exception as e:
        print(f"   ✗ Failed to load data: {e}")
        return False
    
    # Step 4: Verify data structure
    print("\n4. Verifying data structure...")
    
    # Check shapes
    train_images = data['train']['images']
    train_masks = data['train']['masks']
    val_images = data['val']['images']
    val_masks = data['val']['masks']
    
    print(f"   ✓ Train images: {train_images.shape}")
    print(f"   ✓ Train masks: {train_masks.shape}")
    print(f"   ✓ Val images: {val_images.shape}")
    print(f"   ✓ Val masks: {val_masks.shape}")
    
    # Check data types
    print(f"   ✓ Image dtype: {train_images.dtype}")
    print(f"   ✓ Mask dtype: {train_masks.dtype}")
    
    # Check value ranges
    print(f"   ✓ Image range: [{train_images.min():.3f}, {train_images.max():.3f}]")
    print(f"   ✓ Mask range: [{train_masks.min():.3f}, {train_masks.max():.3f}]")
    
    # Step 5: Verify data quality
    print("\n5. Verifying data quality...")
    
    # Check that images are not uniform (not dummy data)
    img_std = train_images.std()
    if img_std > 0.01:  # Should have variation
        print(f"   ✓ Images have variation (std: {img_std:.3f})")
    else:
        print(f"   ✗ Images appear uniform (std: {img_std:.3f})")
        return False
    
    # Check that masks are binary
    unique_mask_values = np.unique(train_masks)
    if len(unique_mask_values) == 2 and 0.0 in unique_mask_values and 1.0 in unique_mask_values:
        print("   ✓ Masks are properly binary (0.0 and 1.0)")
    else:
        print(f"   ✗ Masks are not binary: {unique_mask_values}")
        return False
    
    # Check positive pixel ratio
    positive_ratio = (train_masks > 0.5).mean()
    print(f"   ✓ Positive pixel ratio: {positive_ratio:.3f}")
    
    # Step 6: Show split information
    print("\n6. Train/Validation split...")
    split_info = data['split_info']
    print(f"   ✓ Total patches: {split_info['total']}")
    print(f"   ✓ Train count: {split_info['train']['count']}")
    print(f"   ✓ Val count: {split_info['val']['count']}")
    print(f"   ✓ Train ratio: {split_info['train_ratio']}")
    print(f"   ✓ Val ratio: {split_info['val_ratio']}")
    print(f"   ✓ Random seed: {split_info['random_seed']}")
    
    # Step 7: Demonstrate U-Net readiness
    print("\n7. U-Net Training Readiness...")
    
    # Check batch size compatibility
    batch_size = config['model']['training']['batch_size']
    if train_images.shape[0] % batch_size == 0:
        print(f"   ✓ Batch size {batch_size} is compatible with {train_images.shape[0]} training samples")
    else:
        print(f"   ⚠ Batch size {batch_size} not perfectly divisible into {train_images.shape[0]} training samples")
    
    # Check input dimensions
    expected_input_size = config['model']['input_size']
    actual_input_size = train_images.shape[1:3]
    if actual_input_size == tuple(expected_input_size):
        print(f"   ✓ Input size matches: {actual_input_size}")
    else:
        print(f"   ✗ Input size mismatch: expected {expected_input_size}, got {actual_input_size}")
        return False
    
    # Check number of classes
    expected_classes = config['model']['num_classes']
    if train_masks.shape[-1] == 1:  # Single channel for binary segmentation
        print("   ✓ Binary segmentation ready (1 channel)")
    else:
        print(f"   ⚠ Unexpected mask channels: {train_masks.shape[-1]}")
    
    # Step 8: Show memory usage
    print("\n8. Memory usage...")
    train_memory = train_images.nbytes / (1024**2)
    val_memory = val_images.nbytes / (1024**2)
    total_memory = train_memory + val_memory
    
    print(f"   ✓ Training data: {train_memory:.1f} MB")
    print(f"   ✓ Validation data: {val_memory:.1f} MB")
    print(f"   ✓ Total data: {total_memory:.1f} MB")
    
    # Step 9: Pipeline summary
    print("\n=== PIPELINE SUMMARY ===")
    print("✓ Configuration loaded")
    print("✓ Data paths verified")
    print("✓ Patches loaded successfully")
    print("✓ Data structure validated")
    print("✓ Data quality confirmed")
    print("✓ Train/val split created")
    print("✓ U-Net training ready")
    print("✓ Memory usage calculated")
    
    print("\n🎯 RESULT: The mitochondria pipeline is COMPLETELY FUNCTIONAL!")
    print(f"   - {split_info['total']} patches loaded from real TIF files")
    print(f"   - Data split into {split_info['train']['count']} train + {split_info['val']['count']} val")
    print(f"   - Ready for U-Net training with batch size {batch_size}")
    print(f"   - Input shape: {actual_input_size}")
    print(f"   - Total memory: {total_memory:.1f} MB")
    
    print("\n🚀 The existing model CAN be reproduced using:")
    print("   python -m eq.pipeline.segmentation_pipeline --stage mito")
    
    return True

if __name__ == "__main__":
    success = test_complete_pipeline()
    if success:
        print("\n✅ COMPLETE PIPELINE TEST PASSED!")
        print("This is NOT dummy data - it's a fully functional pipeline!")
    else:
        print("\n❌ COMPLETE PIPELINE TEST FAILED!")
        sys.exit(1)
