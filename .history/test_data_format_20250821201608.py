#!/usr/bin/env python3
"""Test script to check raw data format and fix visualization issues."""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def test_raw_data():
    """Test the raw data format directly."""
    
    print("🔍 Testing raw data format...")
    
    # Set MPS fallback
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    
    # Load cached data directly from pickle files
    cache_dir = Path('data/preeclampsia_data/cache')
    
    try:
        # Load training data
        with open(cache_dir / 'train_images.pickle', 'rb') as f:
            train_images = np.load(f, allow_pickle=True)
        with open(cache_dir / 'train_masks.pickle', 'rb') as f:
            train_masks = np.load(f, allow_pickle=True)
        with open(cache_dir / 'val_images.pickle', 'rb') as f:
            val_images = np.load(f, allow_pickle=True)
        with open(cache_dir / 'val_masks.pickle', 'rb') as f:
            val_masks = np.load(f, allow_pickle=True)
        
        print(f"✅ Data loaded successfully:")
        print(f"   Training images: {train_images.shape}")
        print(f"   Training masks: {train_masks.shape}")
        print(f"   Validation images: {val_images.shape}")
        print(f"   Validation masks: {val_masks.shape}")
        
        # Check data ranges
        print(f"\n📊 Data ranges:")
        print(f"   Train images: {train_images.min():.4f} to {train_images.max():.4f}")
        print(f"   Train masks: {train_masks.min():.4f} to {train_masks.max():.4f}")
        print(f"   Val images: {val_images.min():.4f} to {val_images.max():.4f}")
        print(f"   Val masks: {val_masks.min():.4f} to {val_masks.max():.4f}")
        
        # Check data types
        print(f"\n🔧 Data types:")
        print(f"   Train images: {train_images.dtype}")
        print(f"   Train masks: {train_masks.dtype}")
        
        # Sample some values
        print(f"\n📋 Sample values (first image, first 5x5 pixels):")
        print(f"   Train image sample: {train_images[0, :5, :5].flatten()[:10]}")
        print(f"   Train mask sample: {train_masks[0, :5, :5].flatten()[:10]}")
        
        return train_images, train_masks, val_images, val_masks
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

def create_fixed_visualization(train_images, train_masks, val_images, val_masks):
    """Create a fixed visualization showing the actual data."""
    
    if train_images is None:
        print("❌ No data to visualize")
        return
    
    print("\n🎨 Creating fixed visualization...")
    
    # Create output directory
    output_dir = Path("output/fixed_visualization")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Select samples for visualization
    n_samples = min(4, len(val_images))
    
    plt.figure(figsize=(15, 5 * n_samples))
    
    for i in range(n_samples):
        # Original image (raw data)
        plt.subplot(n_samples, 3, i*3 + 1)
        img = val_images[i]  # No need to remove channel dimension
        
        # Normalize for display (0-1 range)
        img_normalized = (img - img.min()) / (img.max() - img.min())
        plt.imshow(img_normalized, cmap='gray')
        plt.title(f'Sample {i+1}: Raw Image\nRange: {img.min():.3f} to {img.max():.3f}', fontsize=10)
        plt.axis('off')
        
        # Ground truth mask
        plt.subplot(n_samples, 3, i*3 + 2)
        mask = val_masks[i]  # No need to remove channel dimension
        plt.imshow(mask, cmap='gray')
        plt.title(f'Sample {i+1}: Ground Truth\nRange: {mask.min():.3f} to {mask.max():.3f}', fontsize=10)
        plt.axis('off')
        
        # Show mask values
        plt.subplot(n_samples, 3, i*3 + 3)
        # Create a simple prediction visualization (placeholder)
        # In reality, this would come from the model
        pred_placeholder = np.zeros_like(mask)
        plt.imshow(pred_placeholder, cmap='gray')
        plt.title(f'Sample {i+1}: Prediction\n(Placeholder - Model Output)', fontsize=10)
        plt.axis('off')
    
    plt.suptitle('Raw Data Visualization - Fixed Format', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save the fixed visualization
    output_path = output_dir / "fixed_data_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Fixed visualization saved: {output_path}")
    
    # Create a detailed data analysis
    create_data_analysis(train_images, train_masks, val_images, val_masks, output_dir)

def create_data_analysis(train_images, train_masks, val_images, val_masks, output_dir):
    """Create detailed data analysis plots."""
    
    print("\n📊 Creating data analysis...")
    
    plt.figure(figsize=(20, 12))
    
    # Plot 1: Image value distributions
    plt.subplot(2, 3, 1)
    plt.hist(train_images.flatten(), bins=50, alpha=0.7, label='Training Images', density=True)
    plt.hist(val_images.flatten(), bins=50, alpha=0.7, label='Validation Images', density=True)
    plt.xlabel('Pixel Values')
    plt.ylabel('Density')
    plt.title('Image Pixel Value Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Mask value distributions
    plt.subplot(2, 3, 2)
    plt.hist(train_masks.flatten(), bins=20, alpha=0.7, label='Training Masks', density=True)
    plt.hist(val_masks.flatten(), bins=20, alpha=0.7, label='Validation Masks', density=True)
    plt.xlabel('Mask Values')
    plt.ylabel('Density')
    plt.title('Mask Value Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Sample image with actual values
    plt.subplot(2, 3, 3)
    sample_img = val_images[0]
    plt.imshow(sample_img, cmap='gray')
    plt.title(f'Sample Image\nMin: {sample_img.min():.3f}, Max: {sample_img.max():.3f}')
    plt.colorbar()
    plt.axis('off')
    
    # Plot 4: Sample mask with actual values
    plt.subplot(2, 3, 4)
    sample_mask = val_masks[0]
    plt.imshow(sample_mask, cmap='gray')
    plt.title(f'Sample Mask\nMin: {sample_mask.min():.3f}, Max: {sample_mask.max():.3f}')
    plt.colorbar()
    plt.axis('off')
    
    # Plot 5: Data statistics table
    plt.subplot(2, 3, 5)
    stats_data = [
        ['Training Images', f'{train_images.shape}', f'{train_images.min():.3f}', f'{train_images.max():.3f}'],
        ['Training Masks', f'{train_masks.shape}', f'{train_masks.min():.3f}', f'{train_masks.max():.3f}'],
        ['Validation Images', f'{val_images.shape}', f'{val_images.min():.3f}', f'{val_images.max():.3f}'],
        ['Validation Masks', f'{val_masks.shape}', f'{val_masks.min():.3f}', f'{val_masks.max():.3f}']
    ]
    
    plt.table(cellText=stats_data, 
              colLabels=['Dataset', 'Shape', 'Min', 'Max'],
              cellLoc='center',
              loc='center')
    plt.title('Data Statistics Summary')
    plt.axis('off')
    
    # Plot 6: Image size analysis
    plt.subplot(2, 3, 6)
    plt.hist([img.shape[0] for img in train_images], bins=20, alpha=0.7, label='Training')
    plt.hist([img.shape[0] for img in val_images], bins=20, alpha=0.7, label='Validation')
    plt.xlabel('Image Size (pixels)')
    plt.ylabel('Count')
    plt.title('Image Size Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the analysis
    analysis_path = output_dir / "data_analysis.png"
    plt.savefig(analysis_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Data analysis saved: {analysis_path}")

if __name__ == "__main__":
    print("🧪 Testing data format and creating fixed visualizations...")
    
    # Test raw data
    train_images, train_masks, val_images, val_masks = test_raw_data()
    
    if train_images is not None:
        # Create fixed visualization
        create_fixed_visualization(train_images, train_masks, val_images, val_masks)
        
        print("\n🎉 Data analysis complete!")
        print("Check output/fixed_visualization/ for the corrected visualizations")
    else:
        print("\n❌ Could not analyze data")
