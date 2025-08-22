#!/usr/bin/env python3
"""Create better visualizations for the low-contrast medical images."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure


def load_data():
    """Load the cached data."""
    cache_dir = Path('data/preeclampsia_data/cache')
    
    with open(cache_dir / 'train_images.pickle', 'rb') as f:
        train_images = np.load(f, allow_pickle=True)
    with open(cache_dir / 'train_masks.pickle', 'rb') as f:
        train_masks = np.load(f, allow_pickle=True)
    with open(cache_dir / 'val_images.pickle', 'rb') as f:
        val_images = np.load(f, allow_pickle=True)
    with open(cache_dir / 'val_masks.pickle', 'rb') as f:
        val_masks = np.load(f, allow_pickle=True)
    
    return train_images, train_masks, val_images, val_masks

def enhance_contrast(image):
    """Enhance contrast for low-contrast medical images."""
    # Remove the extra dimension if present
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image.squeeze()
    
    # Apply histogram equalization to enhance contrast
    enhanced = exposure.equalize_hist(image)
    
    # Alternative: apply adaptive histogram equalization
    # enhanced = exposure.equalize_adapthist(image, clip_limit=0.03)
    
    return enhanced

def create_enhanced_visualization():
    """Create enhanced visualization with proper contrast."""
    
    print("🎨 Creating enhanced visualization with proper contrast...")
    
    # Load data
    train_images, train_masks, val_images, val_masks = load_data()
    
    # Create output directory
    output_dir = Path("output/enhanced_visualization")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Select samples for visualization
    n_samples = min(4, len(val_images))
    
    plt.figure(figsize=(18, 6 * n_samples))
    
    for i in range(n_samples):
        # Original image (raw data)
        plt.subplot(n_samples, 4, i*4 + 1)
        img_raw = val_images[i].squeeze()
        plt.imshow(img_raw, cmap='gray')
        plt.title(f'Sample {i+1}: Raw Image\nRange: {img_raw.min():.6f} to {img_raw.max():.6f}', fontsize=10)
        plt.axis('off')
        
        # Enhanced image (contrast enhanced)
        plt.subplot(n_samples, 4, i*4 + 2)
        img_enhanced = enhance_contrast(val_images[i])
        plt.imshow(img_enhanced, cmap='gray')
        plt.title(f'Sample {i+1}: Enhanced Image\n(Histogram Equalized)', fontsize=10)
        plt.axis('off')
        
        # Ground truth mask
        plt.subplot(n_samples, 4, i*4 + 3)
        mask = val_masks[i].squeeze()
        plt.imshow(mask, cmap='gray')
        plt.title(f'Sample {i+1}: Ground Truth\nRange: {mask.min():.6f} to {mask.max():.6f}', fontsize=10)
        plt.axis('off')
        
        # Enhanced mask (if needed)
        plt.subplot(n_samples, 4, i*4 + 4)
        mask_enhanced = enhance_contrast(val_masks[i])
        plt.imshow(mask_enhanced, cmap='gray')
        plt.title(f'Sample {i+1}: Enhanced Mask\n(Histogram Equalized)', fontsize=10)
        plt.axis('off')
    
    plt.suptitle('Enhanced Medical Image Visualization - Proper Contrast', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save the enhanced visualization
    output_path = output_dir / "enhanced_medical_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Enhanced visualization saved: {output_path}")
    
    # Create a comparison visualization
    create_comparison_visualization(val_images, val_masks, output_dir)
    
    # Create detailed analysis
    create_detailed_analysis(train_images, train_masks, val_images, val_masks, output_dir)

def create_comparison_visualization(val_images, val_masks, output_dir):
    """Create a comparison showing different contrast enhancement methods."""
    
    print("🔄 Creating comparison visualization...")
    
    n_samples = min(3, len(val_images))
    
    plt.figure(figsize=(20, 6 * n_samples))
    
    for i in range(n_samples):
        img = val_images[i].squeeze()
        mask = val_masks[i].squeeze()
        
        # Row 1: Different contrast enhancement methods
        plt.subplot(n_samples, 6, i*6 + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f'Sample {i+1}: Raw', fontsize=9)
        plt.axis('off')
        
        plt.subplot(n_samples, 6, i*6 + 2)
        img_eq = exposure.equalize_hist(img)
        plt.imshow(img_eq, cmap='gray')
        plt.title('Histogram Equalization', fontsize=9)
        plt.axis('off')
        
        plt.subplot(n_samples, 6, i*6 + 3)
        img_adapt = exposure.equalize_adapthist(img, clip_limit=0.03)
        plt.imshow(img_adapt, cmap='gray')
        plt.title('Adaptive Histogram', fontsize=9)
        plt.axis('off')
        
        plt.subplot(n_samples, 6, i*6 + 4)
        img_gamma = exposure.adjust_gamma(img, gamma=0.3)
        plt.imshow(img_gamma, cmap='gray')
        plt.title('Gamma Correction (0.3)', fontsize=9)
        plt.axis('off')
        
        plt.subplot(n_samples, 6, i*6 + 5)
        img_log = exposure.adjust_log(img, gain=1)
        plt.imshow(img_log, cmap='gray')
        plt.title('Log Adjustment', fontsize=9)
        plt.axis('off')
        
        plt.subplot(n_samples, 6, i*6 + 6)
        plt.imshow(mask, cmap='gray')
        plt.title('Ground Truth Mask', fontsize=9)
        plt.axis('off')
    
    plt.suptitle('Contrast Enhancement Methods Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    comparison_path = output_dir / "contrast_enhancement_comparison.png"
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Comparison visualization saved: {comparison_path}")

def create_detailed_analysis(train_images, train_masks, val_images, val_masks, output_dir):
    """Create detailed analysis of the medical images."""
    
    print("📊 Creating detailed analysis...")
    
    plt.figure(figsize=(24, 16))
    
    # Plot 1: Raw image value distributions
    plt.subplot(3, 4, 1)
    plt.hist(train_images.flatten(), bins=100, alpha=0.7, label='Training Images', density=True)
    plt.hist(val_images.flatten(), bins=100, alpha=0.7, label='Validation Images', density=True)
    plt.xlabel('Pixel Values')
    plt.ylabel('Density')
    plt.title('Raw Image Pixel Value Distribution\n(Note: Very Low Values)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 0.005)  # Focus on the actual range
    
    # Plot 2: Enhanced image value distributions
    plt.subplot(3, 4, 2)
    train_enhanced = np.array([enhance_contrast(img) for img in train_images])
    val_enhanced = np.array([enhance_contrast(img) for img in val_images])
    plt.hist(train_enhanced.flatten(), bins=100, alpha=0.7, label='Training (Enhanced)', density=True)
    plt.hist(val_enhanced.flatten(), bins=100, alpha=0.7, label='Validation (Enhanced)', density=True)
    plt.xlabel('Enhanced Pixel Values')
    plt.ylabel('Density')
    plt.title('Enhanced Image Pixel Value Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Sample raw image
    plt.subplot(3, 4, 3)
    sample_img = val_images[0].squeeze()
    plt.imshow(sample_img, cmap='gray')
    plt.title(f'Raw Sample Image\nMin: {sample_img.min():.6f}, Max: {sample_img.max():.6f}')
    plt.colorbar()
    plt.axis('off')
    
    # Plot 4: Sample enhanced image
    plt.subplot(3, 4, 4)
    sample_enhanced = enhance_contrast(val_images[0])
    plt.imshow(sample_enhanced, cmap='gray')
    plt.title('Enhanced Sample Image\n(Histogram Equalized)')
    plt.colorbar()
    plt.axis('off')
    
    # Plot 5: Mask value distributions
    plt.subplot(3, 4, 5)
    plt.hist(train_masks.flatten(), bins=50, alpha=0.7, label='Training Masks', density=True)
    plt.hist(val_masks.flatten(), bins=50, alpha=0.7, label='Validation Masks', density=True)
    plt.xlabel('Mask Values')
    plt.ylabel('Density')
    plt.title('Mask Value Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Sample mask
    plt.subplot(3, 4, 6)
    sample_mask = val_masks[0].squeeze()
    plt.imshow(sample_mask, cmap='gray')
    plt.title(f'Sample Mask\nMin: {sample_mask.min():.6f}, Max: {sample_mask.max():.6f}')
    plt.colorbar()
    plt.axis('off')
    
    # Plot 7: Image statistics table
    plt.subplot(3, 4, 7)
    stats_data = [
        ['Training Images', f'{train_images.shape}', f'{train_images.min():.6f}', f'{train_images.max():.6f}'],
        ['Training Masks', f'{train_masks.shape}', f'{train_masks.min():.6f}', f'{train_masks.max():.6f}'],
        ['Validation Images', f'{val_images.shape}', f'{val_images.min():.6f}', f'{val_images.max():.6f}'],
        ['Validation Masks', f'{val_masks.shape}', f'{val_masks.min():.6f}', f'{val_masks.max():.6f}']
    ]
    
    plt.table(cellText=stats_data, 
              colLabels=['Dataset', 'Shape', 'Min', 'Max'],
              cellLoc='center',
              loc='center')
    plt.title('Data Statistics Summary\n(Note: Very Low Pixel Values)')
    plt.axis('off')
    
    # Plot 8: Contrast enhancement comparison
    plt.subplot(3, 4, 8)
    methods = ['Raw', 'Histogram Eq', 'Adaptive Hist', 'Gamma 0.3', 'Log Adjust']
    sample_img = val_images[0].squeeze()
    
    # Calculate contrast metrics for each method
    raw_contrast = sample_img.std()
    hist_eq_contrast = exposure.equalize_hist(sample_img).std()
    adapt_hist_contrast = exposure.equalize_adapthist(sample_img, clip_limit=0.03).std()
    gamma_contrast = exposure.adjust_gamma(sample_img, gamma=0.3).std()
    log_contrast = exposure.adjust_log(sample_img, gain=1).std()
    
    contrasts = [raw_contrast, hist_eq_contrast, adapt_hist_contrast, gamma_contrast, log_contrast]
    
    plt.bar(methods, contrasts, color=['red', 'blue', 'green', 'orange', 'purple'])
    plt.title('Contrast Improvement Comparison\n(Standard Deviation)')
    plt.ylabel('Standard Deviation')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Plot 9: Image size analysis
    plt.subplot(3, 4, 9)
    plt.hist([img.shape[0] for img in train_images], bins=20, alpha=0.7, label='Training')
    plt.hist([img.shape[0] for img in val_images], bins=20, alpha=0.7, label='Validation')
    plt.xlabel('Image Size (pixels)')
    plt.ylabel('Count')
    plt.title('Image Size Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 10: Sample image progression
    plt.subplot(3, 4, 10)
    sample_img = val_images[0].squeeze()
    plt.imshow(sample_img, cmap='gray')
    plt.title('Sample Image\n(256x256 pixels)')
    plt.axis('off')
    
    # Plot 11: Sample mask progression
    plt.subplot(3, 4, 11)
    sample_mask = val_masks[0].squeeze()
    plt.imshow(sample_mask, cmap='gray')
    plt.title('Sample Mask\n(256x256 pixels)')
    plt.axis('off')
    
    # Plot 12: Summary
    plt.subplot(3, 4, 12)
    summary_text = f"""Medical Image Analysis Summary

Total Images: {len(train_images) + len(val_images)}
Image Size: 256x256 pixels
Pixel Range: {train_images.min():.6f} to {train_images.max():.6f}

Key Findings:
• Very low contrast images
• Need contrast enhancement
• Masks are binary-like
• Good for segmentation

Recommendation:
Use histogram equalization
for visualization"""
    
    plt.text(0.1, 0.5, summary_text, 
             fontsize=10, fontfamily='monospace',
             transform=plt.gca().transAxes,
             verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    plt.title('Analysis Summary')
    plt.axis('off')
    
    plt.tight_layout()
    
    # Save the detailed analysis
    analysis_path = output_dir / "detailed_medical_analysis.png"
    plt.savefig(analysis_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Detailed analysis saved: {analysis_path}")

if __name__ == "__main__":
    print("🎨 Creating enhanced medical image visualizations...")
    
    try:
        create_enhanced_visualization()
        print("\n🎉 Enhanced visualization complete!")
        print("Check output/enhanced_visualization/ for:")
        print("  - Enhanced medical images with proper contrast")
        print("  - Contrast enhancement comparison")
        print("  - Detailed medical image analysis")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
