#!/usr/bin/env python3
"""
Visualization utilities for debugging and understanding data.

This module provides utilities for visualizing images, masks, and their relationships
to help with debugging and data exploration.
"""

from pathlib import Path
from typing import Optional, Union, Tuple
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from eq.utils.logger import get_logger


def visualize_mask(
    mask_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
    show_stats: bool = True
) -> Path:
    """
    Visualize a mask with multiple display options for better visibility.
    
    Args:
        mask_path: Path to the mask file
        output_path: Where to save the visualization (default: mask_path + '_visualization.png')
        title: Title for the plot
        show_stats: Whether to print mask statistics
        
    Returns:
        Path to the saved visualization
    """
    logger = get_logger("eq.visualization")
    
    mask_path = Path(mask_path)
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
    
    # Load mask
    mask_img = Image.open(mask_path)
    mask_arr = np.array(mask_img)
    
    if show_stats:
        logger.info(f"Mask statistics for {mask_path.name}:")
        logger.info(f"  Shape: {mask_arr.shape}")
        logger.info(f"  Dtype: {mask_arr.dtype}")
        logger.info(f"  Min/Max: {mask_arr.min()}/{mask_arr.max()}")
        logger.info(f"  Unique values: {np.unique(mask_arr)}")
        positive_pixels = (mask_arr > 0).sum()
        logger.info(f"  Positive pixels: {positive_pixels} ({positive_pixels / mask_arr.size * 100:.1f}%)")
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original mask
    axes[0].imshow(mask_arr, cmap='gray')
    axes[0].set_title('Original Mask\n(0=black, 1=white)')
    axes[0].axis('off')
    
    # Hot colormap for better visibility
    axes[1].imshow(mask_arr, cmap='hot', vmin=0, vmax=1)
    axes[1].set_title('Hot Colormap\n(better visibility)')
    axes[1].axis('off')
    
    # Scaled to 0-255 for maximum contrast
    mask_scaled = mask_arr * 255
    axes[2].imshow(mask_scaled, cmap='gray')
    axes[2].set_title('Scaled (0-255)\n(maximum contrast)')
    axes[2].axis('off')
    
    if title:
        fig.suptitle(title, fontsize=16)
    else:
        fig.suptitle(f'Mask Visualization: {mask_path.name}', fontsize=16)
    
    plt.tight_layout()
    
    # Save visualization
    if output_path is None:
        output_path = Path("test_output") / f"{mask_path.stem}_visualization.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path = Path(output_path)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Mask visualization saved to: {output_path}")
    return output_path


def visualize_image_mask_pair(
    image_path: Union[str, Path],
    mask_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
    show_stats: bool = True
) -> Path:
    """
    Visualize an image and its corresponding mask side by side.
    
    Args:
        image_path: Path to the image file
        mask_path: Path to the mask file
        output_path: Where to save the visualization
        title: Title for the plot
        show_stats: Whether to print statistics
        
    Returns:
        Path to the saved visualization
    """
    logger = get_logger("eq.visualization")
    
    image_path = Path(image_path)
    mask_path = Path(mask_path)
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
    
    # Load image and mask
    img = Image.open(image_path)
    mask = Image.open(mask_path)
    
    img_arr = np.array(img)
    mask_arr = np.array(mask)
    
    if show_stats:
        logger.info(f"Image-Mask pair statistics:")
        logger.info(f"  Image: {img_arr.shape}, {img_arr.dtype}, min/max: {img_arr.min()}/{img_arr.max()}")
        logger.info(f"  Mask: {mask_arr.shape}, {mask_arr.dtype}, min/max: {mask_arr.min()}/{mask_arr.max()}")
        positive_pixels = (mask_arr > 0).sum()
        logger.info(f"  Mask positive pixels: {positive_pixels} ({positive_pixels / mask_arr.size * 100:.1f}%)")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original image
    if img_arr.ndim == 3:
        axes[0,0].imshow(img_arr)
        axes[0,0].set_title('Original Image (RGB)')
    else:
        axes[0,0].imshow(img_arr, cmap='gray')
        axes[0,0].set_title('Original Image (Grayscale)')
    axes[0,0].axis('off')
    
    # Mask with hot colormap
    axes[0,1].imshow(mask_arr, cmap='hot', vmin=0, vmax=1)
    axes[0,1].set_title('Mask (Hot Colormap)')
    axes[0,1].axis('off')
    
    # Overlay
    overlay = img_arr.copy()
    if overlay.ndim == 3:
        overlay = overlay.mean(axis=2)  # Convert to grayscale if RGB
    overlay = overlay / overlay.max()  # Normalize
    overlay[mask_arr > 0] = 1  # Highlight mask regions
    axes[1,0].imshow(overlay, cmap='gray')
    axes[1,0].set_title('Image + Mask Overlay')
    axes[1,0].axis('off')
    
    # Mask with high contrast
    mask_contrast = mask_arr * 255  # Scale to 0-255
    axes[1,1].imshow(mask_contrast, cmap='gray')
    axes[1,1].set_title('Mask (Scaled 0-255)')
    axes[1,1].axis('off')
    
    if title:
        fig.suptitle(title, fontsize=16)
    else:
        fig.suptitle(f'Image-Mask Pair: {image_path.name}', fontsize=16)
    
    plt.tight_layout()
    
    # Save visualization
    if output_path is None:
        output_path = Path("test_output") / f"{image_path.stem}_mask_comparison.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path = Path(output_path)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Image-mask visualization saved to: {output_path}")
    return output_path


def visualize_batch_masks(
    mask_paths: list[Union[str, Path]],
    output_path: Optional[Union[str, Path]] = None,
    max_masks: int = 16,
    title: Optional[str] = None
) -> Path:
    """
    Visualize multiple masks in a grid for batch inspection.
    
    Args:
        mask_paths: List of paths to mask files
        output_path: Where to save the visualization
        max_masks: Maximum number of masks to display
        title: Title for the plot
        
    Returns:
        Path to the saved visualization
    """
    logger = get_logger("eq.visualization")
    
    # Limit number of masks
    mask_paths = mask_paths[:max_masks]
    
    # Calculate grid size
    n_masks = len(mask_paths)
    cols = min(4, n_masks)
    rows = (n_masks + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten()
    
    for i, mask_path in enumerate(mask_paths):
        mask_path = Path(mask_path)
        if not mask_path.exists():
            logger.warning(f"Mask file not found: {mask_path}")
            continue
        
        # Load and display mask
        mask = Image.open(mask_path)
        mask_arr = np.array(mask)
        
        axes_flat[i].imshow(mask_arr, cmap='hot', vmin=0, vmax=1)
        axes_flat[i].set_title(f'{mask_path.name}\n{(mask_arr > 0).sum()} pos')
        axes_flat[i].axis('off')
    
    # Hide unused subplots
    for i in range(n_masks, len(axes_flat)):
        axes_flat[i].axis('off')
    
    if title:
        fig.suptitle(title, fontsize=16)
    else:
        fig.suptitle(f'Batch Mask Visualization ({n_masks} masks)', fontsize=16)
    
    plt.tight_layout()
    
    # Save visualization
    if output_path is None:
        output_path = Path("test_output") / "batch_mask_visualization.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path = Path(output_path)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Batch mask visualization saved to: {output_path}")
    return output_path


def main():
    """CLI interface for visualization utilities."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualization utilities for debugging data')
    parser.add_argument('--mask', help='Path to mask file to visualize')
    parser.add_argument('--image', help='Path to image file (for image-mask pair visualization)')
    parser.add_argument('--output', help='Output path for visualization')
    parser.add_argument('--title', help='Title for the visualization')
    parser.add_argument('--batch', nargs='+', help='Multiple mask paths for batch visualization')
    parser.add_argument('--max-masks', type=int, default=16, help='Maximum masks for batch visualization')
    
    args = parser.parse_args()
    
    try:
        if args.batch:
            # Batch visualization
            output_path = visualize_batch_masks(
                args.batch,
                output_path=args.output,
                max_masks=args.max_masks,
                title=args.title
            )
        elif args.image and args.mask:
            # Image-mask pair visualization
            output_path = visualize_image_mask_pair(
                args.image,
                args.mask,
                output_path=args.output,
                title=args.title
            )
        elif args.mask:
            # Single mask visualization
            output_path = visualize_mask(
                args.mask,
                output_path=args.output,
                title=args.title
            )
        else:
            parser.print_help()
            return
        
        print(f"✅ Visualization saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
