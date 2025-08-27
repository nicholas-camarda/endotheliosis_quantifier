#!/usr/bin/env python3
"""
Glomeruli Prediction Pipeline

This module runs predictions using the exact same data augmentation and preprocessing
as the original training, but adapted to the current data structure.
"""

import sys
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from fastai.vision.all import *

from eq.utils.logger import get_logger

logger = get_logger("eq.glomeruli_prediction")

# Define functions needed for loading the backup model
def get_glom_y(o):
    """Get glomeruli mask for a given image file."""
    p2c = [0, 1]  # Default binary mask codes
    return get_glom_mask_file(o, p2c)

def get_glom_mask_file(o, p2c):
    """Get glomeruli mask file with color mapping."""
    import numpy as np
    from PIL import Image

    # Load the mask image
    msk = np.array(Image.open(o))

    # Apply threshold
    thresh = 127
    msk[msk <= thresh] = 0
    msk[msk > thresh] = 1

    # Apply color mapping
    for i, val in enumerate(p2c):
        msk[msk == p2c[i]] = val

    from fastai.vision.core import PILMask
    return PILMask.create(msk)

def get_all_paths(directory_path):
    """Get all file paths from a directory recursively."""
    directory = Path(directory_path)
    paths = []
    for path in directory.glob('**/*'):
        if path.is_file():
            paths.append(path)
    return paths

def get_glom_mask_file_current(image_file, p2c, thresh=127):
    """Get glomeruli mask file for current data structure."""
    # Current structure: image_patches/T19_Image0_10_10.jpg -> mask_patches/T19_Image0_10_10_mask.jpg
    base_path = image_file.parent.parent
    mask_dir = base_path / "mask_patches"
    mask_name = image_file.stem + "_mask" + image_file.suffix
    mask_path = mask_dir / mask_name
    
    # Convert to an array (mask)
    msk = np.array(PILMask.create(mask_path))
    # Convert the image to binary if it isn't already
    msk[msk <= thresh] = 0
    msk[msk > thresh] = 1
    
    # Find all the possible values in the mask (0,255)
    for i, val in enumerate(p2c):
        msk[msk == p2c[i]] = val
    return PILMask.create(msk)

def get_glom_y_current(o):
    """Get glomeruli mask for current data structure."""
    p2c = {0: 0, 1: 1}  # Binary mask codes
    return get_glom_mask_file_current(o, p2c)

def run_glomeruli_prediction(config_path: str = "configs/glomeruli_finetuning_config.yaml"):
    """Run glomeruli prediction using fresh data loading approach."""
    logger.info("🚀 Starting glomeruli prediction with fresh data approach")
    
    # Load configuration
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Configuration loaded: {config.get('name', 'Unknown')}")
    
    # Configure hardware settings using the proper backend management system
    from eq.utils.backend_manager import BackendManager

    # Initialize backend manager (automatically handles MPS fallback)
    backend_manager = BackendManager()
    logger.info(f"✅ Backend manager initialized: {backend_manager.current_backend.backend_type.value}")
    
    # Apply hardware configuration overrides if specified
    hardware_config = config.get('hardware', {})
    if hardware_config.get('force_cpu', False):
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        logger.info("✅ Forced CPU usage")
    
    # Get paths from configuration
    data_config = config.get('data', {})
    processed_config = data_config.get('processed', {})
    
    train_image_path = Path(processed_config.get('train_dir'))
    train_mask_path = Path(processed_config.get('train_mask_dir'))
    
    logger.info(f"📁 Image path: {train_image_path}")
    logger.info(f"📁 Mask path: {train_mask_path}")
    
    # Load fresh data using the data loader
    logger.info("📊 Loading glomeruli data...")
    from eq.data_management.loaders import load_glomeruli_data
    
    data = load_glomeruli_data(config)
    
    # Get validation data
    val_images = data['val']['images']
    val_masks = data['val']['masks']
    
    logger.info(f"✅ Validation data loaded: {val_images.shape}")
    logger.info(f"✅ Validation masks loaded: {val_masks.shape}")
    
    # Load the backup model
    backup_model_path = "backups/glomerulus_segmentation_model-dynamic_unet-e50_b16_s84.pkl"
    logger.info(f"🧠 Loading backup model: {backup_model_path}")
    
    try:
        learn = load_learner(backup_model_path)
        logger.info("✅ Successfully loaded backup glomeruli model")
        model = learn
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return
    
    # Test predictions on validation data
    logger.info("🔍 Testing predictions on validation data...")
    
    total_dice = 0
    total_iou = 0
    total_accuracy = 0
    num_samples = min(10, len(val_images))  # Test on first 10 samples
    
    for i in range(num_samples):
        logger.info(f"Processing sample {i+1}/{num_samples}")
        
        img = val_images[i]
        true_mask = val_masks[i]
        
        # Ensure image is 3-channel for prediction
        if img.ndim == 3 and img.shape[-1] == 1:
            img_3ch = np.repeat(img, 3, axis=-1)
        else:
            img_3ch = img
        
        # Convert to PIL for prediction
        img_pil = Image.fromarray((img_3ch * 255).astype(np.uint8))
        
        # Get prediction
        pred_result = learn.predict(img_pil)
        
        # Extract prediction tensor
        if isinstance(pred_result, tuple) and len(pred_result) >= 2:
            pred_tensor = pred_result[1]
        else:
            pred_tensor = pred_result
        
        # Convert to numpy
        if hasattr(pred_tensor, 'numpy'):
            pred_mask = pred_tensor.numpy()
        elif hasattr(pred_tensor, 'cpu'):
            pred_mask = pred_tensor.cpu().numpy()
        else:
            pred_mask = np.asarray(pred_tensor)
        
        # Resize prediction to match ground truth size if needed
        from scipy.ndimage import zoom
        if pred_mask.shape != true_mask.shape:
            if len(true_mask.shape) == 3 and len(pred_mask.shape) == 2:
                pred_mask = pred_mask[..., None]
            
            if pred_mask.shape != true_mask.shape:
                scale_factors = [true_mask.shape[j] / pred_mask.shape[j] for j in range(len(true_mask.shape))]
                pred_mask = zoom(pred_mask, scale_factors, order=1)
        
        # Calculate metrics
        true_binary = (np.squeeze(true_mask) > 0.5).astype(np.float32)
        pred_binary = (pred_mask > 0.5).astype(np.float32)
        
        # Calculate intersection and union
        intersection = float(np.sum(true_binary * pred_binary))
        union = float(np.sum(true_binary) + np.sum(pred_binary) - intersection)
        
        dice = (2.0 * intersection) / (np.sum(true_binary) + np.sum(pred_binary) + 1e-7)
        iou = intersection / (union + 1e-7)
        accuracy = float(np.sum(true_binary == pred_binary)) / true_binary.size
        
        total_dice += dice
        total_iou += iou
        total_accuracy += accuracy
        
        logger.info(f"  Sample {i+1}: Dice={dice:.4f}, IoU={iou:.4f}, Acc={accuracy:.4f}")
    
    # Calculate averages
    avg_dice = total_dice / num_samples
    avg_iou = total_iou / num_samples
    avg_accuracy = total_accuracy / num_samples
    
    logger.info("🎉 GLOMERULI PREDICTION COMPLETED SUCCESSFULLY!")
    logger.info(f"📊 Dice Score: {avg_dice:.4f}")
    logger.info(f"📊 IoU Score: {avg_iou:.4f}")
    logger.info(f"📊 Pixel Accuracy: {avg_accuracy:.4f}")
    
    return {
        'dice_score': avg_dice,
        'iou_score': avg_iou,
        'pixel_accuracy': avg_accuracy,
        'num_samples': num_samples
    }

if __name__ == "__main__":
    try:
        metrics = run_glomeruli_prediction()
        if metrics:
            print("\n🎉 GLOMERULI PREDICTION COMPLETED SUCCESSFULLY!")
            print(f"📊 Dice Score: {metrics['dice_score']:.4f}")
            print(f"📊 IoU Score: {metrics['iou_score']:.4f}")
            print(f"📊 Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        raise
