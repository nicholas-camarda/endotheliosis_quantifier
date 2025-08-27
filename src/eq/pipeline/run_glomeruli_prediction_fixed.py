#!/usr/bin/env python3
"""
Fixed glomeruli prediction using the correct threshold for the biased model.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from fastai.vision.all import *
from PIL import Image


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

def run_glomeruli_prediction_fixed(config_path: str = "configs/glomeruli_finetuning_config.yaml"):
    """Run glomeruli prediction with the correct threshold for the biased model."""
    
    print("🚀 Starting glomeruli prediction with FIXED threshold approach")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Configuration loaded: {config.get('name', 'Unknown')}")
    
    # Load glomeruli data
    print("📊 Loading glomeruli data...")
    from eq.data_management.loaders import load_glomeruli_data
    
    data = load_glomeruli_data(config)
    
    val_images = data['val']['images']
    val_masks = data['val']['masks']
    
    print(f"✅ Validation data loaded: {val_images.shape}")
    print(f"✅ Validation masks loaded: {val_masks.shape}")
    
    # Load the backup model
    backup_model_path = "backups/glomerulus_segmentation_model-dynamic_unet-e50_b16_s84.pkl"
    print(f"🧠 Loading backup model: {backup_model_path}")
    
    try:
        learn = load_learner(backup_model_path)
        print("✅ Successfully loaded backup glomeruli model")
        
        # Get the PyTorch model
        model = learn.model
        model.eval()
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Test predictions on validation data with different thresholds
    print("🔍 Testing predictions with different thresholds...")
    
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]  # Test different thresholds
    results = {}
    
    num_samples = min(20, len(val_images))  # Test on first 20 samples
    
    for threshold in thresholds:
        print(f"\n--- Testing threshold {threshold} ---")
        
        total_dice = 0
        total_iou = 0
        total_accuracy = 0
        
        for i in range(num_samples):
            if i % 5 == 0:
                print(f"  Processing sample {i+1}/{num_samples}")
            
            img = val_images[i]
            true_mask = val_masks[i]
            
            # Resize to 256x256 and convert to tensor
            img_pil = Image.fromarray((img * 255).astype(np.uint8))
            img_256 = img_pil.resize((256, 256))
            img_tensor = torch.from_numpy(np.array(img_256)).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            
            # Get prediction
            with torch.no_grad():
                logits = model(img_tensor)
                probs = F.softmax(logits, dim=1)
                
                # Use the specified threshold instead of 0.5
                pred_mask = (probs[:, 1] > threshold).float()  # Class 1 (glom) > threshold
                
                # Convert to numpy
                pred_np = pred_mask.squeeze().numpy()
            
            # Resize prediction to match ground truth size
            from scipy.ndimage import zoom
            if pred_np.shape != true_mask.shape:
                scale_factors = [true_mask.shape[j] / pred_np.shape[j] for j in range(len(true_mask.shape))]
                pred_np = zoom(pred_np, scale_factors, order=1)
            
            # Calculate metrics
            true_binary = (true_mask > 0.5).astype(np.float32)
            pred_binary = pred_np.astype(np.float32)
            
            # Calculate intersection and union
            intersection = float(np.sum(true_binary * pred_binary))
            union = float(np.sum(true_binary) + np.sum(pred_binary) - intersection)
            
            dice = (2.0 * intersection) / (np.sum(true_binary) + np.sum(pred_binary) + 1e-7)
            iou = intersection / (union + 1e-7)
            accuracy = float(np.sum(true_binary == pred_binary)) / true_binary.size
            
            total_dice += dice
            total_iou += iou
            total_accuracy += accuracy
        
        # Calculate averages
        avg_dice = total_dice / num_samples
        avg_iou = total_iou / num_samples
        avg_accuracy = total_accuracy / num_samples
        
        results[threshold] = {
            'dice': avg_dice,
            'iou': avg_iou,
            'accuracy': avg_accuracy
        }
        
        print(f"  Threshold {threshold}: Dice={avg_dice:.4f}, IoU={avg_iou:.4f}, Acc={avg_accuracy:.4f}")
    
    # Find best threshold
    best_threshold = max(results.keys(), key=lambda t: results[t]['dice'])
    best_results = results[best_threshold]
    
    print("\n🎉 === BEST RESULTS ===")
    print(f"Best threshold: {best_threshold}")
    print(f"Best Dice Score: {best_results['dice']:.4f}")
    print(f"Best IoU Score: {best_results['iou']:.4f}")
    print(f"Best Pixel Accuracy: {best_results['accuracy']:.4f}")
    
    # Save results
    output_dir = Path("output/glomeruli_prediction_fixed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "threshold_analysis.txt", "w") as f:
        f.write("GLOMERULI PREDICTION - THRESHOLD ANALYSIS\n")
        f.write("=" * 50 + "\n\n")
        
        for threshold in sorted(results.keys()):
            result = results[threshold]
            f.write(f"Threshold {threshold}:\n")
            f.write(f"  Dice: {result['dice']:.4f}\n")
            f.write(f"  IoU: {result['iou']:.4f}\n")
            f.write(f"  Accuracy: {result['accuracy']:.4f}\n\n")
        
        f.write(f"BEST THRESHOLD: {best_threshold}\n")
        f.write(f"Best Dice: {best_results['dice']:.4f}\n")
        f.write(f"Best IoU: {best_results['iou']:.4f}\n")
    
    print(f"✅ Results saved to: {output_dir}")
    
    return {
        'best_threshold': best_threshold,
        'best_dice': best_results['dice'],
        'best_iou': best_results['iou'],
        'best_accuracy': best_results['accuracy'],
        'all_results': results
    }

if __name__ == "__main__":
    try:
        metrics = run_glomeruli_prediction_fixed()
        if metrics:
            print("\n🎉 GLOMERULI PREDICTION COMPLETED SUCCESSFULLY!")
            print(f"📊 Best Dice Score: {metrics['best_dice']:.4f}")
            print(f"📊 Best IoU Score: {metrics['best_iou']:.4f}")
            print(f"📊 Best Pixel Accuracy: {metrics['best_accuracy']:.4f}")
            print(f"📊 Best Threshold: {metrics['best_threshold']}")
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
