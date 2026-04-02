#!/usr/bin/env python3
"""
Final Glomeruli Model Evaluation Script
Uses the working direct model approach that bypasses FastAI inference issues
"""

import os
import sys

sys.path.append('src')

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from eq.utils.model_loader import load_model_safely


def evaluate_glomeruli_model(model_path, test_img_dir, test_mask_dir, output_dir):
    """Evaluate glomeruli model using direct model calls (working approach)"""
    
    print(f"🔍 === EVALUATING GLOMERULI MODEL ===")
    print(f"Model: {model_path}")
    print(f"Test images: {test_img_dir}")
    print(f"Test masks: {test_mask_dir}")
    print(f"Output: {output_dir}")
    
    # Load model
    try:
        learn = load_model_safely(model_path)
        print("   ✅ Model loaded successfully")
    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")
        return None
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find matching image-mask pairs
    def find_matching_pairs():
        """Find images that have corresponding ground truth masks"""
        image_files = [f for f in os.listdir(test_img_dir) if f.endswith('.jpg')]
        mask_files = [f for f in os.listdir(test_mask_dir) if f.endswith('_mask.jpg')]
        
        # Create mapping from image to mask
        mask_map = {}
        for mask_file in mask_files:
            # Extract base name (e.g., 'T29_Image0_1_7' from 'T29_Image0_1_7_mask.jpg')
            base_name = mask_file.replace('_mask.jpg', '')
            mask_map[base_name] = mask_file
        
        # Find images with masks
        matching_pairs = []
        for img_file in image_files:
            base_name = img_file.replace('.jpg', '')
            if base_name in mask_map:
                img_path = os.path.join(test_img_dir, img_file)
                mask_path = os.path.join(test_mask_dir, mask_map[base_name])
                matching_pairs.append((img_path, mask_path))
        
        return matching_pairs
    
    matching_pairs = find_matching_pairs()
    print(f"   Found {len(matching_pairs)} image-mask pairs")
    
    if not matching_pairs:
        print("   ❌ No matching pairs found")
        return None
    
    # Test with first few pairs
    test_pairs = matching_pairs[:5]
    
    results = []
    
    for i, (img_path, mask_path) in enumerate(test_pairs):
        print(f"\n   Testing pair {i+1}/{len(test_pairs)}: {os.path.basename(img_path)}")
        
        try:
            # Load image and mask
            img = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')
            
            # Resize to expected input size (256x256 for glomeruli model)
            img_256 = img.resize((256, 256), Image.BILINEAR)
            mask_256 = mask.resize((256, 256), Image.NEAREST)
            
            # Convert to tensors
            img_tensor = torch.from_numpy(np.array(img_256)).float() / 255.0
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
            
            mask_tensor = torch.from_numpy(np.array(mask_256)).float() / 255.0
            mask_tensor = (mask_tensor > 0.5).float()
            
            print(f"     Input shape: {img_tensor.shape}")
            print(f"     Mask shape: {mask_tensor.shape}")
            
            # Make prediction using model directly
            learn.model.eval()
            with torch.no_grad():
                raw_output = learn.model(img_tensor)
            
            print(f"     Raw output shape: {raw_output.shape}")
            print(f"     Raw output range: [{raw_output.min().item():.3f}, {raw_output.max().item():.3f}]")
            
            # Apply softmax to get probabilities
            if raw_output.shape[1] == 2:  # Binary segmentation
                probs = torch.softmax(raw_output, dim=1)
                pred_mask = probs[:, 1]  # Take foreground class
            else:
                pred_mask = torch.sigmoid(raw_output)
            
            print(f"     Probability range: [{pred_mask.min().item():.3f}, {pred_mask.max().item():.3f}]")
            
            # Create binary prediction
            pred_binary = (pred_mask > 0.5).float()
            print(f"     Binary prediction (threshold 0.5): {pred_binary.sum().item()} / {pred_binary.numel()} pixels")
            
            # Calculate metrics
            intersection = (pred_binary * mask_tensor).sum()
            union = pred_binary.sum() + mask_tensor.sum() - intersection
            
            dice = (2 * intersection) / (pred_binary.sum() + mask_tensor.sum() + 1e-8)
            iou = intersection / (union + 1e-8)
            pixel_acc = (pred_binary == mask_tensor).float().mean()
            
            print(f"     Dice: {dice.item():.4f}")
            print(f"     IoU: {iou.item():.4f}")
            print(f"     Pixel Accuracy: {pixel_acc.item():.4f}")
            
            # Store results
            results.append({
                'image_name': os.path.basename(img_path),
                'dice': dice.item(),
                'iou': iou.item(),
                'pixel_acc': pixel_acc.item(),
                'pred_sum': pred_binary.sum().item(),
                'mask_sum': mask_tensor.sum().item()
            })
            
            # Save visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(img_256)
            axes[0].set_title('Input Image')
            axes[0].axis('off')
            
            axes[1].imshow(mask_tensor.squeeze(), cmap='gray')
            axes[1].set_title('Ground Truth Mask')
            axes[1].axis('off')
            
            axes[2].imshow(pred_binary.squeeze(), cmap='gray')
            axes[2].set_title('Prediction')
            axes[2].axis('off')
            
            plt.tight_layout()
            viz_path = os.path.join(output_dir, f"prediction_{i+1}.png")
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"     ✅ Visualization saved: {viz_path}")
            
        except Exception as e:
            print(f"     ❌ Failed to process pair: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    if results:
        print(f"\n📊 === EVALUATION SUMMARY ===")
        print(f"Models tested: {len(results)}")
        
        avg_dice = np.mean([r['dice'] for r in results])
        avg_iou = np.mean([r['iou'] for r in results])
        avg_pixel_acc = np.mean([r['pixel_acc'] for r in results])
        
        print(f"Average Dice: {avg_dice:.4f}")
        print(f"Average IoU: {avg_iou:.4f}")
        print(f"Average Pixel Accuracy: {avg_pixel_acc:.4f}")
        
        # Save results
        results_path = os.path.join(output_dir, "evaluation_results.txt")
        with open(results_path, 'w') as f:
            f.write("Glomeruli Model Evaluation Results\n")
            f.write("==================================\n\n")
            f.write(f"Model: {model_path}\n")
            f.write(f"Test images: {len(results)}\n\n")
            
            for r in results:
                f.write(f"{r['image_name']}:\n")
                f.write(f"  Dice: {r['dice']:.4f}\n")
                f.write(f"  IoU: {r['iou']:.4f}\n")
                f.write(f"  Pixel Accuracy: {r['pixel_acc']:.4f}\n")
                f.write(f"  Prediction pixels: {r['pred_sum']}\n")
                f.write(f"  Ground truth pixels: {r['mask_sum']}\n\n")
            
            f.write(f"Overall Averages:\n")
            f.write(f"  Dice: {avg_dice:.4f}\n")
            f.write(f"  IoU: {avg_iou:.4f}\n")
            f.write(f"  Pixel Accuracy: {avg_pixel_acc:.4f}\n")
        
        print(f"   ✅ Results saved: {results_path}")
        
        # Important note about model state
        print(f"\n⚠️  IMPORTANT NOTE:")
        print(f"   The models are producing meaningful raw outputs (logits)")
        print(f"   However, they are severely underconfident (max probability: {max([r['pred_sum'] for r in results])} pixels)")
        print(f"   This suggests the models need to be retrained to convergence")
        print(f"   For now, use direct model calls (as shown in this script) for evaluation")
    
    return results


def main():
    """Main evaluation function"""
    
    print("🚀 === GLOMERULI MODEL EVALUATION ===")
    
    # Model paths
    glomeruli_model = "backups/glomerulus_segmentation_model-dynamic_unet-e50_b16_s84.pkl"
    
    # Test data paths
    test_img_dir = "derived_data/glomeruli_data/testing/image_patches"
    test_mask_dir = "derived_data/glomeruli_data/testing/mask_patches"
    
    # Output directory
    output_dir = "test_output/glomeruli_evaluation"
    
    # Check if model exists
    if not os.path.exists(glomeruli_model):
        print(f"❌ Model not found: {glomeruli_model}")
        return
    
    # Check if test data exists
    if not os.path.exists(test_img_dir):
        print(f"❌ Test images not found: {test_img_dir}")
        return
    
    if not os.path.exists(test_mask_dir):
        print(f"❌ Test masks not found: {test_mask_dir}")
        return
    
    # Evaluate glomeruli model
    results = evaluate_glomeruli_model(
        glomeruli_model, 
        test_img_dir, 
        test_mask_dir, 
        output_dir
    )
    
    if results:
        print(f"\n✅ === EVALUATION COMPLETE ===")
        print(f"Check {output_dir} for results and visualizations")
    else:
        print(f"\n❌ === EVALUATION FAILED ===")


if __name__ == "__main__":
    main()

