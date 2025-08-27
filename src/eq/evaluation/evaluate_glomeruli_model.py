#!/usr/bin/env python3
"""
Glomeruli Model Evaluation

This module provides a production-ready evaluation pipeline for glomeruli segmentation models.
Uses direct model calls to bypass FastAI's broken inference pipeline for models trained with training augmentations.

Note: FastAI's learn.predict() method is irreparably broken for models trained with training augmentations
because it inherits those augmentations during inference. The direct model approach is the correct solution.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from eq.utils.logger import get_logger
from eq.utils.model_loader import load_model_safely


class GlomeruliModelEvaluator:
    """
    Evaluates glomeruli segmentation models using direct model calls.
    
    This approach bypasses FastAI's broken inference pipeline for models trained with training augmentations.
    """
    
    def __init__(self, model_path: str, output_dir: str):
        """
        Initialize the evaluator.
        
        Args:
            model_path: Path to the trained model (.pkl file)
            output_dir: Directory to save evaluation results
        """
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = get_logger("eq.glomeruli_evaluation")
        
        # Load the model
        self.logger.info(f"Loading model from {model_path}")
        try:
            self.learn = load_model_safely(model_path)
            self.logger.info("✅ Model loaded successfully")
        except Exception as e:
            self.logger.error(f"❌ Failed to load model: {e}")
            raise
        
        # Set model to evaluation mode
        self.learn.model.eval()
        
        # Determine expected input size from model configuration
        # Most glomeruli models expect 224x224 (based on training code)
        self.expected_size = 224
        
        self.logger.info(f"Model loaded with expected input size: {self.expected_size}x{self.expected_size}")
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess image for model input.
        
        Args:
            image: PIL Image to preprocess
            
        Returns:
            Preprocessed tensor ready for model input
        """
        # Resize to expected input size
        img_resized = image.resize((self.expected_size, self.expected_size), Image.BILINEAR)
        
        # Convert to tensor and normalize
        img_tensor = torch.from_numpy(np.array(img_resized)).float() / 255.0
        
        # Convert to channels-first format (B, C, H, W)
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
        
        return img_tensor
    
    def predict(self, image: Image.Image) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Make prediction using direct model call.
        
        Args:
            image: PIL Image to predict on
            
        Returns:
            Tuple of (raw_output, probabilities, binary_prediction)
        """
        # Preprocess image
        img_tensor = self.preprocess_image(image)
        
        # Make prediction using model directly
        with torch.no_grad():
            raw_output = self.learn.model(img_tensor)
        
        # Apply softmax to get probabilities
        if raw_output.shape[1] == 2:  # Binary segmentation
            probs = torch.softmax(raw_output, dim=1)
            pred_mask = probs[:, 1]  # Take foreground class
        else:
            pred_mask = torch.sigmoid(raw_output)
        
        # Create binary prediction
        pred_binary = (pred_mask > 0.5).float()
        
        return raw_output, pred_mask, pred_binary
    
    def calculate_metrics(self, pred_binary: torch.Tensor, ground_truth: torch.Tensor) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        
        Args:
            pred_binary: Binary prediction tensor
            ground_truth: Ground truth tensor
            
        Returns:
            Dictionary of metrics
        """
        # Ensure both are binary
        pred_binary = (pred_binary > 0.5).float()
        ground_truth = (ground_truth > 0.5).float()
        
        # Calculate intersection and union
        intersection = (pred_binary * ground_truth).sum()
        union = pred_binary.sum() + ground_truth.sum() - intersection
        
        # Dice coefficient
        dice = (2 * intersection) / (pred_binary.sum() + ground_truth.sum() + 1e-8)
        
        # IoU (Jaccard)
        iou = intersection / (union + 1e-8)
        
        # Pixel accuracy
        pixel_acc = (pred_binary == ground_truth).float().mean()
        
        # Precision and recall
        precision = intersection / (pred_binary.sum() + 1e-8)
        recall = intersection / (ground_truth.sum() + 1e-8)
        
        return {
            'dice': dice.item(),
            'iou': iou.item(),
            'pixel_acc': pixel_acc.item(),
            'precision': precision.item(),
            'recall': recall.item(),
            'pred_pixels': pred_binary.sum().item(),
            'gt_pixels': ground_truth.sum().item()
        }
    
    def evaluate_single_image(self, image_path: str, mask_path: str) -> Dict[str, float]:
        """
        Evaluate a single image-mask pair.
        
        Args:
            image_path: Path to input image
            mask_path: Path to ground truth mask
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            # Load image and mask
            image = Image.open(image_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')
            
            # Resize mask to match expected size
            mask_resized = mask.resize((self.expected_size, self.expected_size), Image.NEAREST)
            
            # Convert mask to tensor
            mask_tensor = torch.from_numpy(np.array(mask_resized)).float() / 255.0
            mask_tensor = (mask_tensor > 0.5).float()
            
            # Make prediction
            raw_output, probabilities, pred_binary = self.predict(image)
            
            # Calculate metrics
            metrics = self.calculate_metrics(pred_binary.squeeze(), mask_tensor)
            
            # Add image info
            metrics['image_name'] = os.path.basename(image_path)
            metrics['mask_name'] = os.path.basename(mask_path)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate {image_path}: {e}")
            return {'error': str(e)}
    
    def find_matching_pairs(self, image_dir: str, mask_dir: str) -> List[Tuple[str, str]]:
        """
        Find matching image-mask pairs.
        
        Args:
            image_dir: Directory containing images
            mask_dir: Directory containing masks
            
        Returns:
            List of (image_path, mask_path) tuples
        """
        image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
        mask_files = [f for f in os.listdir(mask_dir) if f.endswith('_mask.jpg')]
        
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
                img_path = os.path.join(image_dir, img_file)
                mask_path = os.path.join(mask_dir, mask_map[base_name])
                matching_pairs.append((img_path, mask_path))
        
        return matching_pairs
    
    def evaluate_dataset(self, image_dir: str, mask_dir: str, max_samples: Optional[int] = None) -> Dict:
        """
        Evaluate the entire dataset.
        
        Args:
            image_dir: Directory containing images
            mask_dir: Directory containing masks
            max_samples: Maximum number of samples to evaluate (None for all)
            
        Returns:
            Dictionary containing evaluation results
        """
        self.logger.info(f"Evaluating dataset: {image_dir} -> {mask_dir}")
        
        # Find matching pairs
        matching_pairs = self.find_matching_pairs(image_dir, mask_dir)
        self.logger.info(f"Found {len(matching_pairs)} image-mask pairs")
        
        if not matching_pairs:
            self.logger.error("No matching pairs found")
            return {'error': 'No matching pairs found'}
        
        # Limit samples if specified
        if max_samples and len(matching_pairs) > max_samples:
            matching_pairs = matching_pairs[:max_samples]
            self.logger.info(f"Limited to {max_samples} samples")
        
        # Evaluate each pair
        results = []
        for i, (img_path, mask_path) in enumerate(matching_pairs):
            self.logger.info(f"Evaluating {i+1}/{len(matching_pairs)}: {os.path.basename(img_path)}")
            
            metrics = self.evaluate_single_image(img_path, mask_path)
            results.append(metrics)
            
            # Save visualization for first few samples
            if i < 5:
                self.save_visualization(img_path, mask_path, metrics, i+1)
        
        # Calculate summary statistics
        summary = self.calculate_summary(results)
        
        # Save results
        self.save_results(results, summary)
        
        return {
            'individual_results': results,
            'summary': summary,
            'total_samples': len(results)
        }
    
    def save_visualization(self, image_path: str, mask_path: str, metrics: Dict, sample_num: int):
        """Save visualization of prediction vs ground truth."""
        try:
            # Load image and mask
            image = Image.open(image_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')
            
            # Make prediction
            raw_output, probabilities, pred_binary = self.predict(image)
            
            # Create visualization
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            
            # Original image
            axes[0].imshow(image.resize((self.expected_size, self.expected_size)))
            axes[0].set_title('Input Image')
            axes[0].axis('off')
            
            # Ground truth mask
            mask_resized = mask.resize((self.expected_size, self.expected_size), Image.NEAREST)
            axes[1].imshow(mask_resized, cmap='gray')
            axes[1].set_title('Ground Truth Mask')
            axes[1].axis('off')
            
            # Probability map
            axes[2].imshow(probabilities.squeeze().cpu().numpy(), cmap='viridis')
            axes[2].set_title('Probability Map')
            axes[2].axis('off')
            
            # Binary prediction
            axes[3].imshow(pred_binary.squeeze().cpu().numpy(), cmap='gray')
            axes[3].set_title(f'Binary Prediction\nDice: {metrics["dice"]:.3f}')
            axes[3].axis('off')
            
            plt.tight_layout()
            
            # Save visualization
            viz_path = self.output_dir / f"prediction_sample_{sample_num}.png"
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Visualization saved: {viz_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save visualization: {e}")
    
    def calculate_summary(self, results: List[Dict]) -> Dict:
        """Calculate summary statistics from individual results."""
        # Filter out error results
        valid_results = [r for r in results if 'error' not in r]
        
        if not valid_results:
            return {'error': 'No valid results'}
        
        # Calculate averages
        summary = {}
        for metric in ['dice', 'iou', 'pixel_acc', 'precision', 'recall']:
            values = [r[metric] for r in valid_results if metric in r]
            if values:
                summary[f'avg_{metric}'] = np.mean(values)
                summary[f'std_{metric}'] = np.std(values)
                summary[f'min_{metric}'] = np.min(values)
                summary[f'max_{metric}'] = np.max(values)
        
        # Add counts
        summary['total_samples'] = len(valid_results)
        summary['error_samples'] = len(results) - len(valid_results)
        
        return summary
    
    def save_results(self, results: List[Dict], summary: Dict):
        """Save evaluation results to files."""
        # Save detailed results
        results_path = self.output_dir / "evaluation_results.txt"
        with open(results_path, 'w') as f:
            f.write("Glomeruli Model Evaluation Results\n")
            f.write("==================================\n\n")
            f.write(f"Model: {self.model_path}\n")
            f.write(f"Total samples: {len(results)}\n\n")
            
            # Individual results
            f.write("Individual Results:\n")
            f.write("-" * 50 + "\n")
            for r in results:
                if 'error' in r:
                    f.write(f"{r.get('image_name', 'Unknown')}: ERROR - {r['error']}\n")
                else:
                    f.write(f"{r['image_name']}:\n")
                    f.write(f"  Dice: {r['dice']:.4f}\n")
                    f.write(f"  IoU: {r['iou']:.4f}\n")
                    f.write(f"  Pixel Accuracy: {r['pixel_acc']:.4f}\n")
                    f.write(f"  Precision: {r['precision']:.4f}\n")
                    f.write(f"  Recall: {r['recall']:.4f}\n")
                    f.write(f"  Prediction pixels: {r['pred_pixels']}\n")
                    f.write(f"  Ground truth pixels: {r['gt_pixels']}\n\n")
            
            # Summary
            f.write("Summary Statistics:\n")
            f.write("-" * 50 + "\n")
            for key, value in summary.items():
                if isinstance(value, float):
                    f.write(f"{key}: {value:.4f}\n")
                else:
                    f.write(f"{key}: {value}\n")
        
        self.logger.info(f"Results saved: {results_path}")
        
        # Save summary as JSON for programmatic access
        import json
        summary_path = self.output_dir / "evaluation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Summary saved: {summary_path}")


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate glomeruli segmentation model")
    parser.add_argument("--model", required=True, help="Path to trained model (.pkl file)")
    parser.add_argument("--image-dir", required=True, help="Directory containing test images")
    parser.add_argument("--mask-dir", required=True, help="Directory containing test masks")
    parser.add_argument("--output-dir", required=True, help="Directory to save results")
    parser.add_argument("--max-samples", type=int, help="Maximum number of samples to evaluate")
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = GlomeruliModelEvaluator(args.model, args.output_dir)
    
    # Evaluate dataset
    results = evaluator.evaluate_dataset(args.image_dir, args.mask_dir, args.max_samples)
    
    if 'error' not in results:
        print("\n✅ === EVALUATION COMPLETE ===")
        print(f"Results saved to: {args.output_dir}")
        print(f"Total samples: {results['total_samples']}")
        
        summary = results['summary']
        if 'avg_dice' in summary:
            print(f"Average Dice: {summary['avg_dice']:.4f}")
            print(f"Average IoU: {summary['avg_iou']:.4f}")
            print(f"Average Pixel Accuracy: {summary['avg_pixel_acc']:.4f}")
    else:
        print("\n❌ === EVALUATION FAILED ===")
        print(f"Error: {results['error']}")


if __name__ == "__main__":
    main()

