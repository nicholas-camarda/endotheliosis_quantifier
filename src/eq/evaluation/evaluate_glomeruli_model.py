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
from eq.data_management.model_loading import load_model_safely


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
        Preprocess image for model input using consolidated prediction core.
        
        Args:
            image: PIL Image to preprocess
            
        Returns:
            Preprocessed tensor ready for model input
        """
        from eq.inference.prediction_core import create_prediction_core
        core = create_prediction_core(self.expected_size)
        return core.preprocess_image(image)
    
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
        Calculate evaluation metrics using consolidated metric functions.
        
        Args:
            pred_binary: Binary prediction tensor
            ground_truth: Ground truth tensor
            
        Returns:
            Dictionary of metrics
        """
        # Ensure both are binary
        pred_binary = (pred_binary > 0.5).float()
        ground_truth = (ground_truth > 0.5).float()
        
        # Convert to numpy for metric calculation
        pred_np = pred_binary.squeeze().cpu().numpy()
        gt_np = ground_truth.squeeze().cpu().numpy()
        
        # Use consolidated metric functions
        from eq.evaluation.segmentation_metrics import (
            dice_coefficient, iou_score, precision_score, recall_score
        )
        
        dice = dice_coefficient(pred_np, gt_np)
        iou = iou_score(pred_np, gt_np)
        precision = precision_score(pred_np, gt_np)
        recall = recall_score(pred_np, gt_np)
        
        # Pixel accuracy via consolidated helper
        from eq.evaluation.segmentation_metrics import pixel_accuracy
        pixel_acc = pixel_accuracy(pred_np, gt_np)
        
        return {
            'dice': dice,
            'iou': iou,
            'pixel_acc': float(pixel_acc),
            'precision': precision,
            'recall': recall,
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
    
    def evaluate_numpy_arrays(self, images: np.ndarray, masks: np.ndarray, output_dir: str, model_name: str) -> Dict[str, float]:
        """
        Evaluate numpy arrays directly (for pipeline compatibility).
        
        Args:
            images: Numpy array of images (N, H, W[, C]) normalized 0-1
            masks: Numpy array of masks (N, H, W[, 1]) binary 0/1
            output_dir: Base output directory
            model_name: Folder name to write artifacts under output_dir
            
        Returns:
            Metrics dict with means/stds and sample count
        """
        output_path = Path(output_dir) / model_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        dice_scores = []
        iou_scores = []
        pixel_accuracies = []
        
        for i in range(len(images)):
            img = images[i]
            true_mask = masks[i]
            
            # Ensure image 3-channel for PIL
            if img.ndim == 3 and img.shape[-1] == 1:
                img = np.repeat(img, 3, axis=-1)
            elif img.ndim == 2:
                img = np.repeat(img[..., None], 3, axis=-1)
            
            # Convert to PIL Image
            img_pil = Image.fromarray((img * 255).astype(np.uint8))
            
            # Make prediction
            raw_output, probabilities, pred_binary = self.predict(img_pil)
            
            # Extract prediction tensor
            pred_mask = pred_binary.squeeze().cpu().numpy()
            
            # Squeeze ground truth to 2D
            true_binary = (np.squeeze(true_mask) > 0.5).astype(np.float32)
            
            # Resize prediction if necessary
            if pred_mask.shape != true_binary.shape:
                from scipy.ndimage import zoom
                scale_factors = [true_binary.shape[j] / pred_mask.shape[j] for j in range(len(true_binary.shape))]
                pred_mask = zoom(pred_mask, scale_factors, order=1)
            
            pred_binary_np = (pred_mask > 0.5).astype(np.float32)
            
            # Calculate metrics using consolidated functions
            from eq.evaluation.segmentation_metrics import (
                dice_coefficient,
                iou_score,
                pixel_accuracy,
            )
            dice = dice_coefficient(pred_binary_np, true_binary)
            iou = iou_score(pred_binary_np, true_binary)
            pix_acc = pixel_accuracy(pred_binary_np, true_binary)
            
            dice_scores.append(dice)
            iou_scores.append(iou)
            pixel_accuracies.append(pix_acc)
        
        # Calculate summary metrics
        metrics = {
            'dice_mean': float(np.mean(dice_scores)) if dice_scores else 0.0,
            'dice_std': float(np.std(dice_scores)) if dice_scores else 0.0,
            'iou_mean': float(np.mean(iou_scores)) if iou_scores else 0.0,
            'iou_std': float(np.std(iou_scores)) if iou_scores else 0.0,
            'pixel_acc_mean': float(np.mean(pixel_accuracies)) if pixel_accuracies else 0.0,
            'pixel_acc_std': float(np.std(pixel_accuracies)) if pixel_accuracies else 0.0,
            'num_samples': int(len(images)),
        }
        
        # Save sample predictions grid
        self.save_sample_predictions_grid_from_arrays(images, masks, max_samples=4)
        
        # Save summary text file
        summary_path = output_path / "evaluation_summary.txt"
        try:
            with open(summary_path, 'w') as f:
                f.write("Glomeruli Segmentation Model Evaluation Summary\n")
                f.write("================================================\n\n")
                if os.getenv('QUICK_TEST', 'false').lower() == 'true':
                    f.write("TESTING RUN - QUICK_TEST MODE\n")
                    f.write("This is a TESTING run. DO NOT use for production.\n\n")
                f.write(f"Model: {model_name}\n")
                f.write(f"Evaluation samples: {len(images)}\n")
                f.write(f"Output directory: {output_path}\n\n")
                f.write("QUANTITATIVE EVALUATION METRICS:\n")
                f.write("================================\n")
                f.write(f"Dice Score:      {metrics['dice_mean']:.4f} ± {metrics['dice_std']:.4f}\n")
                f.write(f"IoU Score:       {metrics['iou_mean']:.4f} ± {metrics['iou_std']:.4f}\n")
                f.write(f"Pixel Accuracy:  {metrics['pixel_acc_mean']:.4f} ± {metrics['pixel_acc_std']:.4f}\n")
                f.write(f"Sample Count:    {metrics['num_samples']}\n")
        except Exception as e:
            self.logger.error(f"Could not write evaluation summary: {e}")
        
        return metrics
    
    def save_sample_predictions_grid_from_arrays(self, images: np.ndarray, masks: np.ndarray, max_samples: int = 4):
        """Save a grid of sample predictions from numpy arrays."""
        try:
            n_show = min(max_samples, len(images))
            if n_show <= 0:
                return
            
            fig, axes = plt.subplots(n_show, 3, figsize=(12, 4 * n_show))
            if n_show == 1:
                axes = np.array([axes])
            
            # Check for QUICK_TEST environment variable
            is_quick_test = os.getenv('QUICK_TEST', 'false').lower() == 'true'
            title = 'Glomeruli Evaluation: Image | Ground Truth | Prediction'
            if is_quick_test:
                title = 'TESTING RUN - ' + title
            fig.suptitle(title, fontsize=16)
            
            for i in range(n_show):
                # Get image and mask
                img = images[i]
                mask = masks[i]
                
                # Ensure image 3-channel for display
                if img.ndim == 3 and img.shape[-1] == 1:
                    img_disp = img.squeeze()
                elif img.ndim == 2:
                    img_disp = img
                else:
                    img_disp = img
                
                # Display original image
                axes[i, 0].imshow(img_disp, cmap='gray')
                axes[i, 0].set_title(f'Image {i+1}')
                axes[i, 0].axis('off')
                
                # Display ground truth mask
                axes[i, 1].imshow(np.squeeze(mask), cmap='gray')
                axes[i, 1].set_title(f'Ground Truth {i+1}')
                axes[i, 1].axis('off')
                
                # Make prediction and display
                if img.ndim == 3 and img.shape[-1] == 1:
                    img_for_pred = np.repeat(img, 3, axis=-1)
                elif img.ndim == 2:
                    img_for_pred = np.repeat(img[..., None], 3, axis=-1)
                else:
                    img_for_pred = img
                
                img_pil = Image.fromarray((img_for_pred * 255).astype(np.uint8))
                raw_output, probabilities, pred_binary = self.predict(img_pil)
                pred_np = pred_binary.squeeze().cpu().numpy()
                axes[i, 2].imshow(pred_np, cmap='gray')
                axes[i, 2].set_title(f'Prediction {i+1}')
                axes[i, 2].axis('off')
            
            plt.tight_layout()
            
            # Save grid visualization
            grid_path = self.output_dir / "sample_predictions_grid.png"
            plt.savefig(grid_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Sample predictions grid saved: {grid_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save sample predictions grid: {e}")
    
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
        image_paths = []
        mask_paths = []
        
        for i, (img_path, mask_path) in enumerate(matching_pairs):
            self.logger.info(f"Evaluating {i+1}/{len(matching_pairs)}: {os.path.basename(img_path)}")
            
            metrics = self.evaluate_single_image(img_path, mask_path)
            results.append(metrics)
            
            # Collect paths for grid visualization
            image_paths.append(img_path)
            mask_paths.append(mask_path)
            
            # Save individual visualization for first few samples
            if i < 5:
                self.save_visualization(img_path, mask_path, metrics, i+1)
        
        # Save sample predictions grid
        self.save_sample_predictions_grid(image_paths, mask_paths, max_samples=4)
        
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
    
    def save_sample_predictions_grid(self, image_paths: List[str], mask_paths: List[str], max_samples: int = 4):
        """Save a grid of sample predictions similar to the legacy evaluator."""
        try:
            n_show = min(max_samples, len(image_paths))
            if n_show <= 0:
                return
            
            fig, axes = plt.subplots(n_show, 3, figsize=(12, 4 * n_show))
            if n_show == 1:
                axes = np.array([axes])
            
            # Check for QUICK_TEST environment variable
            is_quick_test = os.getenv('QUICK_TEST', 'false').lower() == 'true'
            title = 'Glomeruli Evaluation: Image | Ground Truth | Prediction'
            if is_quick_test:
                title = 'TESTING RUN - ' + title
            fig.suptitle(title, fontsize=16)
            
            for i in range(n_show):
                # Load image and mask
                image = Image.open(image_paths[i]).convert('RGB')
                mask = Image.open(mask_paths[i]).convert('L')
                
                # Resize to expected size
                image_resized = image.resize((self.expected_size, self.expected_size), Image.BILINEAR)
                mask_resized = mask.resize((self.expected_size, self.expected_size), Image.NEAREST)
                
                # Display original image
                axes[i, 0].imshow(image_resized)
                axes[i, 0].set_title(f'Image {i+1}')
                axes[i, 0].axis('off')
                
                # Display ground truth mask
                axes[i, 1].imshow(mask_resized, cmap='gray')
                axes[i, 1].set_title(f'Ground Truth {i+1}')
                axes[i, 1].axis('off')
                
                # Make prediction and display
                raw_output, probabilities, pred_binary = self.predict(image)
                pred_np = pred_binary.squeeze().cpu().numpy()
                axes[i, 2].imshow(pred_np, cmap='gray')
                axes[i, 2].set_title(f'Prediction {i+1}')
                axes[i, 2].axis('off')
            
            plt.tight_layout()
            
            # Save grid visualization
            grid_path = self.output_dir / "sample_predictions_grid.png"
            plt.savefig(grid_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Sample predictions grid saved: {grid_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save sample predictions grid: {e}")
    
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
            
            # Check for QUICK_TEST environment variable
            is_quick_test = os.getenv('QUICK_TEST', 'false').lower() == 'true'
            if is_quick_test:
                f.write("TESTING RUN - QUICK_TEST MODE\n")
                f.write("This is a TESTING run. DO NOT use for production.\n\n")
            
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
        
        # Show QUICK_TEST status
        is_quick_test = os.getenv('QUICK_TEST', 'false').lower() == 'true'
        if is_quick_test:
            print("⚠️  QUICK_TEST MODE - This was a testing run")
        
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
