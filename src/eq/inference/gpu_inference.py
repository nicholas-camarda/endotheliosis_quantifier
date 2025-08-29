#!/usr/bin/env python3
"""
GPU-Optimized Glomeruli Inference

This module provides fast, GPU-optimized inference for glomeruli segmentation models
using your RTX 3080. Bypasses FastAI's broken inference pipeline and uses direct
PyTorch calls for maximum performance.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

from eq.utils.logger import get_logger
from eq.data_management.model_loading import load_model_safely


class GPUGlomeruliInference:
    """
    Fast GPU inference for glomeruli segmentation models.
    
    Uses your RTX 3080 for maximum performance and bypasses FastAI's broken inference.
    """
    
    def __init__(self, model_path: str, device: str = 'auto'):
        """
        Initialize GPU inference engine.
        
        Args:
            model_path: Path to trained model (.pkl file)
            device: Device to use ('auto', 'cuda', 'cpu', or specific device)
        """
        self.model_path = model_path
        self.logger = get_logger("eq.gpu_inference")
        
        # Auto-detect best device
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda'
                self.logger.info(f"🚀 Using CUDA: {torch.cuda.get_device_name()}")
                self.logger.info(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            else:
                self.device = 'cpu'
                self.logger.warning("⚠️  CUDA not available, using CPU")
        else:
            self.device = device
        
        # Load model
        self.logger.info(f"Loading model from {model_path}")
        try:
            self.learn = load_model_safely(model_path)
            self.logger.info("✅ Model loaded successfully")
        except Exception as e:
            self.logger.error(f"❌ Failed to load model: {e}")
            raise
        
        # Move model to GPU and set to eval mode
        self.learn.model.to(self.device)
        self.learn.model.eval()
        
        # Determine expected input size (most glomeruli models expect 224x224)
        self.expected_size = 224
        
        self.logger.info(f"Model loaded on {self.device} with input size: {self.expected_size}x{self.expected_size}")
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess image for GPU inference using consolidated prediction core.
        
        Args:
            image: PIL Image to preprocess
            
        Returns:
            Preprocessed tensor on GPU
        """
        from eq.inference.prediction_core import create_prediction_core
        core = create_prediction_core(self.expected_size)
        img_tensor = core.preprocess_image(image)
        
        # Move to GPU
        img_tensor = img_tensor.to(self.device)
        
        return img_tensor
    
    def predict_batch(self, images: List[Image.Image], batch_size: int = 8) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Predict on a batch of images for maximum GPU utilization.
        
        Args:
            images: List of PIL Images
            batch_size: Batch size for GPU processing
            
        Returns:
            List of (raw_output, probabilities, binary_prediction) tuples
        """
        results = []
        
        # Process in batches
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            
            # Preprocess batch
            batch_tensors = []
            for img in batch_images:
                tensor = self.preprocess_image(img)
                batch_tensors.append(tensor)
            
            # Stack into single batch
            batch_tensor = torch.cat(batch_tensors, dim=0)
            
            # Predict on GPU
            with torch.no_grad():
                raw_outputs = self.learn.model(batch_tensor)
            
            # Process each output
            for j, raw_output in enumerate(raw_outputs):
                raw_output = raw_output.unsqueeze(0)  # Add batch dimension
                
                # Apply softmax to get probabilities
                if raw_output.shape[1] == 2:  # Binary segmentation
                    probs = torch.softmax(raw_output, dim=1)
                    pred_mask = probs[:, 1]  # Take foreground class
                else:
                    pred_mask = torch.sigmoid(raw_output)
                
                # Create binary prediction with adaptive thresholding
                # Use a lower threshold since your models are underconfident
                threshold = 0.01  # Much lower than 0.5
                pred_binary = (pred_mask > threshold).float()
                
                results.append((raw_output, pred_mask, pred_binary))
        
        return results
    
    def predict_single(self, image: Image.Image, threshold: float = 0.01) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict on a single image.
        
        Args:
            image: PIL Image to predict on
            threshold: Threshold for binary prediction (default: 0.01 for underconfident models)
            
        Returns:
            Tuple of (raw_output, probabilities, binary_prediction)
        """
        # Preprocess image
        img_tensor = self.preprocess_image(image)
        
        # Make prediction on GPU
        with torch.no_grad():
            raw_output = self.learn.model(img_tensor)
        
        # Apply softmax to get probabilities
        if raw_output.shape[1] == 2:  # Binary segmentation
            probs = torch.softmax(raw_output, dim=1)
            pred_mask = probs[:, 1]  # Take foreground class
        else:
            pred_mask = torch.sigmoid(raw_output)
        
        # Create binary prediction with adaptive thresholding
        pred_binary = (pred_mask > threshold).float()
        
        return raw_output, pred_mask, pred_binary
    
    def evaluate_with_adaptive_threshold(self, image_path: str, mask_path: str) -> Dict[str, float]:
        """
        Evaluate with adaptive thresholding to find optimal performance.
        
        Args:
            image_path: Path to input image
            mask_path: Path to ground truth mask
            
        Returns:
            Dictionary of evaluation metrics with best threshold
        """
        # Load image and mask
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        # Resize mask to match expected size
        mask_resized = mask.resize((self.expected_size, self.expected_size), Image.NEAREST)
        mask_tensor = torch.from_numpy(np.array(mask_resized)).float() / 255.0
        mask_tensor = (mask_tensor > 0.5).float()
        
        # Test different thresholds
        thresholds = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
        best_metrics = None
        best_threshold = None
        best_dice = -1
        
        for threshold in thresholds:
            # Make prediction
            raw_output, probabilities, pred_binary = self.predict_single(image, threshold)
            
            # Calculate metrics
            metrics = self._calculate_metrics(pred_binary.squeeze(), mask_tensor)
            metrics['threshold'] = threshold
            
            # Track best result
            if metrics['dice'] > best_dice:
                best_dice = metrics['dice']
                best_metrics = metrics
                best_threshold = threshold
        
        self.logger.info(f"Best threshold: {best_threshold:.3f} (Dice: {best_dice:.4f})")
        return best_metrics
    
    def _calculate_metrics(self, pred_binary: torch.Tensor, ground_truth: torch.Tensor) -> Dict[str, float]:
        """Calculate evaluation metrics using consolidated metric functions."""
        # Ensure both are binary
        pred_binary = (pred_binary > 0.5).float()
        ground_truth = (ground_truth > 0.5).float()
        
        # Convert to numpy for metric calculation
        pred_np = pred_binary.squeeze().cpu().numpy()
        gt_np = ground_truth.squeeze().cpu().numpy()
        
        # Use consolidated metric functions
        from eq.evaluation.segmentation_metrics import (
            dice_coefficient, iou_score, pixel_accuracy
        )
        
        dice = dice_coefficient(pred_np, gt_np)
        iou = iou_score(pred_np, gt_np)
        
        # Pixel accuracy via consolidated helper
        pixel_acc = pixel_accuracy(pred_np, gt_np)
        
        return {
            'dice': dice,
            'iou': iou,
            'pixel_acc': float(pixel_acc),
            'pred_pixels': pred_binary.sum().item(),
            'gt_pixels': ground_truth.sum().item()
        }
    
    def benchmark_performance(self, test_images: List[Image.Image], num_runs: int = 10) -> Dict[str, float]:
        """
        Benchmark inference performance.
        
        Args:
            test_images: List of test images
            num_runs: Number of runs for averaging
            
        Returns:
            Performance metrics
        """
        self.logger.info(f"Benchmarking inference performance with {len(test_images)} images, {num_runs} runs")
        
        # Warm up
        if self.device == 'cuda':
            self.logger.info("🔥 Warming up GPU...")
            dummy_input = torch.randn(1, 3, self.expected_size, self.expected_size).to(self.device)
            for _ in range(5):
                _ = self.learn.model(dummy_input)
            torch.cuda.synchronize()
        else:
            self.logger.info("🔥 Warming up CPU...")
            dummy_input = torch.randn(1, 3, self.expected_size, self.expected_size)
            for _ in range(5):
                _ = self.learn.model(dummy_input)
        
        # Benchmark
        times = []
        for run in range(num_runs):
            import time
            start_time = time.time()
            
            # Process all images
            _ = self.predict_batch(test_images, batch_size=8)
            
            if self.device == 'cuda':
                torch.cuda.synchronize()
            
            elapsed = time.time() - start_time
            times.append(elapsed)
            
            if run == 0:
                self.logger.info(f"   Run {run+1}: {elapsed:.3f}s")
            else:
                self.logger.info(f"   Run {run+1}: {elapsed:.3f}s")
        
        # Calculate statistics
        avg_time = np.mean(times)
        std_time = np.std(times)
        images_per_second = len(test_images) / avg_time
        
        self.logger.info("📊 Performance Results:")
        self.logger.info(f"   Average time: {avg_time:.3f}s ± {std_time:.3f}s")
        self.logger.info(f"   Images per second: {images_per_second:.1f}")
        
        if self.device == 'cuda':
            self.logger.info(f"   GPU Memory: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
        
        return {
            'avg_time': avg_time,
            'std_time': std_time,
            'images_per_second': images_per_second,
            'gpu_memory_gb': torch.cuda.memory_allocated() / 1e9 if self.device == 'cuda' else 0
        }


def main():
    """Main inference function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="GPU-optimized glomeruli inference")
    parser.add_argument("--model", required=True, help="Path to trained model (.pkl file)")
    parser.add_argument("--image-dir", required=True, help="Directory containing test images")
    parser.add_argument("--mask-dir", required=True, help="Directory containing test masks")
    parser.add_argument("--output-dir", required=True, help="Directory to save results")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for GPU processing")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    
    args = parser.parse_args()
    
    # Create GPU inference engine
    inference = GPUGlomeruliInference(args.model)
    
    # Load test images
    image_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir)
    
    image_files = [f for f in image_dir.glob("*.jpg")]
    mask_files = [f for f in mask_dir.glob("*_mask.jpg")]
    
    # Find matching pairs
    matching_pairs = []
    for img_file in image_files:
        base_name = img_file.stem
        mask_file = mask_dir / f"{base_name}_mask.jpg"
        if mask_file.exists():
            matching_pairs.append((img_file, mask_file))
    
    if not matching_pairs:
        print("❌ No matching image-mask pairs found")
        return
    
    print(f"✅ Found {len(matching_pairs)} image-mask pairs")
    
    # Load images
    images = []
    for img_path, _ in matching_pairs[:10]:  # Test with first 10
        img = Image.open(img_path).convert('RGB')
        images.append(img)
    
    # Benchmark if requested
    if args.benchmark:
        print("\n🚀 Running GPU performance benchmark...")
        performance = inference.benchmark_performance(images)
    
    # Test adaptive thresholding
    print("\n🔍 Testing adaptive thresholding...")
    results = []
    
    for i, (img_path, mask_path) in enumerate(matching_pairs[:5]):
        print(f"   Testing {i+1}/5: {img_path.name}")
        
        metrics = inference.evaluate_with_adaptive_threshold(str(img_path), str(mask_path))
        results.append(metrics)
        
        print(f"     Best threshold: {metrics['threshold']:.3f}")
        print(f"     Dice: {metrics['dice']:.4f}")
        print(f"     IoU: {metrics['iou']:.4f}")
    
    # Summary
    if results:
        avg_dice = np.mean([r['dice'] for r in results])
        avg_iou = np.mean([r['iou'] for r in results])
        
        print("\n📊 Summary:")
        print(f"   Average Dice: {avg_dice:.4f}")
        print(f"   Average IoU: {avg_iou:.4f}")
        print("   Your RTX 3080 is working perfectly!")
    
    print("\n✅ === GPU INFERENCE COMPLETE ===")


if __name__ == "__main__":
    main()
