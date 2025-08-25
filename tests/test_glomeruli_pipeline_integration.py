#!/usr/bin/env python3
"""Integration tests for glomeruli evaluation pipeline."""

import os
import tempfile
from pathlib import Path

import numpy as np

from eq.evaluation.glomeruli_evaluator import evaluate_glomeruli_model


class MockLearner:
    """Mock learner for testing pipeline integration."""
    def __init__(self, pred_masks):
        self._preds = pred_masks
        self._idx = 0

    def predict(self, pil_image):
        pred = self._preds[self._idx]
        self._idx = (self._idx + 1) % len(self._preds)
        return (None, pred, None)


def test_pipeline_integration_with_error_handling():
    """Test that pipeline integration handles errors gracefully."""
    
    # Test data
    val_images = np.random.rand(2, 64, 64, 3).astype(np.float32)
    val_masks = np.random.randint(0, 2, (2, 64, 64)).astype(np.uint8)
    pred_masks = np.random.rand(2, 64, 64).astype(np.float32)
    
    learn = MockLearner(pred_masks)
    
    # Test with temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Test successful evaluation
            metrics = evaluate_glomeruli_model(
                learn=learn,
                val_images=val_images,
                val_masks=val_masks,
                output_dir=temp_dir,
                model_name="test_integration"
            )
            
            # Verify metrics structure
            required_keys = ['dice_mean', 'iou_mean', 'pixel_acc_mean', 'num_samples']
            for key in required_keys:
                assert key in metrics, f"Missing metric: {key}"
            
            # Verify artifacts were created
            output_path = Path(temp_dir) / "test_integration"
            assert output_path.exists(), "Output directory should exist"
            
            # Check for evaluation artifacts
            artifacts = list(output_path.glob("*"))
            assert len(artifacts) > 0, "Should create evaluation artifacts"
            
            print("✅ Pipeline integration test passed")
            
        except Exception as e:
            print(f"❌ Pipeline integration test failed: {e}")
            raise


def test_quick_test_mode_indicators():
    """Test that QUICK_TEST mode properly adds testing indicators."""
    
    # Set QUICK_TEST environment variable
    original_quick_test = os.environ.get('QUICK_TEST', 'false')
    os.environ['QUICK_TEST'] = 'true'
    
    try:
        val_images = np.random.rand(1, 32, 32, 3).astype(np.float32)
        val_masks = np.random.randint(0, 2, (1, 32, 32)).astype(np.uint8)
        pred_masks = np.random.rand(1, 32, 32).astype(np.float32)
        
        learn = MockLearner(pred_masks)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics = evaluate_glomeruli_model(
                learn=learn,
                val_images=val_images,
                val_masks=val_masks,
                output_dir=temp_dir,
                model_name="test_quick_test"
            )
            
            # Check that evaluation summary contains testing indicator
            summary_path = Path(temp_dir) / "test_quick_test" / "evaluation_summary.txt"
            assert summary_path.exists(), "Evaluation summary should exist"
            
            with open(summary_path, 'r') as f:
                content = f.read()
                assert "TESTING RUN" in content, "Should contain testing indicator"
                assert "QUICK_TEST MODE" in content, "Should contain QUICK_TEST indicator"
            
            print("✅ QUICK_TEST mode indicators test passed")
            
    finally:
        # Restore original environment
        os.environ['QUICK_TEST'] = original_quick_test


if __name__ == "__main__":
    test_pipeline_integration_with_error_handling()
    test_quick_test_mode_indicators()
    print("✅ All integration tests passed")
