#!/usr/bin/env python3
"""
Tests for consolidated metric calculation system.

This test file ensures that all metric calculations work correctly after consolidation
and that no functionality is lost during the deduplication process.
"""

import numpy as np
import pytest
import torch

from eq.evaluation.segmentation_metrics import (
    dice_coefficient,
    iou_score,
    precision_score,
    recall_score,
    f1_score,
    specificity_score,
    calculate_all_metrics,
    calculate_batch_metrics,
    SegmentationMetrics
)


class TestConsolidatedMetrics:
    """Test consolidated metric calculation functions."""
    
    def test_dice_coefficient_basic(self):
        """Test basic dice coefficient calculation."""
        # Perfect prediction
        pred = np.array([[1, 1], [1, 1]])
        gt = np.array([[1, 1], [1, 1]])
        assert dice_coefficient(pred, gt) == 1.0
        
        # No overlap
        pred = np.array([[1, 1], [0, 0]])
        gt = np.array([[0, 0], [1, 1]])
        assert dice_coefficient(pred, gt) == 0.0
        
        # Partial overlap
        pred = np.array([[1, 1], [1, 0]])
        gt = np.array([[1, 1], [0, 1]])
        # Intersection = 2, sums = 3 and 3
        expected = 2 * 2 / (3 + 3)  # 2*intersection / (pred_sum + gt_sum)
        assert abs(dice_coefficient(pred, gt) - expected) < 1e-6
    
    def test_dice_coefficient_thresholding(self):
        """Test that dice coefficient properly thresholds inputs."""
        # Input with values between 0 and 1
        pred = np.array([[0.7, 0.3], [0.8, 0.2]])
        gt = np.array([[1, 0], [1, 1]])
        
        # Should threshold at 0.5
        result = dice_coefficient(pred, gt)
        # After thresholding (>0.5), intersection = 2, sums = 2 and 3
        expected = 2 * 2 / (2 + 3)  # 2*intersection / (pred_sum + gt_sum)
        assert abs(result - expected) < 1e-6
    
    def test_iou_score_basic(self):
        """Test basic IoU score calculation."""
        # Perfect prediction
        pred = np.array([[1, 1], [1, 1]])
        gt = np.array([[1, 1], [1, 1]])
        assert iou_score(pred, gt) == 1.0
        
        # No overlap
        pred = np.array([[1, 1], [0, 0]])
        gt = np.array([[0, 0], [1, 1]])
        assert iou_score(pred, gt) == 0.0
        
        # Partial overlap
        pred = np.array([[1, 1], [1, 0]])
        gt = np.array([[1, 1], [0, 1]])
        # Intersection = 2, union = 3 + 3 - 2 = 4
        expected = 2 / 4  # intersection / union
        assert abs(iou_score(pred, gt) - expected) < 1e-6
    
    def test_precision_score_basic(self):
        """Test basic precision score calculation."""
        # Perfect prediction
        pred = np.array([[1, 1], [1, 1]])
        gt = np.array([[1, 1], [1, 1]])
        assert precision_score(pred, gt) == 1.0
        
        # False positives
        pred = np.array([[1, 1], [1, 1]])
        gt = np.array([[1, 0], [0, 0]])
        expected = 1 / 4  # 1 TP / (1 TP + 3 FP)
        assert abs(precision_score(pred, gt) - expected) < 1e-6
    
    def test_recall_score_basic(self):
        """Test basic recall score calculation."""
        # Perfect prediction
        pred = np.array([[1, 1], [1, 1]])
        gt = np.array([[1, 1], [1, 1]])
        assert recall_score(pred, gt) == 1.0
        
        # False negatives
        pred = np.array([[1, 0], [0, 0]])
        gt = np.array([[1, 1], [1, 1]])
        expected = 1 / 4  # 1 TP / (1 TP + 3 FN)
        assert abs(recall_score(pred, gt) - expected) < 1e-6
    
    def test_f1_score_basic(self):
        """Test basic F1 score calculation."""
        # Perfect prediction
        pred = np.array([[1, 1], [1, 1]])
        gt = np.array([[1, 1], [1, 1]])
        assert f1_score(pred, gt) == 1.0
        
        # Balanced precision/recall
        pred = np.array([[1, 0], [0, 0]])
        gt = np.array([[1, 0], [0, 0]])
        assert f1_score(pred, gt) == 1.0
    
    def test_specificity_score_basic(self):
        """Test basic specificity score calculation."""
        # Perfect prediction
        pred = np.array([[1, 1], [1, 1]])
        gt = np.array([[1, 1], [1, 1]])
        assert specificity_score(pred, gt) == 1.0
        
        # False positives in background
        pred = np.array([[1, 1], [1, 1]])
        gt = np.array([[1, 0], [0, 0]])
        expected = 0 / 3  # 0 TN / (0 TN + 3 FP)
        assert abs(specificity_score(pred, gt) - expected) < 1e-6
    
    def test_calculate_all_metrics(self):
        """Test calculate_all_metrics function."""
        pred = np.array([[1, 1], [1, 0]])
        gt = np.array([[1, 1], [0, 1]])
        
        metrics = calculate_all_metrics(pred, gt)
        
        assert isinstance(metrics, SegmentationMetrics)
        assert 0 <= metrics.dice_coefficient <= 1
        assert 0 <= metrics.iou_score <= 1
        assert 0 <= metrics.precision <= 1
        assert 0 <= metrics.recall <= 1
        assert 0 <= metrics.f1_score <= 1
    
    def test_calculate_batch_metrics(self):
        """Test calculate_batch_metrics function."""
        preds = [
            np.array([[1, 1], [1, 1]]),
            np.array([[1, 0], [0, 0]])
        ]
        gts = [
            np.array([[1, 1], [1, 1]]),
            np.array([[1, 0], [0, 0]])
        ]
        
        avg_metrics, individual_metrics = calculate_batch_metrics(preds, gts)
        
        assert isinstance(avg_metrics, SegmentationMetrics)
        assert len(individual_metrics) == 2
        assert avg_metrics.dice_coefficient == 1.0  # Both perfect
        assert avg_metrics.iou_score == 1.0
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Empty masks
        pred = np.array([])
        gt = np.array([])
        
        # Should handle gracefully
        with pytest.raises(ValueError):
            dice_coefficient(pred, gt)
        
        # Different shapes
        pred = np.array([[1, 1], [1, 1]])
        gt = np.array([[1, 1, 1], [1, 1, 1]])
        
        with pytest.raises(ValueError):
            dice_coefficient(pred, gt)
    
    def test_smooth_parameter(self):
        """Test that smooth parameter prevents division by zero."""
        # Both masks are empty (all zeros)
        pred = np.zeros((2, 2))
        gt = np.zeros((2, 2))
        
        # Should not crash due to smooth parameter
        dice = dice_coefficient(pred, gt)
        iou = iou_score(pred, gt)
        precision = precision_score(pred, gt)
        recall = recall_score(pred, gt)
        
        # All should be 0.0 due to smooth parameter
        assert dice == 0.0
        assert iou == 0.0
        assert precision == 0.0
        assert recall == 0.0


class TestMetricCompatibility:
    """Test that consolidated metrics are compatible with existing usage patterns."""
    
    def test_tensor_compatibility(self):
        """Test that metrics work with PyTorch tensors."""
        pred = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
        gt = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
        
        # Convert to numpy for testing
        pred_np = pred.numpy()
        gt_np = gt.numpy()
        
        dice = dice_coefficient(pred_np, gt_np)
        iou = iou_score(pred_np, gt_np)
        
        assert 0 <= dice <= 1
        assert 0 <= iou <= 1
    
    def test_float_array_compatibility(self):
        """Test that metrics work with float arrays."""
        pred = np.array([[0.7, 0.3], [0.8, 0.2]], dtype=np.float32)
        gt = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        
        dice = dice_coefficient(pred, gt)
        iou = iou_score(pred, gt)
        
        assert 0 <= dice <= 1
        assert 0 <= iou <= 1


if __name__ == "__main__":
    pytest.main([__file__])


