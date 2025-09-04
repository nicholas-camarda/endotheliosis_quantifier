#!/usr/bin/env python3
"""Segmentation metrics for evaluating model performance."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class SegmentationMetrics:
    """Container for segmentation evaluation metrics."""
    dice_coefficient: float
    iou_score: float
    precision: float
    recall: float
    f1_score: float
    hausdorff_distance: Optional[float] = None
    boundary_accuracy: Optional[float] = None
    
    def __str__(self) -> str:
        return (f"SegmentationMetrics(dice={self.dice_coefficient:.4f}, "
                f"iou={self.iou_score:.4f}, precision={self.precision:.4f}, "
                f"recall={self.recall:.4f}, f1={self.f1_score:.4f})")


def dice_coefficient(predicted: np.ndarray, ground_truth: np.ndarray, smooth: float = 0.0) -> float:
    """
    Calculate Dice coefficient (Sørensen-Dice coefficient).
    
    Args:
        predicted: Predicted binary mask
        ground_truth: Ground truth binary mask
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        Dice coefficient score (0-1, higher is better)
    """
    # Validate inputs
    if predicted.shape != ground_truth.shape:
        raise ValueError('Predicted and ground truth must have the same shape')
    if predicted.size == 0:
        raise ValueError('Predicted and ground truth must be non-empty')
    
    # Ensure masks are binary
    pred_binary = (predicted > 0.5).astype(np.uint8)
    gt_binary = (ground_truth > 0.5).astype(np.uint8)
    
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    denom = pred_binary.sum() + gt_binary.sum()
    if denom == 0:
        return 0.0
    dice = (2 * intersection) / denom
    
    return float(dice)


def iou_score(predicted: np.ndarray, ground_truth: np.ndarray, smooth: float = 0.0) -> float:
    """
    Calculate Intersection over Union (IoU) score, also known as Jaccard index.
    
    Args:
        predicted: Predicted binary mask
        ground_truth: Ground truth binary mask
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        IoU score (0-1, higher is better)
    """
    # Validate inputs
    if predicted.shape != ground_truth.shape:
        raise ValueError('Predicted and ground truth must have the same shape')
    if predicted.size == 0:
        raise ValueError('Predicted and ground truth must be non-empty')
    
    # Ensure masks are binary
    pred_binary = (predicted > 0.5).astype(np.uint8)
    gt_binary = (ground_truth > 0.5).astype(np.uint8)
    
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()
    if union == 0:
        return 0.0
    
    iou = intersection / union
    return float(iou)


def precision_score(predicted: np.ndarray, ground_truth: np.ndarray, smooth: float = 0.0) -> float:
    """
    Calculate precision score for binary segmentation.
    
    Args:
        predicted: Predicted binary mask
        ground_truth: Ground truth binary mask
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        Precision score (0-1, higher is better)
    """
    # Validate inputs
    if predicted.shape != ground_truth.shape:
        raise ValueError('Predicted and ground truth must have the same shape')
    if predicted.size == 0:
        raise ValueError('Predicted and ground truth must be non-empty')
    
    # Ensure masks are binary
    pred_binary = (predicted > 0.5).astype(np.uint8)
    gt_binary = (ground_truth > 0.5).astype(np.uint8)
    
    tp = np.logical_and(pred_binary, gt_binary).sum()
    fp = pred_binary.sum() - tp
    denom = tp + fp
    if denom == 0:
        return 0.0
    
    precision = tp / denom
    return float(precision)


def recall_score(predicted: np.ndarray, ground_truth: np.ndarray, smooth: float = 0.0) -> float:
    """
    Calculate recall score (sensitivity) for binary segmentation.
    
    Args:
        predicted: Predicted binary mask
        ground_truth: Ground truth binary mask
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        Recall score (0-1, higher is better)
    """
    # Validate inputs
    if predicted.shape != ground_truth.shape:
        raise ValueError('Predicted and ground truth must have the same shape')
    if predicted.size == 0:
        raise ValueError('Predicted and ground truth must be non-empty')
    
    # Ensure masks are binary
    pred_binary = (predicted > 0.5).astype(np.uint8)
    gt_binary = (ground_truth > 0.5).astype(np.uint8)
    
    tp = np.logical_and(pred_binary, gt_binary).sum()
    fn = gt_binary.sum() - tp
    denom = tp + fn
    if denom == 0:
        return 0.0
    
    recall = tp / denom
    return float(recall)


def f1_score(predicted: np.ndarray, ground_truth: np.ndarray, smooth: float = 1e-8) -> float:
    """
    Calculate F1 score for binary segmentation.
    
    Args:
        predicted: Predicted binary mask
        ground_truth: Ground truth binary mask
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        F1 score (0-1, higher is better)
    """
    precision = precision_score(predicted, ground_truth, smooth)
    recall = recall_score(predicted, ground_truth, smooth)
    
    f1 = (2 * precision * recall + smooth) / (precision + recall + smooth)
    return float(f1)


def pixel_accuracy(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Calculate pixel accuracy for binary segmentation masks.

    Args:
        predicted: Predicted mask (binary or probabilities)
        ground_truth: Ground truth mask (binary or probabilities)

    Returns:
        Pixel accuracy (0-1, higher is better)
    """
    pred_binary = (predicted > 0.5).astype(np.uint8)
    gt_binary = (ground_truth > 0.5).astype(np.uint8)
    return float(np.mean(pred_binary == gt_binary))


def specificity_score(predicted: np.ndarray, ground_truth: np.ndarray, smooth: float = 1e-8) -> float:
    """
    Calculate specificity score for binary segmentation.
    
    Args:
        predicted: Predicted binary mask
        ground_truth: Ground truth binary mask
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        Specificity score (0-1, higher is better)
    """
    # Ensure masks are binary
    pred_binary = (predicted > 0.5).astype(np.uint8)
    gt_binary = (ground_truth > 0.5).astype(np.uint8)
    
    tn = np.logical_and(1 - pred_binary, 1 - gt_binary).sum()
    fp = pred_binary.sum() - np.logical_and(pred_binary, gt_binary).sum()
    
    specificity = (tn + smooth) / (tn + fp + smooth)
    return float(specificity)


def hausdorff_distance(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Calculate Hausdorff distance between predicted and ground truth masks.
    
    Args:
        predicted: Predicted binary mask
        ground_truth: Ground truth binary mask
        
    Returns:
        Hausdorff distance (lower is better)
    """
    try:
        from scipy.spatial.distance import directed_hausdorff

        # Ensure masks are binary
        pred_binary = (predicted > 0.5).astype(np.uint8)
        gt_binary = (ground_truth > 0.5).astype(np.uint8)
        
        # Get coordinates of positive pixels
        pred_coords = np.column_stack(np.where(pred_binary))
        gt_coords = np.column_stack(np.where(gt_binary))
        
        if len(pred_coords) == 0 or len(gt_coords) == 0:
            return float('inf')
        
        # Calculate bidirectional Hausdorff distance
        hd1 = directed_hausdorff(pred_coords, gt_coords)[0]
        hd2 = directed_hausdorff(gt_coords, pred_coords)[0]
        
        return float(max(hd1, hd2))
    
    except ImportError:
        # Fallback if scipy is not available
        return float('nan')


def calculate_all_metrics(predicted: np.ndarray, ground_truth: np.ndarray, 
                         include_hausdorff: bool = False) -> SegmentationMetrics:
    """
    Calculate all segmentation metrics for a single prediction.
    
    Args:
        predicted: Predicted binary mask
        ground_truth: Ground truth binary mask
        include_hausdorff: Whether to calculate Hausdorff distance (computationally expensive)
        
    Returns:
        SegmentationMetrics object with all calculated metrics
    """
    dice = dice_coefficient(predicted, ground_truth)
    iou = iou_score(predicted, ground_truth)
    precision = precision_score(predicted, ground_truth)
    recall = recall_score(predicted, ground_truth)
    f1 = f1_score(predicted, ground_truth)
    
    hausdorff_dist = None
    if include_hausdorff:
        hausdorff_dist = hausdorff_distance(predicted, ground_truth)
    
    return SegmentationMetrics(
        dice_coefficient=dice,
        iou_score=iou,
        precision=precision,
        recall=recall,
        f1_score=f1,
        hausdorff_distance=hausdorff_dist
    )


def calculate_batch_metrics(predicted_masks: List[np.ndarray], 
                           ground_truth_masks: List[np.ndarray],
                           include_hausdorff: bool = False) -> Tuple[SegmentationMetrics, List[SegmentationMetrics]]:
    """
    Calculate segmentation metrics for a batch of predictions.
    
    Args:
        predicted_masks: List of predicted binary masks
        ground_truth_masks: List of ground truth binary masks
        include_hausdorff: Whether to calculate Hausdorff distance
        
    Returns:
        Tuple of (average_metrics, individual_metrics)
    """
    if len(predicted_masks) != len(ground_truth_masks):
        raise ValueError("Number of predicted masks must match number of ground truth masks")
    
    individual_metrics = []
    dice_scores = []
    iou_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    hausdorff_distances = []
    
    for pred_mask, gt_mask in zip(predicted_masks, ground_truth_masks):
        metrics = calculate_all_metrics(pred_mask, gt_mask, include_hausdorff)
        individual_metrics.append(metrics)
        
        dice_scores.append(metrics.dice_coefficient)
        iou_scores.append(metrics.iou_score)
        precision_scores.append(metrics.precision)
        recall_scores.append(metrics.recall)
        f1_scores.append(metrics.f1_score)
        
        if metrics.hausdorff_distance is not None:
            hausdorff_distances.append(metrics.hausdorff_distance)
    
    # Calculate average metrics
    avg_hausdorff = None
    if hausdorff_distances:
        valid_distances = [d for d in hausdorff_distances if not np.isnan(d) and not np.isinf(d)]
        if valid_distances:
            avg_hausdorff = np.mean(valid_distances)
    
    average_metrics = SegmentationMetrics(
        dice_coefficient=float(np.mean(dice_scores)),
        iou_score=float(np.mean(iou_scores)),
        precision=float(np.mean(precision_scores)),
        recall=float(np.mean(recall_scores)),
        f1_score=float(np.mean(f1_scores)),
        hausdorff_distance=float(avg_hausdorff) if avg_hausdorff is not None else None
    )
    
    return average_metrics, individual_metrics


def mean_iou_multiclass(predicted: np.ndarray, ground_truth: np.ndarray, num_classes: int) -> float:
    """
    Calculate mean IoU for multiclass segmentation.
    
    Args:
        predicted: Predicted mask with class indices
        ground_truth: Ground truth mask with class indices
        num_classes: Number of classes
        
    Returns:
        Mean IoU score across all classes
    """
    iou_scores = []
    
    for class_id in range(num_classes):
        pred_class = (predicted == class_id).astype(np.uint8)
        gt_class = (ground_truth == class_id).astype(np.uint8)
        
        intersection = np.logical_and(pred_class, gt_class).sum()
        union = np.logical_or(pred_class, gt_class).sum()
        
        if union == 0:
            # If no pixels of this class exist in either prediction or ground truth
            iou_class = 1.0 if intersection == 0 else 0.0
        else:
            iou_class = intersection / union
        
        iou_scores.append(iou_class)
    
    return float(np.mean(iou_scores))
