#!/usr/bin/env python3
"""Quantification metrics for endotheliosis assessment."""

import cv2
import numpy as np
from typing import Union, Optional
from dataclasses import dataclass


@dataclass
class QuantificationMetrics:
    """Container for quantification evaluation metrics."""
    openness_score: float
    grade: int
    total_area: int
    open_area: int
    threshold_pixel_value: float
    
    def __str__(self) -> str:
        return (f"QuantificationMetrics(openness={self.openness_score:.4f}, "
                f"grade={self.grade}, total_area={self.total_area}, "
                f"open_area={self.open_area})")


def openness_score(mask: np.ndarray, preprocessed_image: np.ndarray, 
                  threshold_ratio: float = 0.85, verbose: bool = False) -> float:
    """
    Calculate openness score for a glomerulus.
    
    The openness score measures the ratio of open capillaries to total glomerular area.
    Open capillaries are identified by high-intensity pixels in the preprocessed image.
    
    Args:
        mask: Binary mask of the glomerulus (white pixels indicate glomerulus)
        preprocessed_image: Preprocessed grayscale image
        threshold_ratio: Ratio of max pixel value to use as threshold (0-1)
        verbose: Whether to print debug information
        
    Returns:
        Openness score (0-1, higher indicates more open capillaries)
    """
    # Calculate the area of the glomerulus (white pixels in the mask)
    total_area = cv2.countNonZero(mask)
    if verbose:
        print(f'Total area: {total_area}')
    
    if total_area == 0:
        return 0.0
    
    # Find the maximum pixel value in the preprocessed image
    max_pixel_value = np.max(preprocessed_image)
    if verbose:
        print(f'Max pixel value: {max_pixel_value}')
    
    # Calculate the threshold pixel value
    threshold_pixel_value = threshold_ratio * max_pixel_value
    if verbose:
        print(f'Threshold pixel value: {threshold_pixel_value}')
    
    # Create a binary mask with maximum pixel values in the preprocessed image
    max_pixel_mask = (preprocessed_image >= threshold_pixel_value).astype(np.uint8)
    
    if verbose:
        print(f'Preprocessed image shape: {preprocessed_image.shape}')
        print(f'Mask shape: {mask.shape}')
    
    # Calculate the area of open capillaries (maximum pixel value occurrences within the mask)
    open_area = cv2.countNonZero(cv2.bitwise_and(max_pixel_mask, mask))
    if verbose:
        print(f'Open area: {open_area}')
    
    # Calculate the ratio of open area to total area
    score = open_area / total_area
    
    return score


def grade_glomerulus(openness_score: float, 
                    grade_thresholds: Optional[list] = None) -> int:
    """
    Grade glomerulus based on openness score.
    
    Args:
        openness_score: Openness score (0-1)
        grade_thresholds: List of threshold values for each grade.
                         Default: [0.6, 0.4, 0.2] representing 60%, 40%, 20% open
        
    Returns:
        Grade (0 = most open, higher numbers = less open)
    """
    if grade_thresholds is None:
        # Default thresholds based on openness percentage
        grade_thresholds = [0.6, 0.4, 0.2]  # 60% open, 40% open, 20% open
    
    # Grade the glomerulus based on the openness score
    for i, threshold in enumerate(grade_thresholds):
        if openness_score >= threshold:
            return i
    
    # If openness score is below all thresholds
    return len(grade_thresholds)


def calculate_quantification_metrics(mask: np.ndarray, preprocessed_image: np.ndarray,
                                   threshold_ratio: float = 0.85,
                                   grade_thresholds: Optional[list] = None,
                                   verbose: bool = False) -> QuantificationMetrics:
    """
    Calculate comprehensive quantification metrics for a glomerulus.
    
    Args:
        mask: Binary mask of the glomerulus
        preprocessed_image: Preprocessed grayscale image
        threshold_ratio: Ratio of max pixel value to use as threshold
        grade_thresholds: Custom grade thresholds
        verbose: Whether to print debug information
        
    Returns:
        QuantificationMetrics object with all calculated metrics
    """
    # Calculate basic areas
    total_area = cv2.countNonZero(mask)
    
    if total_area == 0:
        return QuantificationMetrics(
            openness_score=0.0,
            grade=len(grade_thresholds or [0.6, 0.4, 0.2]),
            total_area=0,
            open_area=0,
            threshold_pixel_value=0.0
        )
    
    # Calculate openness score
    max_pixel_value = np.max(preprocessed_image)
    threshold_pixel_value = threshold_ratio * max_pixel_value
    max_pixel_mask = (preprocessed_image >= threshold_pixel_value).astype(np.uint8)
    open_area = cv2.countNonZero(cv2.bitwise_and(max_pixel_mask, mask))
    
    openness = open_area / total_area
    grade = grade_glomerulus(openness, grade_thresholds)
    
    if verbose:
        print(f'Total area: {total_area}')
        print(f'Open area: {open_area}')
        print(f'Openness score: {openness:.4f}')
        print(f'Grade: {grade}')
    
    return QuantificationMetrics(
        openness_score=openness,
        grade=grade,
        total_area=total_area,
        open_area=open_area,
        threshold_pixel_value=threshold_pixel_value
    )


def batch_quantification_metrics(masks: list, preprocessed_images: list,
                                threshold_ratio: float = 0.85,
                                grade_thresholds: Optional[list] = None,
                                verbose: bool = False) -> list:
    """
    Calculate quantification metrics for a batch of glomeruli.
    
    Args:
        masks: List of binary masks
        preprocessed_images: List of preprocessed grayscale images
        threshold_ratio: Ratio of max pixel value to use as threshold
        grade_thresholds: Custom grade thresholds
        verbose: Whether to print debug information
        
    Returns:
        List of QuantificationMetrics objects
    """
    if len(masks) != len(preprocessed_images):
        raise ValueError("Number of masks must match number of preprocessed images")
    
    results = []
    for i, (mask, image) in enumerate(zip(masks, preprocessed_images)):
        if verbose:
            print(f"\nProcessing glomerulus {i+1}/{len(masks)}")
        
        metrics = calculate_quantification_metrics(
            mask, image, threshold_ratio, grade_thresholds, verbose
        )
        results.append(metrics)
    
    return results


def endotheliosis_severity_classification(openness_score: float) -> str:
    """
    Classify endotheliosis severity based on openness score.
    
    Args:
        openness_score: Openness score (0-1)
        
    Returns:
        Severity classification string
    """
    if openness_score >= 0.7:
        return "Normal"
    elif openness_score >= 0.5:
        return "Mild Endotheliosis"
    elif openness_score >= 0.3:
        return "Moderate Endotheliosis"
    elif openness_score >= 0.1:
        return "Severe Endotheliosis"
    else:
        return "Critical Endotheliosis"


def calculate_summary_statistics(metrics_list: list) -> dict:
    """
    Calculate summary statistics for a batch of quantification metrics.
    
    Args:
        metrics_list: List of QuantificationMetrics objects
        
    Returns:
        Dictionary with summary statistics
    """
    if not metrics_list:
        return {}
    
    openness_scores = [m.openness_score for m in metrics_list]
    grades = [m.grade for m in metrics_list]
    total_areas = [m.total_area for m in metrics_list]
    open_areas = [m.open_area for m in metrics_list]
    
    return {
        "count": len(metrics_list),
        "openness_score": {
            "mean": np.mean(openness_scores),
            "std": np.std(openness_scores),
            "min": np.min(openness_scores),
            "max": np.max(openness_scores),
            "median": np.median(openness_scores)
        },
        "grade": {
            "mean": np.mean(grades),
            "distribution": {i: grades.count(i) for i in set(grades)}
        },
        "total_area": {
            "mean": np.mean(total_areas),
            "std": np.std(total_areas),
            "total": np.sum(total_areas)
        },
        "open_area": {
            "mean": np.mean(open_areas),
            "std": np.std(open_areas),
            "total": np.sum(open_areas)
        }
    }
