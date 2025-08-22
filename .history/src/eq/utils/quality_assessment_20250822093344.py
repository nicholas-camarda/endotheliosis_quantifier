#!/usr/bin/env python3
"""Quality assessment metrics and visualization system for the endotheliosis quantifier."""

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from eq.utils.logger import get_logger


@dataclass
class SegmentationQualityMetrics:
    """Quality metrics for segmentation results."""
    dice_coefficient: float
    iou_score: float
    precision: float
    recall: float
    f1_score: float
    hausdorff_distance: Optional[float] = None
    boundary_accuracy: Optional[float] = None
    confidence_score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class ROIQualityMetrics:
    """Quality metrics for ROI extraction."""
    roi_count: int
    avg_roi_size: float
    roi_size_std: float
    roi_quality_scores: List[float]
    extraction_success_rate: float
    overlap_ratio: Optional[float] = None
    boundary_smoothness: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        return data


@dataclass
class FeatureQualityMetrics:
    """Quality metrics for feature extraction."""
    feature_count: int
    feature_completeness: float
    feature_correlation_matrix: Optional[np.ndarray] = None
    feature_importance_scores: Optional[Dict[str, float]] = None
    outlier_percentage: Optional[float] = None
    data_quality_score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        if self.feature_correlation_matrix is not None:
            data['feature_correlation_matrix'] = self.feature_correlation_matrix.tolist()
        return data


class QualityAssessor:
    """Assesses quality of pipeline outputs."""
    
    def __init__(self, output_dir: Path):
        """
        Initialize the quality assessor.
        
        Args:
            output_dir: Directory to save quality assessment results
        """
        self.output_dir = Path(output_dir)
        self.logger = get_logger("eq.quality_assessor")
        
        # Create quality assessment directory
        self.quality_dir = self.output_dir / "quality_assessment"
        self.quality_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Quality assessor initialized for output directory: {output_dir}")
    
    def assess_segmentation_quality(self, predicted_masks: List[np.ndarray], 
                                  ground_truth_masks: List[np.ndarray],
                                  confidence_scores: Optional[List[float]] = None) -> SegmentationQualityMetrics:
        """Assess the quality of segmentation results."""
        if len(predicted_masks) != len(ground_truth_masks):
            raise ValueError("Number of predicted masks must match number of ground truth masks")
        
        dice_scores = []
        iou_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        for pred_mask, gt_mask in zip(predicted_masks, ground_truth_masks):
            # Ensure masks are binary
            pred_binary = (pred_mask > 0.5).astype(np.uint8)
            gt_binary = (gt_mask > 0.5).astype(np.uint8)
            
            # Calculate metrics
            intersection = np.logical_and(pred_binary, gt_binary).sum()
            union = np.logical_or(pred_binary, gt_binary).sum()
            
            # Dice coefficient
            dice = (2 * intersection) / (pred_binary.sum() + gt_binary.sum() + 1e-8)
            dice_scores.append(dice)
            
            # IoU (Jaccard index)
            iou = intersection / (union + 1e-8)
            iou_scores.append(iou)
            
            # Precision, Recall, F1
            tp = intersection
            fp = pred_binary.sum() - intersection
            fn = gt_binary.sum() - intersection
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
            
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
        
        # Calculate average metrics
        avg_dice = np.mean(dice_scores)
        avg_iou = np.mean(iou_scores)
        avg_precision = np.mean(precision_scores)
        avg_recall = np.mean(recall_scores)
        avg_f1 = np.mean(f1_scores)
        
        # Calculate confidence score if available
        confidence_score = None
        if confidence_scores:
            confidence_score = np.mean(confidence_scores)
        
        metrics = SegmentationQualityMetrics(
            dice_coefficient=avg_dice,
            iou_score=avg_iou,
            precision=avg_precision,
            recall=avg_recall,
            f1_score=avg_f1,
            confidence_score=confidence_score
        )
        
        self.logger.info(f"Segmentation quality assessment completed - Dice: {avg_dice:.4f}, IoU: {avg_iou:.4f}")
        return metrics
    
    def assess_roi_quality(self, extracted_rois: List[np.ndarray],
                          roi_sizes: List[Tuple[int, int]],
                          quality_scores: Optional[List[float]] = None) -> ROIQualityMetrics:
        """Assess the quality of ROI extraction."""
        roi_count = len(extracted_rois)
        
        if roi_count == 0:
            return ROIQualityMetrics(
                roi_count=0,
                avg_roi_size=0.0,
                roi_size_std=0.0,
                roi_quality_scores=[],
                extraction_success_rate=0.0
            )
        
        # Calculate ROI size statistics
        roi_areas = [width * height for width, height in roi_sizes]
        avg_roi_size = np.mean(roi_areas)
        roi_size_std = np.std(roi_areas)
        
        # Use provided quality scores or calculate basic ones
        if quality_scores is None:
            # Simple quality score based on ROI size (assuming reasonable size range)
            quality_scores = []
            for area in roi_areas:
                # Normalize area to 0-1 range (assuming reasonable ROI sizes)
                normalized_area = min(area / 10000, 1.0)  # 10000 pixels as max
                quality_scores.append(normalized_area)
        
        extraction_success_rate = roi_count / max(roi_count, 1)  # Should be 1.0 if all ROIs extracted
        
        metrics = ROIQualityMetrics(
            roi_count=roi_count,
            avg_roi_size=avg_roi_size,
            roi_size_std=roi_size_std,
            roi_quality_scores=quality_scores,
            extraction_success_rate=extraction_success_rate
        )
        
        self.logger.info(f"ROI quality assessment completed - Count: {roi_count}, Avg Size: {avg_roi_size:.2f}")
        return metrics
    
    def assess_feature_quality(self, features: np.ndarray,
                             feature_names: List[str],
                             target_values: Optional[np.ndarray] = None) -> FeatureQualityMetrics:
        """Assess the quality of extracted features."""
        feature_count = features.shape[1] if len(features.shape) > 1 else 1
        
        # Calculate feature completeness (percentage of non-null values)
        feature_completeness = 1.0 - np.isnan(features).sum() / features.size
        
        # Calculate correlation matrix
        feature_correlation_matrix = None
        if feature_count > 1:
            try:
                # Remove any rows with NaN values for correlation calculation
                clean_features = features[~np.isnan(features).any(axis=1)]
                if len(clean_features) > 1:
                    feature_correlation_matrix = np.corrcoef(clean_features.T)
            except Exception as e:
                self.logger.warning(f"Could not calculate feature correlation matrix: {e}")
        
        # Calculate feature importance if target values are provided
        feature_importance_scores = None
        if target_values is not None and feature_count > 1:
            try:
                from sklearn.feature_selection import mutual_info_regression
                clean_features = features[~np.isnan(features).any(axis=1)]
                clean_targets = target_values[~np.isnan(features).any(axis=1)]
                
                if len(clean_features) > 1:
                    importance_scores = mutual_info_regression(clean_features, clean_targets)
                    feature_importance_scores = dict(zip(feature_names, importance_scores))
            except ImportError:
                self.logger.warning("sklearn not available for feature importance calculation")
            except Exception as e:
                self.logger.warning(f"Could not calculate feature importance: {e}")
        
        # Calculate outlier percentage
        outlier_percentage = None
        try:
            # Simple outlier detection using IQR method
            outliers = 0
            total_values = 0
            
            for i in range(feature_count):
                feature_values = features[:, i]
                clean_values = feature_values[~np.isnan(feature_values)]
                
                if len(clean_values) > 0:
                    q1 = np.percentile(clean_values, 25)
                    q3 = np.percentile(clean_values, 75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    outliers += np.sum((clean_values < lower_bound) | (clean_values > upper_bound))
                    total_values += len(clean_values)
            
            if total_values > 0:
                outlier_percentage = outliers / total_values
        except Exception as e:
            self.logger.warning(f"Could not calculate outlier percentage: {e}")
        
        metrics = FeatureQualityMetrics(
            feature_count=feature_count,
            feature_completeness=feature_completeness,
            feature_correlation_matrix=feature_correlation_matrix,
            feature_importance_scores=feature_importance_scores,
            outlier_percentage=outlier_percentage
        )
        
        self.logger.info(f"Feature quality assessment completed - Count: {feature_count}, Completeness: {feature_completeness:.4f}")
        return metrics


class QualityVisualizer:
    """Creates visualizations of quality assessment results."""
    
    def __init__(self, output_dir: Path):
        """
        Initialize the quality visualizer.
        
        Args:
            output_dir: Directory to save quality plots
        """
        self.output_dir = Path(output_dir)
        self.logger = get_logger("eq.quality_visualizer")
        
        # Create quality plots directory
        self.quality_plots_dir = self.output_dir / "quality_plots"
        self.quality_plots_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Quality visualizer initialized for output directory: {output_dir}")
    
    def create_segmentation_quality_plots(self, metrics: SegmentationQualityMetrics,
                                        individual_scores: Optional[Dict[str, List[float]]] = None) -> List[Path]:
        """Create visualizations for segmentation quality metrics."""
        plot_paths = []
        
        # Create overall metrics plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metric_names = ['Dice', 'IoU', 'Precision', 'Recall', 'F1']
        metric_values = [
            metrics.dice_coefficient,
            metrics.iou_score,
            metrics.precision,
            metrics.recall,
            metrics.f1_score
        ]
        
        bars = ax.bar(metric_names, metric_values, color=['blue', 'green', 'orange', 'red', 'purple'])
        ax.set_ylabel('Score')
        ax.set_title('Segmentation Quality Metrics')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        overall_plot = self.quality_plots_dir / "segmentation_quality_overall.png"
        plt.savefig(overall_plot, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths.append(overall_plot)
        
        # Create individual scores plot if available
        if individual_scores:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
            
            for i, (metric_name, scores) in enumerate(individual_scores.items()):
                if i < 4:  # Limit to 4 subplots
                    ax = axes[i]
                    ax.hist(scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                    ax.set_xlabel(f'{metric_name} Score')
                    ax.set_ylabel('Frequency')
                    ax.set_title(f'{metric_name} Distribution')
                    ax.axvline(np.mean(scores), color='red', linestyle='--', label=f'Mean: {np.mean(scores):.3f}')
                    ax.legend()
            
            plt.tight_layout()
            distribution_plot = self.quality_plots_dir / "segmentation_quality_distributions.png"
            plt.savefig(distribution_plot, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths.append(distribution_plot)
        
        return plot_paths
    
    def create_roi_quality_plots(self, metrics: ROIQualityMetrics) -> List[Path]:
        """Create visualizations for ROI quality metrics."""
        plot_paths = []
        
        # Create ROI size distribution plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # ROI size histogram
        if metrics.roi_quality_scores:
            ax1.hist(metrics.roi_quality_scores, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
            ax1.set_xlabel('ROI Quality Score')
            ax1.set_ylabel('Frequency')
            ax1.set_title('ROI Quality Score Distribution')
            ax1.axvline(np.mean(metrics.roi_quality_scores), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(metrics.roi_quality_scores):.3f}')
            ax1.legend()
        
        # ROI count and success rate
        categories = ['ROI Count', 'Success Rate']
        values = [metrics.roi_count, metrics.extraction_success_rate]
        colors = ['blue', 'green']
        
        bars = ax2.bar(categories, values, color=colors)
        ax2.set_ylabel('Value')
        ax2.set_title('ROI Extraction Summary')
        
        # Add value labels
        for bar, value in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        roi_plot = self.quality_plots_dir / "roi_quality_metrics.png"
        plt.savefig(roi_plot, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths.append(roi_plot)
        
        return plot_paths
    
    def create_feature_quality_plots(self, metrics: FeatureQualityMetrics) -> List[Path]:
        """Create visualizations for feature quality metrics."""
        plot_paths = []
        
        # Create feature correlation heatmap if available
        if metrics.feature_correlation_matrix is not None:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            im = ax.imshow(metrics.feature_correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            ax.set_title('Feature Correlation Matrix')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Correlation Coefficient')
            
            plt.tight_layout()
            correlation_plot = self.quality_plots_dir / "feature_correlation_matrix.png"
            plt.savefig(correlation_plot, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths.append(correlation_plot)
        
        # Create feature importance plot if available
        if metrics.feature_importance_scores:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Sort features by importance
            sorted_features = sorted(metrics.feature_importance_scores.items(), 
                                   key=lambda x: x[1], reverse=True)
            
            feature_names, importance_scores = zip(*sorted_features[:20])  # Top 20 features
            
            bars = ax.barh(range(len(feature_names)), importance_scores, color='orange')
            ax.set_yticks(range(len(feature_names)))
            ax.set_yticklabels(feature_names)
            ax.set_xlabel('Importance Score')
            ax.set_title('Feature Importance (Top 20)')
            
            plt.tight_layout()
            importance_plot = self.quality_plots_dir / "feature_importance.png"
            plt.savefig(importance_plot, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths.append(importance_plot)
        
        # Create quality summary plot
        fig, ax = plt.subplots(figsize=(8, 6))
        
        quality_metrics = ['Completeness']
        quality_values = [metrics.feature_completeness]
        
        if metrics.outlier_percentage is not None:
            quality_metrics.append('Outlier %')
            quality_values.append(metrics.outlier_percentage)
        
        bars = ax.bar(quality_metrics, quality_values, color=['green', 'red'])
        ax.set_ylabel('Score')
        ax.set_title('Feature Quality Summary')
        ax.set_ylim(0, 1)
        
        # Add value labels
        for bar, value in zip(bars, quality_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        summary_plot = self.quality_plots_dir / "feature_quality_summary.png"
        plt.savefig(summary_plot, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths.append(summary_plot)
        
        return plot_paths
    
    def generate_quality_report(self, segmentation_metrics: Optional[SegmentationQualityMetrics] = None,
                              roi_metrics: Optional[ROIQualityMetrics] = None,
                              feature_metrics: Optional[FeatureQualityMetrics] = None) -> str:
        """Generate a comprehensive quality assessment report."""
        report_lines = [
            "# Quality Assessment Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]
        
        if segmentation_metrics:
            report_lines.extend([
                "## Segmentation Quality",
                f"- **Dice Coefficient**: {segmentation_metrics.dice_coefficient:.4f}",
                f"- **IoU Score**: {segmentation_metrics.iou_score:.4f}",
                f"- **Precision**: {segmentation_metrics.precision:.4f}",
                f"- **Recall**: {segmentation_metrics.recall:.4f}",
                f"- **F1 Score**: {segmentation_metrics.f1_score:.4f}",
            ])
            
            if segmentation_metrics.confidence_score is not None:
                report_lines.append(f"- **Average Confidence**: {segmentation_metrics.confidence_score:.4f}")
            
            report_lines.append("")
        
        if roi_metrics:
            report_lines.extend([
                "## ROI Extraction Quality",
                f"- **ROI Count**: {roi_metrics.roi_count}",
                f"- **Average ROI Size**: {roi_metrics.avg_roi_size:.2f} pixels",
                f"- **ROI Size Standard Deviation**: {roi_metrics.roi_size_std:.2f}",
                f"- **Extraction Success Rate**: {roi_metrics.extraction_success_rate:.2%}",
            ])
            
            if roi_metrics.roi_quality_scores:
                avg_quality = np.mean(roi_metrics.roi_quality_scores)
                report_lines.append(f"- **Average ROI Quality Score**: {avg_quality:.4f}")
            
            report_lines.append("")
        
        if feature_metrics:
            report_lines.extend([
                "## Feature Extraction Quality",
                f"- **Feature Count**: {feature_metrics.feature_count}",
                f"- **Feature Completeness**: {feature_metrics.feature_completeness:.2%}",
            ])
            
            if feature_metrics.outlier_percentage is not None:
                report_lines.append(f"- **Outlier Percentage**: {feature_metrics.outlier_percentage:.2%}")
            
            if feature_metrics.feature_importance_scores:
                report_lines.append("### Top Feature Importance:")
                sorted_features = sorted(feature_metrics.feature_importance_scores.items(),
                                       key=lambda x: x[1], reverse=True)
                for i, (feature, score) in enumerate(sorted_features[:10], 1):
                    report_lines.append(f"{i}. **{feature}**: {score:.4f}")
            
            report_lines.append("")
        
        return "\n".join(report_lines)
