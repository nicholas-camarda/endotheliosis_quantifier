#!/usr/bin/env python3
"""Stage-specific reporting system for the endotheliosis quantifier pipeline."""

import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from eq.utils.logger import get_logger
from eq.utils.pipeline_tracker import PipelineStage, PipelineStageInfo


class StageReporter:
    """Generates detailed reports for each pipeline stage."""
    
    def __init__(self, output_dir: Path):
        """
        Initialize the stage reporter.
        
        Args:
            output_dir: Directory to save stage reports
        """
        self.output_dir = Path(output_dir)
        self.logger = get_logger("eq.stage_reporter")
        
        # Create reports directory
        self.reports_dir = self.output_dir / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Stage reporter initialized for output directory: {output_dir}")
    
    def generate_segmentation_report(self, stage_info: PipelineStageInfo, 
                                   training_history: Optional[Dict] = None,
                                   model_performance: Optional[Dict] = None) -> str:
        """Generate a detailed report for segmentation training stage."""
        report_lines = [
            "# Segmentation Training Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Stage Information",
            f"- **Stage**: {stage_info.stage.value}",
            f"- **Status**: {stage_info.status.value}",
            f"- **Start Time**: {stage_info.metrics.start_time}",
            f"- **End Time**: {stage_info.metrics.end_time}",
            f"- **Duration**: {stage_info.metrics.duration_seconds:.2f} seconds",
            "",
            "## Configuration",
        ]
        
        # Add configuration details
        for key, value in stage_info.config.items():
            report_lines.append(f"- **{key}**: {value}")
        
        report_lines.extend([
            "",
            "## Performance Metrics",
            f"- **Images Processed**: {stage_info.metrics.images_processed}",
            f"- **Models Trained**: {stage_info.metrics.models_trained}",
        ])
        
        if stage_info.metrics.accuracy is not None:
            report_lines.append(f"- **Accuracy**: {stage_info.metrics.accuracy:.4f}")
        
        if stage_info.metrics.loss is not None:
            report_lines.append(f"- **Loss**: {stage_info.metrics.loss:.4f}")
        
        if stage_info.metrics.validation_score is not None:
            report_lines.append(f"- **Validation Score**: {stage_info.metrics.validation_score:.4f}")
        
        if stage_info.metrics.memory_used_mb is not None:
            report_lines.append(f"- **Memory Used**: {stage_info.metrics.memory_used_mb:.2f} MB")
        
        if stage_info.metrics.gpu_utilization is not None:
            report_lines.append(f"- **GPU Utilization**: {stage_info.metrics.gpu_utilization:.2f}%")
        
        # Add training history if available
        if training_history:
            report_lines.extend([
                "",
                "## Training History",
                "### Loss Progression",
            ])
            
            if 'train_loss' in training_history:
                losses = training_history['train_loss']
                report_lines.append(f"- **Final Loss**: {losses[-1]:.4f}")
                report_lines.append(f"- **Loss Improvement**: {losses[0] - losses[-1]:.4f}")
                report_lines.append(f"- **Training Epochs**: {len(losses)}")
            
            if 'val_loss' in training_history:
                val_losses = training_history['val_loss']
                report_lines.append(f"- **Final Validation Loss**: {val_losses[-1]:.4f}")
                report_lines.append(f"- **Validation Loss Improvement**: {val_losses[0] - val_losses[-1]:.4f}")
        
        # Add model performance details
        if model_performance:
            report_lines.extend([
                "",
                "## Model Performance",
            ])
            
            for metric, value in model_performance.items():
                if isinstance(value, float):
                    report_lines.append(f"- **{metric}**: {value:.4f}")
                else:
                    report_lines.append(f"- **{metric}**: {value}")
        
        # Add output files
        if stage_info.output_paths:
            report_lines.extend([
                "",
                "## Output Files",
            ])
            
            for path in stage_info.output_paths:
                report_lines.append(f"- {path}")
        
        # Add errors if any
        if stage_info.metrics.errors:
            report_lines.extend([
                "",
                "## Errors and Warnings",
            ])
            
            for error in stage_info.metrics.errors:
                report_lines.append(f"- {error}")
        
        return "\n".join(report_lines)
    
    def generate_quantification_report(self, stage_info: PipelineStageInfo,
                                     roi_stats: Optional[Dict] = None,
                                     feature_stats: Optional[Dict] = None,
                                     regression_performance: Optional[Dict] = None) -> str:
        """Generate a detailed report for quantification training stage."""
        report_lines = [
            "# Quantification Training Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Stage Information",
            f"- **Stage**: {stage_info.stage.value}",
            f"- **Status**: {stage_info.status.value}",
            f"- **Start Time**: {stage_info.metrics.start_time}",
            f"- **End Time**: {stage_info.metrics.end_time}",
            f"- **Duration**: {stage_info.metrics.duration_seconds:.2f} seconds",
            "",
            "## Configuration",
        ]
        
        # Add configuration details
        for key, value in stage_info.config.items():
            report_lines.append(f"- **{key}**: {value}")
        
        report_lines.extend([
            "",
            "## Performance Metrics",
            f"- **Images Processed**: {stage_info.metrics.images_processed}",
            f"- **Models Trained**: {stage_info.metrics.models_trained}",
        ])
        
        if stage_info.metrics.accuracy is not None:
            report_lines.append(f"- **Accuracy**: {stage_info.metrics.accuracy:.4f}")
        
        if stage_info.metrics.loss is not None:
            report_lines.append(f"- **Loss**: {stage_info.metrics.loss:.4f}")
        
        if stage_info.metrics.validation_score is not None:
            report_lines.append(f"- **Validation Score**: {stage_info.metrics.validation_score:.4f}")
        
        # Add ROI statistics
        if roi_stats:
            report_lines.extend([
                "",
                "## ROI Extraction Statistics",
                f"- **Total ROIs Extracted**: {roi_stats.get('total_rois', 0)}",
                f"- **Average ROI Size**: {roi_stats.get('avg_roi_size', 0):.2f} pixels",
                f"- **ROI Extraction Success Rate**: {roi_stats.get('success_rate', 0):.2%}",
            ])
            
            if 'roi_quality_scores' in roi_stats:
                scores = roi_stats['roi_quality_scores']
                report_lines.extend([
                    f"- **Average ROI Quality Score**: {np.mean(scores):.4f}",
                    f"- **ROI Quality Score Std**: {np.std(scores):.4f}",
                ])
        
        # Add feature statistics
        if feature_stats:
            report_lines.extend([
                "",
                "## Feature Extraction Statistics",
                f"- **Features Extracted**: {feature_stats.get('num_features', 0)}",
                f"- **Feature Extraction Success Rate**: {feature_stats.get('success_rate', 0):.2%}",
            ])
            
            if 'feature_importance' in feature_stats:
                report_lines.append("### Top Feature Importance:")
                importance = feature_stats['feature_importance']
                for i, (feature, score) in enumerate(importance[:10], 1):
                    report_lines.append(f"{i}. **{feature}**: {score:.4f}")
        
        # Add regression performance
        if regression_performance:
            report_lines.extend([
                "",
                "## Regression Model Performance",
            ])
            
            for metric, value in regression_performance.items():
                if isinstance(value, float):
                    report_lines.append(f"- **{metric}**: {value:.4f}")
                else:
                    report_lines.append(f"- **{metric}**: {value}")
        
        # Add output files
        if stage_info.output_paths:
            report_lines.extend([
                "",
                "## Output Files",
            ])
            
            for path in stage_info.output_paths:
                report_lines.append(f"- {path}")
        
        # Add errors if any
        if stage_info.metrics.errors:
            report_lines.extend([
                "",
                "## Errors and Warnings",
            ])
            
            for error in stage_info.metrics.errors:
                report_lines.append(f"- {error}")
        
        return "\n".join(report_lines)
    
    def generate_production_report(self, stage_info: PipelineStageInfo,
                                 inference_results: Optional[Dict] = None,
                                 quality_metrics: Optional[Dict] = None) -> str:
        """Generate a detailed report for production inference stage."""
        report_lines = [
            "# Production Inference Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Stage Information",
            f"- **Stage**: {stage_info.stage.value}",
            f"- **Status**: {stage_info.status.value}",
            f"- **Start Time**: {stage_info.metrics.start_time}",
            f"- **End Time**: {stage_info.metrics.end_time}",
            f"- **Duration**: {stage_info.metrics.duration_seconds:.2f} seconds",
            "",
            "## Configuration",
        ]
        
        # Add configuration details
        for key, value in stage_info.config.items():
            report_lines.append(f"- **{key}**: {value}")
        
        report_lines.extend([
            "",
            "## Performance Metrics",
            f"- **Images Processed**: {stage_info.metrics.images_processed}",
            f"- **Processing Rate**: {stage_info.metrics.images_processed / (stage_info.metrics.duration_seconds or 1):.2f} images/second",
        ])
        
        if stage_info.metrics.memory_used_mb is not None:
            report_lines.append(f"- **Memory Used**: {stage_info.metrics.memory_used_mb:.2f} MB")
        
        if stage_info.metrics.gpu_utilization is not None:
            report_lines.append(f"- **GPU Utilization**: {stage_info.metrics.gpu_utilization:.2f}%")
        
        # Add inference results
        if inference_results:
            report_lines.extend([
                "",
                "## Inference Results",
                f"- **Total Predictions**: {inference_results.get('total_predictions', 0)}",
                f"- **Average Confidence**: {inference_results.get('avg_confidence', 0):.4f}",
                f"- **Processing Time per Image**: {inference_results.get('avg_processing_time', 0):.2f} seconds",
            ])
            
            if 'prediction_distribution' in inference_results:
                report_lines.append("### Prediction Distribution:")
                dist = inference_results['prediction_distribution']
                for category, count in dist.items():
                    percentage = (count / inference_results['total_predictions']) * 100
                    report_lines.append(f"- **{category}**: {count} ({percentage:.1f}%)")
        
        # Add quality metrics
        if quality_metrics:
            report_lines.extend([
                "",
                "## Quality Assessment",
            ])
            
            for metric, value in quality_metrics.items():
                if isinstance(value, float):
                    report_lines.append(f"- **{metric}**: {value:.4f}")
                else:
                    report_lines.append(f"- **{metric}**: {value}")
        
        # Add output files
        if stage_info.output_paths:
            report_lines.extend([
                "",
                "## Output Files",
            ])
            
            for path in stage_info.output_paths:
                report_lines.append(f"- {path}")
        
        # Add errors if any
        if stage_info.metrics.errors:
            report_lines.extend([
                "",
                "## Errors and Warnings",
            ])
            
            for error in stage_info.metrics.errors:
                report_lines.append(f"- {error}")
        
        return "\n".join(report_lines)
    
    def save_stage_report(self, stage: PipelineStage, stage_info: PipelineStageInfo, 
                         report_content: str, additional_data: Optional[Dict] = None) -> Path:
        """Save a stage report to disk."""
        # Create stage-specific report file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"{stage.value}_{timestamp}.md"
        report_path = self.reports_dir / report_filename
        
        # Save the report
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        # Save additional data as JSON if provided
        if additional_data:
            json_filename = f"{stage.value}_{timestamp}_data.json"
            json_path = self.reports_dir / json_filename
            with open(json_path, 'w') as f:
                json.dump(additional_data, f, indent=2, default=str)
        
        self.logger.info(f"Stage report saved: {report_path}")
        return report_path
    
    def generate_stage_report(self, stage: PipelineStage, stage_info: PipelineStageInfo,
                            **kwargs) -> Path:
        """Generate and save a report for a specific stage."""
        if stage == PipelineStage.SEGMENTATION_TRAINING:
            report_content = self.generate_segmentation_report(stage_info, **kwargs)
        elif stage == PipelineStage.QUANTIFICATION_TRAINING:
            report_content = self.generate_quantification_report(stage_info, **kwargs)
        elif stage == PipelineStage.PRODUCTION_INFERENCE:
            report_content = self.generate_production_report(stage_info, **kwargs)
        else:
            raise ValueError(f"Unknown pipeline stage: {stage}")
        
        return self.save_stage_report(stage, stage_info, report_content, kwargs)
