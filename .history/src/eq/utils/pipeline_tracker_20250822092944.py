#!/usr/bin/env python3
"""Pipeline progression tracking system for the endotheliosis quantifier."""

import json
import logging
import time
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any

from eq.utils.logger import get_logger


class PipelineStage(Enum):
    """Enumeration of pipeline stages."""
    SEGMENTATION_TRAINING = "segmentation_training"
    QUANTIFICATION_TRAINING = "quantification_training"
    PRODUCTION_INFERENCE = "production_inference"


class StageStatus(Enum):
    """Enumeration of stage statuses."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StageMetrics:
    """Metrics for a pipeline stage."""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    images_processed: int = 0
    models_trained: int = 0
    accuracy: Optional[float] = None
    loss: Optional[float] = None
    validation_score: Optional[float] = None
    memory_used_mb: Optional[float] = None
    gpu_utilization: Optional[float] = None
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        if self.start_time:
            data['start_time'] = self.start_time.isoformat()
        if self.end_time:
            data['end_time'] = self.end_time.isoformat()
        return data


@dataclass
class PipelineStageInfo:
    """Information about a pipeline stage."""
    stage: PipelineStage
    status: StageStatus
    metrics: StageMetrics
    config: Dict[str, Any]
    output_paths: List[str] = None
    
    def __post_init__(self):
        if self.output_paths is None:
            self.output_paths = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'stage': self.stage.value,
            'status': self.status.value,
            'metrics': self.metrics.to_dict(),
            'config': self.config,
            'output_paths': self.output_paths
        }


class PipelineTracker:
    """Tracks progression through the endotheliosis quantifier pipeline."""
    
    def __init__(self, output_dir: Path):
        """
        Initialize the pipeline tracker.
        
        Args:
            output_dir: Directory to save tracking information
        """
        self.output_dir = Path(output_dir)
        self.logger = get_logger("eq.pipeline_tracker")
        
        # Create tracking directory
        self.tracking_dir = self.output_dir / "tracking"
        self.tracking_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize pipeline state
        self.pipeline_start_time = datetime.now()
        self.stages: Dict[PipelineStage, PipelineStageInfo] = {}
        self.overall_status = StageStatus.NOT_STARTED
        
        # Initialize all stages
        for stage in PipelineStage:
            self.stages[stage] = PipelineStageInfo(
                stage=stage,
                status=StageStatus.NOT_STARTED,
                metrics=StageMetrics(),
                config={}
            )
        
        self.logger.info(f"Pipeline tracker initialized for output directory: {output_dir}")
    
    def start_pipeline(self, config: Dict[str, Any]) -> None:
        """Start the pipeline tracking."""
        self.overall_status = StageStatus.IN_PROGRESS
        self.pipeline_start_time = datetime.now()
        
        # Save initial configuration
        config_file = self.tracking_dir / "pipeline_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        self.logger.info("Pipeline tracking started")
        self._save_tracking_state()
    
    def start_stage(self, stage: PipelineStage, config: Dict[str, Any]) -> None:
        """Start tracking a specific pipeline stage."""
        if stage not in self.stages:
            raise ValueError(f"Unknown pipeline stage: {stage}")
        
        stage_info = self.stages[stage]
        stage_info.status = StageStatus.IN_PROGRESS
        stage_info.metrics.start_time = datetime.now()
        stage_info.config = config
        
        self.logger.info(f"Started tracking stage: {stage.value}")
        self._save_tracking_state()
    
    def update_stage_metrics(self, stage: PipelineStage, **metrics) -> None:
        """Update metrics for a specific stage."""
        if stage not in self.stages:
            raise ValueError(f"Unknown pipeline stage: {stage}")
        
        stage_info = self.stages[stage]
        metrics_obj = stage_info.metrics
        
        # Update metrics
        for key, value in metrics.items():
            if hasattr(metrics_obj, key):
                setattr(metrics_obj, key, value)
            else:
                self.logger.warning(f"Unknown metric key: {key}")
        
        self._save_tracking_state()
    
    def complete_stage(self, stage: PipelineStage, output_paths: List[str] = None) -> None:
        """Mark a stage as completed."""
        if stage not in self.stages:
            raise ValueError(f"Unknown pipeline stage: {stage}")
        
        stage_info = self.stages[stage]
        stage_info.status = StageStatus.COMPLETED
        stage_info.metrics.end_time = datetime.now()
        
        if stage_info.metrics.start_time:
            duration = (stage_info.metrics.end_time - stage_info.metrics.start_time).total_seconds()
            stage_info.metrics.duration_seconds = duration
        
        if output_paths:
            stage_info.output_paths = output_paths
        
        self.logger.info(f"Completed stage: {stage.value} (duration: {stage_info.metrics.duration_seconds:.2f}s)")
        self._save_tracking_state()
    
    def fail_stage(self, stage: PipelineStage, error: str) -> None:
        """Mark a stage as failed."""
        if stage not in self.stages:
            raise ValueError(f"Unknown pipeline stage: {stage}")
        
        stage_info = self.stages[stage]
        stage_info.status = StageStatus.FAILED
        stage_info.metrics.end_time = datetime.now()
        stage_info.metrics.errors.append(error)
        
        if stage_info.metrics.start_time:
            duration = (stage_info.metrics.end_time - stage_info.metrics.start_time).total_seconds()
            stage_info.metrics.duration_seconds = duration
        
        self.logger.error(f"Failed stage: {stage.value} - {error}")
        self._save_tracking_state()
    
    def skip_stage(self, stage: PipelineStage, reason: str) -> None:
        """Mark a stage as skipped."""
        if stage not in self.stages:
            raise ValueError(f"Unknown pipeline stage: {stage}")
        
        stage_info = self.stages[stage]
        stage_info.status = StageStatus.SKIPPED
        stage_info.metrics.errors.append(f"Skipped: {reason}")
        
        self.logger.info(f"Skipped stage: {stage.value} - {reason}")
        self._save_tracking_state()
    
    def complete_pipeline(self) -> None:
        """Mark the pipeline as completed."""
        self.overall_status = StageStatus.COMPLETED
        self.logger.info("Pipeline tracking completed")
        self._save_tracking_state()
    
    def fail_pipeline(self, error: str) -> None:
        """Mark the pipeline as failed."""
        self.overall_status = StageStatus.FAILED
        self.logger.error(f"Pipeline failed: {error}")
        self._save_tracking_state()
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get a summary of the pipeline execution."""
        total_duration = None
        if self.overall_status in [StageStatus.COMPLETED, StageStatus.FAILED]:
            total_duration = (datetime.now() - self.pipeline_start_time).total_seconds()
        
        completed_stages = sum(1 for stage in self.stages.values() if stage.status == StageStatus.COMPLETED)
        failed_stages = sum(1 for stage in self.stages.values() if stage.status == StageStatus.FAILED)
        skipped_stages = sum(1 for stage in self.stages.values() if stage.status == StageStatus.SKIPPED)
        
        return {
            'overall_status': self.overall_status.value,
            'pipeline_start_time': self.pipeline_start_time.isoformat(),
            'total_duration_seconds': total_duration,
            'stages_summary': {
                'total': len(self.stages),
                'completed': completed_stages,
                'failed': failed_stages,
                'skipped': skipped_stages,
                'in_progress': len(self.stages) - completed_stages - failed_stages - skipped_stages
            },
            'stages': {stage.value: stage_info.to_dict() for stage, stage_info in self.stages.items()}
        }
    
    def generate_progression_report(self) -> str:
        """Generate a human-readable progression report."""
        summary = self.get_pipeline_summary()
        
        report_lines = [
            "# Pipeline Progression Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"## Overall Status: {summary['overall_status'].upper()}",
            f"Pipeline Start: {summary['pipeline_start_time']}",
        ]
        
        if summary['total_duration_seconds']:
            report_lines.append(f"Total Duration: {summary['total_duration_seconds']:.2f} seconds")
        
        report_lines.extend([
            "",
            "## Stage Summary",
            f"- Total Stages: {summary['stages_summary']['total']}",
            f"- Completed: {summary['stages_summary']['completed']}",
            f"- Failed: {summary['stages_summary']['failed']}",
            f"- Skipped: {summary['stages_summary']['skipped']}",
            f"- In Progress: {summary['stages_summary']['in_progress']}",
            "",
            "## Stage Details"
        ])
        
        for stage_name, stage_data in summary['stages'].items():
            status_emoji = {
                'completed': '✅',
                'failed': '❌',
                'skipped': '⏭️',
                'in_progress': '🔄',
                'not_started': '⏳'
            }.get(stage_data['status'], '❓')
            
            report_lines.append(f"### {status_emoji} {stage_name.replace('_', ' ').title()}")
            report_lines.append(f"Status: {stage_data['status']}")
            
            metrics = stage_data['metrics']
            if metrics['duration_seconds']:
                report_lines.append(f"Duration: {metrics['duration_seconds']:.2f} seconds")
            
            if metrics['images_processed'] > 0:
                report_lines.append(f"Images Processed: {metrics['images_processed']}")
            
            if metrics['accuracy'] is not None:
                report_lines.append(f"Accuracy: {metrics['accuracy']:.4f}")
            
            if metrics['loss'] is not None:
                report_lines.append(f"Loss: {metrics['loss']:.4f}")
            
            if metrics['errors']:
                report_lines.append("Errors:")
                for error in metrics['errors']:
                    report_lines.append(f"  - {error}")
            
            if stage_data['output_paths']:
                report_lines.append("Output Files:")
                for path in stage_data['output_paths']:
                    report_lines.append(f"  - {path}")
            
            report_lines.append("")
        
        return "\n".join(report_lines)
    
    def _save_tracking_state(self) -> None:
        """Save the current tracking state to disk."""
        # Save detailed state
        state_file = self.tracking_dir / "pipeline_state.json"
        with open(state_file, 'w') as f:
            json.dump(self.get_pipeline_summary(), f, indent=2, default=str)
        
        # Save progression report
        report_file = self.tracking_dir / "progression_report.md"
        with open(report_file, 'w') as f:
            f.write(self.generate_progression_report())
        
        self.logger.debug("Pipeline tracking state saved")
