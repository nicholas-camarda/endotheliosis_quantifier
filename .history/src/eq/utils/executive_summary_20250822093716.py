#!/usr/bin/env python3
"""Executive summary report generator for the endotheliosis quantifier pipeline."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from eq.utils.error_reporter import ErrorReporter
from eq.utils.logger import get_logger
from eq.utils.performance_metrics import PerformanceMonitor
from eq.utils.pipeline_tracker import PipelineTracker
from eq.utils.quality_assessment import QualityAssessor


class ExecutiveSummaryGenerator:
    """Generates executive summary reports for stakeholders."""
    
    def __init__(self, output_dir: Path):
        """
        Initialize the executive summary generator.
        
        Args:
            output_dir: Directory to save executive summaries
        """
        self.output_dir = Path(output_dir)
        self.logger = get_logger("eq.executive_summary")
        
        # Create executive summary directory
        self.executive_dir = self.output_dir / "executive_summary"
        self.executive_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Executive summary generator initialized for output directory: {output_dir}")
    
    def generate_executive_summary(self, pipeline_tracker: PipelineTracker,
                                 performance_monitor: Optional[PerformanceMonitor] = None,
                                 quality_assessor: Optional[QualityAssessor] = None,
                                 error_reporter: Optional[ErrorReporter] = None,
                                 additional_metrics: Optional[Dict[str, Any]] = None) -> str:
        """Generate a comprehensive executive summary report."""
        
        # Get pipeline summary
        pipeline_summary = pipeline_tracker.get_pipeline_summary()
        
        # Get performance summary if available
        performance_summary = None
        if performance_monitor:
            performance_summary = performance_monitor.get_performance_summary()
        
        # Get error summary if available
        error_summary = None
        if error_reporter:
            error_summary = error_reporter.get_error_summary()
        
        # Generate the report
        report_lines = [
            "# Executive Summary Report",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "**Pipeline**: Endotheliosis Quantifier",
            "",
            "## 🎯 Executive Overview",
        ]
        
        # Overall status
        status_emoji = {
            'completed': '✅',
            'failed': '❌',
            'in_progress': '🔄',
            'not_started': '⏳'
        }.get(pipeline_summary['overall_status'], '❓')
        
        report_lines.extend([
            f"**Overall Status**: {status_emoji} {pipeline_summary['overall_status'].upper()}",
            f"**Pipeline Duration**: {pipeline_summary.get('total_duration_seconds', 0) or 0:.1f} seconds",
            f"**Total Stages**: {pipeline_summary['stages_summary']['total']}",
            f"**Completed Stages**: {pipeline_summary['stages_summary']['completed']}",
            "",
        ])
        
        # Key achievements
        report_lines.extend([
            "## 🏆 Key Achievements",
        ])
        
        completed_stages = pipeline_summary['stages_summary']['completed']
        if completed_stages > 0:
            report_lines.append(f"- Successfully completed {completed_stages} pipeline stages")
        
        if pipeline_summary['overall_status'] == 'completed':
            report_lines.append("- Full pipeline execution completed successfully")
        
        if performance_summary and 'processing' in performance_summary:
            avg_rate = performance_summary['processing']['rate_mean']
            report_lines.append(f"- Achieved average processing rate of {avg_rate:.2f} images/second")
        
        if error_summary and error_summary['total_errors'] == 0:
            report_lines.append("- Zero errors encountered during execution")
        
        report_lines.append("")
        
        # Performance highlights
        if performance_summary:
            report_lines.extend([
                "## 📊 Performance Highlights",
            ])
            
            if 'cpu' in performance_summary:
                cpu_mean = performance_summary['cpu']['mean']
                cpu_max = performance_summary['cpu']['max']
                report_lines.append(f"- **CPU Utilization**: Average {cpu_mean:.1f}%, Peak {cpu_max:.1f}%")
            
            if 'memory' in performance_summary:
                memory_mean = performance_summary['memory']['mean_percent']
                memory_max = performance_summary['memory']['max_percent']
                report_lines.append(f"- **Memory Usage**: Average {memory_mean:.1f}%, Peak {memory_max:.1f}%")
            
            if 'gpu' in performance_summary:
                gpu_mean = performance_summary['gpu']['utilization_mean']
                gpu_max = performance_summary['gpu']['utilization_max']
                report_lines.append(f"- **GPU Utilization**: Average {gpu_mean:.1f}%, Peak {gpu_max:.1f}%")
            
            if 'processing' in performance_summary:
                processing_rate = performance_summary['processing']['rate_mean']
                report_lines.append(f"- **Processing Efficiency**: {processing_rate:.2f} images/second")
            
            report_lines.append("")
        
        # Quality metrics
        if quality_assessor and additional_metrics:
            report_lines.extend([
                "## 🔍 Quality Assessment",
            ])
            
            if 'segmentation_quality' in additional_metrics:
                seg_metrics = additional_metrics['segmentation_quality']
                if hasattr(seg_metrics, 'dice_coefficient'):
                    report_lines.append(f"- **Segmentation Accuracy**: {seg_metrics.dice_coefficient:.3f} (Dice Score)")
            
            if 'roi_quality' in additional_metrics:
                roi_metrics = additional_metrics['roi_quality']
                if hasattr(roi_metrics, 'roi_count'):
                    report_lines.append(f"- **ROIs Extracted**: {roi_metrics.roi_count} regions of interest")
            
            if 'feature_quality' in additional_metrics:
                feature_metrics = additional_metrics['feature_quality']
                if hasattr(feature_metrics, 'feature_completeness'):
                    completeness = feature_metrics.feature_completeness * 100
                    report_lines.append(f"- **Feature Completeness**: {completeness:.1f}%")
            
            report_lines.append("")
        
        # Issues and challenges
        report_lines.extend([
            "## ⚠️ Issues and Challenges",
        ])
        
        if error_summary and error_summary['total_errors'] > 0:
            report_lines.append(f"- **Total Errors**: {error_summary['total_errors']} errors encountered")
            report_lines.append(f"- **Resolution Rate**: {error_summary['resolution_rate']:.1%} errors resolved")
            
            if error_summary['most_common_category']:
                report_lines.append(f"- **Most Common Issue**: {error_summary['most_common_category']}")
            
            if error_summary['most_common_severity']:
                severity = error_summary['most_common_severity']
                if severity in ['high', 'critical']:
                    report_lines.append(f"- **Critical Issues**: {severity.upper()} severity errors detected")
        else:
            report_lines.append("- No significant issues encountered")
        
        failed_stages = pipeline_summary['stages_summary']['failed']
        if failed_stages > 0:
            report_lines.append(f"- **Failed Stages**: {failed_stages} pipeline stages failed")
        
        report_lines.append("")
        
        # Recommendations
        report_lines.extend([
            "## 💡 Recommendations",
        ])
        
        # Performance recommendations
        if performance_summary:
            if 'memory' in performance_summary and performance_summary['memory']['max_percent'] > 90:
                report_lines.append("- **Memory Optimization**: Consider reducing batch size or image resolution")
            
            if 'gpu' in performance_summary and performance_summary['gpu']['utilization_mean'] < 50:
                report_lines.append("- **GPU Utilization**: Consider increasing batch size for better GPU utilization")
        
        # Error recommendations
        if error_summary and error_summary['total_errors'] > 0:
            if error_summary['most_common_category'] == 'memory':
                report_lines.append("- **Memory Management**: Implement memory monitoring and optimization strategies")
            elif error_summary['most_common_category'] == 'data_loading':
                report_lines.append("- **Data Validation**: Implement robust data validation and preprocessing")
            elif error_summary['most_common_category'] == 'hardware':
                report_lines.append("- **Hardware Compatibility**: Verify hardware requirements and driver compatibility")
        
        # General recommendations
        if pipeline_summary['overall_status'] == 'completed':
            report_lines.append("- **Production Readiness**: Pipeline is ready for production deployment")
        elif pipeline_summary['overall_status'] == 'failed':
            report_lines.append("- **Debugging Required**: Address critical errors before production deployment")
        
        report_lines.append("")
        
        # Technical details (condensed)
        report_lines.extend([
            "## 🔧 Technical Details",
            f"- **Pipeline Start**: {pipeline_summary['pipeline_start_time']}",
            f"- **Execution Environment**: {self._get_environment_info()}",
        ])
        
        # Stage completion details
        for stage_name, stage_data in pipeline_summary['stages'].items():
            status = stage_data['status']
            status_emoji = {
                'completed': '✅',
                'failed': '❌',
                'in_progress': '🔄',
                'not_started': '⏳',
                'skipped': '⏭️'
            }.get(status, '❓')
            
            report_lines.append(f"- **{stage_name.replace('_', ' ').title()}**: {status_emoji} {status}")
            
            if status == 'completed' and stage_data['metrics']['duration_seconds']:
                duration = stage_data['metrics']['duration_seconds']
                report_lines.append(f"  - Duration: {duration:.1f} seconds")
            
            if status == 'completed' and stage_data['metrics']['images_processed']:
                images = stage_data['metrics']['images_processed']
                report_lines.append(f"  - Images Processed: {images}")
        
        report_lines.append("")
        
        # Next steps
        report_lines.extend([
            "## 🚀 Next Steps",
        ])
        
        if pipeline_summary['overall_status'] == 'completed':
            report_lines.extend([
                "- Deploy pipeline to production environment",
                "- Monitor performance and quality metrics",
                "- Scale processing capacity as needed",
                "- Implement automated quality checks"
            ])
        elif pipeline_summary['overall_status'] == 'failed':
            report_lines.extend([
                "- Review and address critical errors",
                "- Implement error handling improvements",
                "- Re-run pipeline with fixes",
                "- Validate pipeline stability"
            ])
        else:
            report_lines.extend([
                "- Complete remaining pipeline stages",
                "- Address any blocking issues",
                "- Validate pipeline outputs",
                "- Prepare for production deployment"
            ])
        
        report_lines.append("")
        
        # Contact information
        report_lines.extend([
            "## 📞 Contact Information",
            "- **Technical Support**: Development Team",
            "- **Report Generated**: Automated by Endotheliosis Quantifier Pipeline",
            "- **For Questions**: Review detailed logs and technical documentation",
            "",
            "---",
            f"*Report generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}*"
        ])
        
        return "\n".join(report_lines)
    
    def _get_environment_info(self) -> str:
        """Get basic environment information."""
        try:
            import platform

            import psutil
            
            system = platform.system()
            machine = platform.machine()
            processor = platform.processor()
            memory_gb = psutil.virtual_memory().total / (1024**3)
            
            return f"{system} {machine} ({processor}), {memory_gb:.1f}GB RAM"
        except Exception:
            return "Environment information unavailable"
    
    def save_executive_summary(self, summary_content: str, 
                             additional_data: Optional[Dict[str, Any]] = None) -> Path:
        """Save the executive summary to disk."""
        # Save main summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = self.executive_dir / f"executive_summary_{timestamp}.md"
        
        with open(summary_file, 'w') as f:
            f.write(summary_content)
        
        # Save additional data as JSON if provided
        if additional_data:
            data_file = self.executive_dir / f"executive_summary_data_{timestamp}.json"
            with open(data_file, 'w') as f:
                json.dump(additional_data, f, indent=2, default=str)
        
        # Create a latest version for easy access
        latest_file = self.executive_dir / "executive_summary_latest.md"
        with open(latest_file, 'w') as f:
            f.write(summary_content)
        
        self.logger.info(f"Executive summary saved: {summary_file}")
        return summary_file
    
    def generate_quick_summary(self, pipeline_tracker: PipelineTracker) -> str:
        """Generate a quick, one-page executive summary."""
        pipeline_summary = pipeline_tracker.get_pipeline_summary()
        
        # Determine overall status with emoji
        status_emoji = {
            'completed': '✅',
            'failed': '❌',
            'in_progress': '🔄',
            'not_started': '⏳'
        }.get(pipeline_summary['overall_status'], '❓')
        
        # Calculate completion percentage
        total_stages = pipeline_summary['stages_summary']['total']
        completed_stages = pipeline_summary['stages_summary']['completed']
        completion_percentage = (completed_stages / total_stages * 100) if total_stages > 0 else 0
        
        # Generate quick summary
        summary_lines = [
            "# Quick Executive Summary",
            f"**Date**: {datetime.now().strftime('%Y-%m-%d')}",
            f"**Time**: {datetime.now().strftime('%H:%M:%S')}",
            "",
            f"## Status: {status_emoji} {pipeline_summary['overall_status'].upper()}",
            f"**Completion**: {completion_percentage:.1f}% ({completed_stages}/{total_stages} stages)",
            "",
            "## Key Metrics",
            f"- **Duration**: {pipeline_summary.get('total_duration_seconds', 0):.1f} seconds",
            f"- **Failed Stages**: {pipeline_summary['stages_summary']['failed']}",
            f"- **Skipped Stages**: {pipeline_summary['stages_summary']['skipped']}",
            "",
            "## Stage Status",
        ]
        
        # Add stage status
        for stage_name, stage_data in pipeline_summary['stages'].items():
            status = stage_data['status']
            emoji = {
                'completed': '✅',
                'failed': '❌',
                'in_progress': '🔄',
                'not_started': '⏳',
                'skipped': '⏭️'
            }.get(status, '❓')
            
            summary_lines.append(f"- {emoji} {stage_name.replace('_', ' ').title()}: {status}")
        
        summary_lines.extend([
            "",
            "## Action Items",
        ])
        
        if pipeline_summary['overall_status'] == 'completed':
            summary_lines.append("- ✅ Pipeline completed successfully")
            summary_lines.append("- 🚀 Ready for production deployment")
        elif pipeline_summary['overall_status'] == 'failed':
            summary_lines.append("- ❌ Critical errors detected")
            summary_lines.append("- 🔧 Immediate attention required")
        else:
            summary_lines.append("- 🔄 Pipeline in progress")
            summary_lines.append("- 📊 Monitor for completion")
        
        return "\n".join(summary_lines)
    
    def create_dashboard_summary(self, pipeline_tracker: PipelineTracker,
                               performance_monitor: Optional[PerformanceMonitor] = None,
                               error_reporter: Optional[ErrorReporter] = None) -> Dict[str, Any]:
        """Create a dashboard-friendly summary with key metrics."""
        pipeline_summary = pipeline_tracker.get_pipeline_summary()
        
        # Calculate key metrics
        total_stages = pipeline_summary['stages_summary']['total']
        completed_stages = pipeline_summary['stages_summary']['completed']
        failed_stages = pipeline_summary['stages_summary']['failed']
        
        completion_rate = (completed_stages / total_stages * 100) if total_stages > 0 else 0
        success_rate = (completed_stages / (completed_stages + failed_stages) * 100) if (completed_stages + failed_stages) > 0 else 0
        
        # Performance metrics
        performance_metrics = {}
        if performance_monitor:
            perf_summary = performance_monitor.get_performance_summary()
            if 'cpu' in perf_summary:
                performance_metrics['cpu_avg'] = perf_summary['cpu']['mean']
                performance_metrics['cpu_peak'] = perf_summary['cpu']['max']
            if 'memory' in perf_summary:
                performance_metrics['memory_avg'] = perf_summary['memory']['mean_percent']
                performance_metrics['memory_peak'] = perf_summary['memory']['max_percent']
            if 'processing' in perf_summary:
                performance_metrics['processing_rate'] = perf_summary['processing']['rate_mean']
        
        # Error metrics
        error_metrics = {}
        if error_reporter:
            error_summary = error_reporter.get_error_summary()
            error_metrics['total_errors'] = error_summary['total_errors']
            error_metrics['resolution_rate'] = error_summary['resolution_rate']
            error_metrics['most_common_category'] = error_summary['most_common_category']
        
        # Create dashboard summary
        dashboard_data = {
            'timestamp': datetime.now().isoformat(),
            'pipeline_status': pipeline_summary['overall_status'],
            'completion_rate': completion_rate,
            'success_rate': success_rate,
            'total_stages': total_stages,
            'completed_stages': completed_stages,
            'failed_stages': failed_stages,
            'duration_seconds': pipeline_summary.get('total_duration_seconds', 0),
            'performance': performance_metrics,
            'errors': error_metrics,
            'stage_details': {
                stage_name: {
                    'status': stage_data['status'],
                    'duration': stage_data['metrics'].get('duration_seconds', 0),
                    'images_processed': stage_data['metrics'].get('images_processed', 0)
                }
                for stage_name, stage_data in pipeline_summary['stages'].items()
            }
        }
        
        return dashboard_data
