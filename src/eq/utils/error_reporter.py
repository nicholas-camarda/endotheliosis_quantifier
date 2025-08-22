#!/usr/bin/env python3
"""Comprehensive error reporting and recovery suggestions system for the endotheliosis quantifier."""

import json
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from eq.utils.logger import get_logger


class ErrorSeverity(Enum):
    """Enumeration of error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Enumeration of error categories."""
    DATA_LOADING = "data_loading"
    MODEL_LOADING = "model_loading"
    TRAINING = "training"
    INFERENCE = "inference"
    MEMORY = "memory"
    HARDWARE = "hardware"
    CONFIGURATION = "configuration"
    NETWORK = "network"
    PERMISSION = "permission"
    UNKNOWN = "unknown"


@dataclass
class ErrorInfo:
    """Container for error information."""
    timestamp: datetime
    error_type: str
    error_message: str
    severity: ErrorSeverity
    category: ErrorCategory
    stage: str
    stack_trace: str
    context: Dict[str, Any]
    recovery_suggestions: List[str]
    resolved: bool = False
    resolution_notes: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['severity'] = self.severity.value
        data['category'] = self.category.value
        return data


class ErrorReporter:
    """Comprehensive error reporting and recovery system."""
    
    def __init__(self, output_dir: Path):
        """
        Initialize the error reporter.
        
        Args:
            output_dir: Directory to save error reports
        """
        self.output_dir = Path(output_dir)
        self.logger = get_logger("eq.error_reporter")
        
        # Create error reporting directory
        self.error_dir = self.output_dir / "errors"
        self.error_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize error storage
        self.errors: List[ErrorInfo] = []
        self.error_handlers: Dict[ErrorCategory, List[Callable]] = {}
        
        # Register default error handlers
        self._register_default_handlers()
        
        self.logger.info(f"Error reporter initialized for output directory: {output_dir}")
    
    def _register_default_handlers(self) -> None:
        """Register default error handlers for different categories."""
        self.register_error_handler(ErrorCategory.DATA_LOADING, self._handle_data_loading_error)
        self.register_error_handler(ErrorCategory.MODEL_LOADING, self._handle_model_loading_error)
        self.register_error_handler(ErrorCategory.TRAINING, self._handle_training_error)
        self.register_error_handler(ErrorCategory.INFERENCE, self._handle_inference_error)
        self.register_error_handler(ErrorCategory.MEMORY, self._handle_memory_error)
        self.register_error_handler(ErrorCategory.HARDWARE, self._handle_hardware_error)
        self.register_error_handler(ErrorCategory.CONFIGURATION, self._handle_configuration_error)
        self.register_error_handler(ErrorCategory.NETWORK, self._handle_network_error)
        self.register_error_handler(ErrorCategory.PERMISSION, self._handle_permission_error)
    
    def register_error_handler(self, category: ErrorCategory, handler: Callable) -> None:
        """Register an error handler for a specific category."""
        if category not in self.error_handlers:
            self.error_handlers[category] = []
        self.error_handlers[category].append(handler)
    
    def report_error(self, error: Exception, stage: str, context: Dict[str, Any] = None) -> ErrorInfo:
        """Report an error with automatic categorization and recovery suggestions."""
        if context is None:
            context = {}
        
        # Analyze the error
        error_type = type(error).__name__
        error_message = str(error)
        stack_trace = traceback.format_exc()
        
        # Categorize the error
        category = self._categorize_error(error, error_message)
        
        # Determine severity
        severity = self._determine_severity(error, category, context)
        
        # Generate recovery suggestions
        recovery_suggestions = self._generate_recovery_suggestions(error, category, context)
        
        # Create error info
        error_info = ErrorInfo(
            timestamp=datetime.now(),
            error_type=error_type,
            error_message=error_message,
            severity=severity,
            category=category,
            stage=stage,
            stack_trace=stack_trace,
            context=context,
            recovery_suggestions=recovery_suggestions
        )
        
        # Add to error list
        self.errors.append(error_info)
        
        # Log the error
        self.logger.error(f"Error in {stage}: {error_type} - {error_message}")
        self.logger.error(f"Severity: {severity.value}, Category: {category.value}")
        
        # Call category-specific handlers
        if category in self.error_handlers:
            for handler in self.error_handlers[category]:
                try:
                    handler(error_info)
                except Exception as handler_error:
                    self.logger.warning(f"Error handler failed: {handler_error}")
        
        # Save error report
        self._save_error_report(error_info)
        
        return error_info
    
    def _categorize_error(self, error: Exception, error_message: str) -> ErrorCategory:
        """Categorize an error based on its type and message."""
        error_type = type(error).__name__
        error_message_lower = error_message.lower()
        
        # Data loading errors
        if any(keyword in error_message_lower for keyword in ['file', 'path', 'directory', 'not found', 'no such file']):
            return ErrorCategory.DATA_LOADING
        
        # Model loading errors
        if any(keyword in error_message_lower for keyword in ['model', 'checkpoint', 'weights', 'load']):
            return ErrorCategory.MODEL_LOADING
        
        # Training errors
        if any(keyword in error_message_lower for keyword in ['training', 'epoch', 'batch', 'loss', 'gradient']):
            return ErrorCategory.TRAINING
        
        # Inference errors
        if any(keyword in error_message_lower for keyword in ['inference', 'prediction', 'forward']):
            return ErrorCategory.INFERENCE
        
        # Memory errors
        if any(keyword in error_message_lower for keyword in ['memory', 'out of memory', 'oom', 'cuda out of memory']):
            return ErrorCategory.MEMORY
        
        # Hardware errors
        if any(keyword in error_message_lower for keyword in ['cuda', 'gpu', 'device', 'hardware', 'mps']):
            return ErrorCategory.HARDWARE
        
        # Configuration errors
        if any(keyword in error_message_lower for keyword in ['config', 'parameter', 'argument', 'setting']):
            return ErrorCategory.CONFIGURATION
        
        # Network errors
        if any(keyword in error_message_lower for keyword in ['network', 'connection', 'timeout', 'http']):
            return ErrorCategory.NETWORK
        
        # Permission errors
        if any(keyword in error_message_lower for keyword in ['permission', 'access', 'denied', 'forbidden']):
            return ErrorCategory.PERMISSION
        
        return ErrorCategory.UNKNOWN
    
    def _determine_severity(self, error: Exception, category: ErrorCategory, context: Dict[str, Any]) -> ErrorSeverity:
        """Determine the severity of an error."""
        error_message_lower = str(error).lower()
        
        # Critical errors
        if any(keyword in error_message_lower for keyword in ['cuda out of memory', 'segmentation fault', 'kernel panic']):
            return ErrorSeverity.CRITICAL
        
        # High severity errors
        if category in [ErrorCategory.MEMORY, ErrorCategory.HARDWARE]:
            return ErrorSeverity.HIGH
        
        if any(keyword in error_message_lower for keyword in ['model', 'checkpoint', 'weights']):
            return ErrorSeverity.HIGH
        
        # Medium severity errors
        if category in [ErrorCategory.DATA_LOADING, ErrorCategory.CONFIGURATION]:
            return ErrorSeverity.MEDIUM
        
        # Low severity errors
        if category in [ErrorCategory.NETWORK, ErrorCategory.PERMISSION]:
            return ErrorSeverity.LOW
        
        return ErrorSeverity.MEDIUM
    
    def _generate_recovery_suggestions(self, error: Exception, category: ErrorCategory, context: Dict[str, Any]) -> List[str]:
        """Generate recovery suggestions based on error category and context."""
        suggestions = []
        error_message_lower = str(error).lower()
        
        if category == ErrorCategory.DATA_LOADING:
            suggestions.extend([
                "Check if the data directory exists and contains the expected files",
                "Verify file permissions and ensure read access",
                "Check file format compatibility (supported: .jpg, .png, .tiff)",
                "Ensure data directory structure matches expected format",
                "Try using absolute paths instead of relative paths"
            ])
        
        elif category == ErrorCategory.MODEL_LOADING:
            suggestions.extend([
                "Verify the model file exists and is not corrupted",
                "Check if the model was saved with the same framework version",
                "Ensure all required dependencies are installed",
                "Try loading the model in a different environment",
                "Check if the model file path is correct"
            ])
        
        elif category == ErrorCategory.TRAINING:
            suggestions.extend([
                "Reduce batch size to decrease memory usage",
                "Check if training data is properly formatted",
                "Verify learning rate and optimizer settings",
                "Ensure GPU memory is sufficient for the model",
                "Try using mixed precision training to reduce memory usage"
            ])
        
        elif category == ErrorCategory.INFERENCE:
            suggestions.extend([
                "Check if the model is properly loaded",
                "Verify input data format and preprocessing",
                "Ensure sufficient memory for inference",
                "Check if the model expects the correct input shape",
                "Try running inference on a smaller batch"
            ])
        
        elif category == ErrorCategory.MEMORY:
            suggestions.extend([
                "Reduce batch size or image size",
                "Close other applications to free memory",
                "Use gradient checkpointing if available",
                "Try using CPU instead of GPU for processing",
                "Check available system memory and GPU memory"
            ])
        
        elif category == ErrorCategory.HARDWARE:
            suggestions.extend([
                "Check GPU drivers and CUDA installation",
                "Verify hardware compatibility with the framework",
                "Try using CPU fallback mode",
                "Check GPU temperature and cooling",
                "Ensure proper hardware initialization"
            ])
        
        elif category == ErrorCategory.CONFIGURATION:
            suggestions.extend([
                "Check configuration file format and syntax",
                "Verify all required parameters are provided",
                "Ensure parameter values are within valid ranges",
                "Check environment variables and settings",
                "Review configuration documentation"
            ])
        
        elif category == ErrorCategory.NETWORK:
            suggestions.extend([
                "Check internet connection",
                "Verify network firewall settings",
                "Try using a different network connection",
                "Check if the remote server is accessible",
                "Increase network timeout settings"
            ])
        
        elif category == ErrorCategory.PERMISSION:
            suggestions.extend([
                "Check file and directory permissions",
                "Run with appropriate user privileges",
                "Verify write permissions for output directories",
                "Check if the user has access to required resources",
                "Try creating output directories manually"
            ])
        
        # Add general suggestions
        suggestions.extend([
            "Check the logs for more detailed error information",
            "Verify all dependencies are installed and up to date",
            "Try running with QUICK_TEST=true for faster debugging",
            "Check if the error occurs consistently or intermittently"
        ])
        
        return suggestions
    
    def _handle_data_loading_error(self, error_info: ErrorInfo) -> None:
        """Handle data loading errors."""
        self.logger.warning("Data loading error detected - check data directory and file formats")
    
    def _handle_model_loading_error(self, error_info: ErrorInfo) -> None:
        """Handle model loading errors."""
        self.logger.warning("Model loading error detected - check model file and dependencies")
    
    def _handle_training_error(self, error_info: ErrorInfo) -> None:
        """Handle training errors."""
        self.logger.warning("Training error detected - consider reducing batch size or memory usage")
    
    def _handle_inference_error(self, error_info: ErrorInfo) -> None:
        """Handle inference errors."""
        self.logger.warning("Inference error detected - check model and input data")
    
    def _handle_memory_error(self, error_info: ErrorInfo) -> None:
        """Handle memory errors."""
        self.logger.error("Memory error detected - this is a critical issue requiring immediate attention")
    
    def _handle_hardware_error(self, error_info: ErrorInfo) -> None:
        """Handle hardware errors."""
        self.logger.error("Hardware error detected - check GPU drivers and hardware compatibility")
    
    def _handle_configuration_error(self, error_info: ErrorInfo) -> None:
        """Handle configuration errors."""
        self.logger.warning("Configuration error detected - check parameter values and settings")
    
    def _handle_network_error(self, error_info: ErrorInfo) -> None:
        """Handle network errors."""
        self.logger.warning("Network error detected - check internet connection and firewall settings")
    
    def _handle_permission_error(self, error_info: ErrorInfo) -> None:
        """Handle permission errors."""
        self.logger.warning("Permission error detected - check file and directory permissions")
    
    def _save_error_report(self, error_info: ErrorInfo) -> None:
        """Save an individual error report to disk."""
        timestamp = error_info.timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"error_{error_info.category.value}_{timestamp}.json"
        filepath = self.error_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(error_info.to_dict(), f, indent=2)
        
        self.logger.debug(f"Error report saved: {filepath}")
    
    def mark_error_resolved(self, error_info: ErrorInfo, resolution_notes: str = None) -> None:
        """Mark an error as resolved."""
        error_info.resolved = True
        error_info.resolution_notes = resolution_notes
        
        self.logger.info(f"Error marked as resolved: {error_info.error_type}")
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get a summary of all reported errors."""
        if not self.errors:
            return {"total_errors": 0}
        
        # Count errors by category and severity
        category_counts = {}
        severity_counts = {}
        stage_counts = {}
        
        for error in self.errors:
            # Category counts
            category = error.category.value
            category_counts[category] = category_counts.get(category, 0) + 1
            
            # Severity counts
            severity = error.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            # Stage counts
            stage = error.stage
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
        
        # Calculate resolution rate
        resolved_count = sum(1 for error in self.errors if error.resolved)
        resolution_rate = resolved_count / len(self.errors) if self.errors else 0
        
        return {
            "total_errors": len(self.errors),
            "resolved_errors": resolved_count,
            "resolution_rate": resolution_rate,
            "category_distribution": category_counts,
            "severity_distribution": severity_counts,
            "stage_distribution": stage_counts,
            "most_common_category": max(category_counts.items(), key=lambda x: x[1])[0] if category_counts else None,
            "most_common_severity": max(severity_counts.items(), key=lambda x: x[1])[0] if severity_counts else None
        }
    
    def generate_error_report(self) -> str:
        """Generate a comprehensive error report."""
        summary = self.get_error_summary()
        
        report_lines = [
            "# Error Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            f"- **Total Errors**: {summary['total_errors']}",
            f"- **Resolved Errors**: {summary['resolved_errors']}",
            f"- **Resolution Rate**: {summary['resolution_rate']:.2%}",
            f"- **Most Common Category**: {summary['most_common_category']}",
            f"- **Most Common Severity**: {summary['most_common_severity']}",
            "",
        ]
        
        if summary['category_distribution']:
            report_lines.extend([
                "## Error Distribution by Category",
            ])
            for category, count in summary['category_distribution'].items():
                percentage = (count / summary['total_errors']) * 100
                report_lines.append(f"- **{category}**: {count} ({percentage:.1f}%)")
            report_lines.append("")
        
        if summary['severity_distribution']:
            report_lines.extend([
                "## Error Distribution by Severity",
            ])
            for severity, count in summary['severity_distribution'].items():
                percentage = (count / summary['total_errors']) * 100
                report_lines.append(f"- **{severity}**: {count} ({percentage:.1f}%)")
            report_lines.append("")
        
        if summary['stage_distribution']:
            report_lines.extend([
                "## Error Distribution by Pipeline Stage",
            ])
            for stage, count in summary['stage_distribution'].items():
                percentage = (count / summary['total_errors']) * 100
                report_lines.append(f"- **{stage}**: {count} ({percentage:.1f}%)")
            report_lines.append("")
        
        # Add recent errors
        recent_errors = sorted(self.errors, key=lambda x: x.timestamp, reverse=True)[:10]
        if recent_errors:
            report_lines.extend([
                "## Recent Errors (Last 10)",
            ])
            
            for error in recent_errors:
                status = "✅ RESOLVED" if error.resolved else "❌ UNRESOLVED"
                report_lines.extend([
                    f"### {error.error_type} - {status}",
                    f"- **Time**: {error.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
                    f"- **Stage**: {error.stage}",
                    f"- **Severity**: {error.severity.value}",
                    f"- **Category**: {error.category.value}",
                    f"- **Message**: {error.error_message}",
                ])
                
                if error.recovery_suggestions:
                    report_lines.append("- **Recovery Suggestions**:")
                    for suggestion in error.recovery_suggestions[:3]:  # Show top 3
                        report_lines.append(f"  - {suggestion}")
                
                if error.resolution_notes:
                    report_lines.append(f"- **Resolution Notes**: {error.resolution_notes}")
                
                report_lines.append("")
        
        return "\n".join(report_lines)
    
    def save_error_summary(self) -> Path:
        """Save error summary to disk."""
        # Save summary JSON
        summary_file = self.error_dir / "error_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(self.get_error_summary(), f, indent=2)
        
        # Save comprehensive report
        report_file = self.error_dir / "error_report.md"
        with open(report_file, 'w') as f:
            f.write(self.generate_error_report())
        
        self.logger.info(f"Error summary saved to: {self.error_dir}")
        return self.error_dir
