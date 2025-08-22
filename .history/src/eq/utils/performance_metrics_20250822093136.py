#!/usr/bin/env python3
"""Performance metrics collection and display system for the endotheliosis quantifier."""

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import psutil

from eq.utils.logger import get_logger


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    gpu_utilization: Optional[float] = None
    gpu_memory_used_mb: Optional[float] = None
    processing_rate: Optional[float] = None
    batch_size: Optional[int] = None
    epoch: Optional[int] = None
    loss: Optional[float] = None
    accuracy: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class PerformanceMonitor:
    """Monitors and collects performance metrics during pipeline execution."""
    
    def __init__(self, output_dir: Path):
        """
        Initialize the performance monitor.
        
        Args:
            output_dir: Directory to save performance data
        """
        self.output_dir = Path(output_dir)
        self.logger = get_logger("eq.performance_monitor")
        
        # Create performance directory
        self.performance_dir = self.output_dir / "performance"
        self.performance_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics storage
        self.metrics: List[PerformanceMetrics] = []
        self.monitoring_active = False
        self.monitor_thread = None
        
        self.logger.info(f"Performance monitor initialized for output directory: {output_dir}")
    
    def start_monitoring(self, interval: float = 1.0) -> None:
        """Start continuous performance monitoring."""
        if self.monitoring_active:
            self.logger.warning("Performance monitoring already active")
            return
        
        self.monitoring_active = True
        self.logger.info(f"Started performance monitoring (interval: {interval}s)")
        
        # Start monitoring in a separate thread
        import threading
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> None:
        """Stop continuous performance monitoring."""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        self.logger.info("Stopped performance monitoring")
    
    def record_metrics(self, **kwargs) -> None:
        """Record a single performance measurement."""
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            # Create metrics object
            metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / (1024 * 1024),
                **kwargs
            )
            
            self.metrics.append(metrics)
            
        except Exception as e:
            self.logger.error(f"Error recording performance metrics: {e}")
    
    def get_gpu_metrics(self) -> Tuple[Optional[float], Optional[float]]:
        """Get GPU utilization and memory usage if available."""
        try:
            # Try to get GPU metrics using nvidia-ml-py if available
            import pynvml
            pynvml.nvmlInit()
            
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count > 0:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                gpu_utilization = utilization.gpu
                gpu_memory_used_mb = memory_info.used / (1024 * 1024)
                
                return gpu_utilization, gpu_memory_used_mb
        except ImportError:
            self.logger.debug("pynvml not available, skipping GPU metrics")
        except Exception as e:
            self.logger.debug(f"Error getting GPU metrics: {e}")
        
        return None, None
    
    def _monitor_loop(self, interval: float) -> None:
        """Continuous monitoring loop."""
        while self.monitoring_active:
            try:
                gpu_util, gpu_memory = self.get_gpu_metrics()
                self.record_metrics(
                    gpu_utilization=gpu_util,
                    gpu_memory_used_mb=gpu_memory
                )
                time.sleep(interval)
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of collected performance metrics."""
        if not self.metrics:
            return {"error": "No metrics collected"}
        
        # Calculate statistics
        cpu_percentages = [m.cpu_percent for m in self.metrics]
        memory_percentages = [m.memory_percent for m in self.metrics]
        memory_used_mb = [m.memory_used_mb for m in self.metrics]
        
        gpu_utilizations = [m.gpu_utilization for m in self.metrics if m.gpu_utilization is not None]
        gpu_memory_used = [m.gpu_memory_used_mb for m in self.metrics if m.gpu_memory_used_mb is not None]
        
        processing_rates = [m.processing_rate for m in self.metrics if m.processing_rate is not None]
        
        summary = {
            "total_measurements": len(self.metrics),
            "monitoring_duration_seconds": (self.metrics[-1].timestamp - self.metrics[0].timestamp).total_seconds(),
            "cpu": {
                "mean": np.mean(cpu_percentages),
                "max": np.max(cpu_percentages),
                "min": np.min(cpu_percentages),
                "std": np.std(cpu_percentages)
            },
            "memory": {
                "mean_percent": np.mean(memory_percentages),
                "max_percent": np.max(memory_percentages),
                "mean_used_mb": np.mean(memory_used_mb),
                "max_used_mb": np.max(memory_used_mb)
            }
        }
        
        if gpu_utilizations:
            summary["gpu"] = {
                "utilization_mean": np.mean(gpu_utilizations),
                "utilization_max": np.max(gpu_utilizations),
                "memory_mean_mb": np.mean(gpu_memory_used),
                "memory_max_mb": np.max(gpu_memory_used)
            }
        
        if processing_rates:
            summary["processing"] = {
                "rate_mean": np.mean(processing_rates),
                "rate_max": np.max(processing_rates),
                "rate_min": np.min(processing_rates)
            }
        
        return summary
    
    def save_performance_data(self) -> Path:
        """Save collected performance data to disk."""
        # Save raw metrics
        metrics_file = self.performance_dir / "performance_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump([m.to_dict() for m in self.metrics], f, indent=2)
        
        # Save summary
        summary_file = self.performance_dir / "performance_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(self.get_performance_summary(), f, indent=2)
        
        self.logger.info(f"Performance data saved to: {self.performance_dir}")
        return self.performance_dir


class PerformanceVisualizer:
    """Creates visualizations of performance metrics."""
    
    def __init__(self, output_dir: Path):
        """
        Initialize the performance visualizer.
        
        Args:
            output_dir: Directory to save performance plots
        """
        self.output_dir = Path(output_dir)
        self.logger = get_logger("eq.performance_visualizer")
        
        # Create plots directory
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Performance visualizer initialized for output directory: {output_dir}")
    
    def create_performance_plots(self, metrics: List[PerformanceMetrics]) -> List[Path]:
        """Create performance visualization plots."""
        if not metrics:
            self.logger.warning("No metrics provided for visualization")
            return []
        
        plot_paths = []
        
        # Extract data
        timestamps = [m.timestamp for m in metrics]
        cpu_percentages = [m.cpu_percent for m in metrics]
        memory_percentages = [m.memory_percent for m in metrics]
        memory_used_mb = [m.memory_used_mb for m in metrics]
        
        gpu_utilizations = [m.gpu_utilization for m in metrics if m.gpu_utilization is not None]
        gpu_timestamps = [m.timestamp for m in metrics if m.gpu_utilization is not None]
        
        processing_rates = [m.processing_rate for m in metrics if m.processing_rate is not None]
        processing_timestamps = [m.timestamp for m in metrics if m.processing_rate is not None]
        
        # Create CPU and Memory plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # CPU usage
        ax1.plot(timestamps, cpu_percentages, 'b-', label='CPU Usage (%)')
        ax1.set_ylabel('CPU Usage (%)')
        ax1.set_title('System Performance Metrics')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Memory usage
        ax2.plot(timestamps, memory_percentages, 'r-', label='Memory Usage (%)')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Memory Usage (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        cpu_memory_plot = self.plots_dir / "cpu_memory_usage.png"
        plt.savefig(cpu_memory_plot, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths.append(cpu_memory_plot)
        
        # Create GPU plot if GPU data is available
        if gpu_utilizations:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # GPU utilization
            ax1.plot(gpu_timestamps, gpu_utilizations, 'g-', label='GPU Utilization (%)')
            ax1.set_ylabel('GPU Utilization (%)')
            ax1.set_title('GPU Performance Metrics')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # GPU memory usage
            gpu_memory_used = [m.gpu_memory_used_mb for m in metrics if m.gpu_memory_used_mb is not None]
            ax2.plot(gpu_timestamps, gpu_memory_used, 'm-', label='GPU Memory Used (MB)')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('GPU Memory (MB)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            gpu_plot = self.plots_dir / "gpu_performance.png"
            plt.savefig(gpu_plot, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths.append(gpu_plot)
        
        # Create processing rate plot if available
        if processing_rates:
            plt.figure(figsize=(12, 6))
            plt.plot(processing_timestamps, processing_rates, 'c-', label='Processing Rate (images/sec)')
            plt.xlabel('Time')
            plt.ylabel('Processing Rate (images/sec)')
            plt.title('Processing Performance')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            processing_plot = self.plots_dir / "processing_rate.png"
            plt.savefig(processing_plot, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths.append(processing_plot)
        
        # Create training metrics plot if available
        training_metrics = [(m.epoch, m.loss, m.accuracy) for m in metrics 
                           if m.epoch is not None and m.loss is not None]
        
        if training_metrics:
            epochs, losses, accuracies = zip(*training_metrics)
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Loss plot
            ax1.plot(epochs, losses, 'r-', label='Training Loss')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training Metrics')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Accuracy plot
            if any(acc is not None for acc in accuracies):
                valid_accuracies = [(e, acc) for e, acc in zip(epochs, accuracies) if acc is not None]
                if valid_accuracies:
                    valid_epochs, valid_accs = zip(*valid_accuracies)
                    ax2.plot(valid_epochs, valid_accs, 'b-', label='Accuracy')
                    ax2.set_xlabel('Epoch')
                    ax2.set_ylabel('Accuracy')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            training_plot = self.plots_dir / "training_metrics.png"
            plt.savefig(training_plot, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths.append(training_plot)
        
        self.logger.info(f"Created {len(plot_paths)} performance plots")
        return plot_paths
    
    def generate_performance_report(self, metrics: List[PerformanceMetrics]) -> str:
        """Generate a text report of performance metrics."""
        if not metrics:
            return "No performance metrics available."
        
        summary = PerformanceMonitor(self.output_dir).get_performance_summary()
        
        report_lines = [
            "# Performance Metrics Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            f"- **Total Measurements**: {summary['total_measurements']}",
            f"- **Monitoring Duration**: {summary['monitoring_duration_seconds']:.2f} seconds",
            "",
            "## CPU Performance",
            f"- **Mean Usage**: {summary['cpu']['mean']:.2f}%",
            f"- **Peak Usage**: {summary['cpu']['max']:.2f}%",
            f"- **Minimum Usage**: {summary['cpu']['min']:.2f}%",
            f"- **Standard Deviation**: {summary['cpu']['std']:.2f}%",
            "",
            "## Memory Performance",
            f"- **Mean Usage**: {summary['memory']['mean_percent']:.2f}%",
            f"- **Peak Usage**: {summary['memory']['max_percent']:.2f}%",
            f"- **Mean Memory Used**: {summary['memory']['mean_used_mb']:.2f} MB",
            f"- **Peak Memory Used**: {summary['memory']['max_used_mb']:.2f} MB",
        ]
        
        if 'gpu' in summary:
            report_lines.extend([
                "",
                "## GPU Performance",
                f"- **Mean Utilization**: {summary['gpu']['utilization_mean']:.2f}%",
                f"- **Peak Utilization**: {summary['gpu']['utilization_max']:.2f}%",
                f"- **Mean Memory Used**: {summary['gpu']['memory_mean_mb']:.2f} MB",
                f"- **Peak Memory Used**: {summary['gpu']['memory_max_mb']:.2f} MB",
            ])
        
        if 'processing' in summary:
            report_lines.extend([
                "",
                "## Processing Performance",
                f"- **Mean Processing Rate**: {summary['processing']['rate_mean']:.2f} images/sec",
                f"- **Peak Processing Rate**: {summary['processing']['rate_max']:.2f} images/sec",
                f"- **Minimum Processing Rate**: {summary['processing']['rate_min']:.2f} images/sec",
            ])
        
        return "\n".join(report_lines)
