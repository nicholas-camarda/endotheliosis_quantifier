#!/usr/bin/env python3
"""Tests for the comprehensive reporting system."""

# Add src to path for imports
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eq.utils.error_reporter import ErrorCategory, ErrorInfo, ErrorReporter, ErrorSeverity
from eq.utils.executive_summary import ExecutiveSummaryGenerator
from eq.utils.performance_metrics import PerformanceMonitor
from eq.utils.pipeline_tracker import (
    PipelineStage,
    PipelineStageInfo,
    PipelineTracker,
    StageMetrics,
    StageStatus,
)
from eq.utils.quality_assessment import (
    FeatureQualityMetrics,
    QualityAssessor,
    ROIQualityMetrics,
    SegmentationQualityMetrics,
)
from eq.utils.stage_reporter import StageReporter


class TestPipelineTracker(unittest.TestCase):
    """Test pipeline progression tracking system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir)
        self.tracker = PipelineTracker(self.output_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test pipeline tracker initialization."""
        self.assertEqual(self.tracker.overall_status, StageStatus.NOT_STARTED)
        self.assertEqual(len(self.tracker.stages), 3)  # 3 pipeline stages
        self.assertTrue(self.tracker.tracking_dir.exists())
    
    def test_start_pipeline(self):
        """Test starting pipeline tracking."""
        config = {"epochs": 10, "batch_size": 8}
        self.tracker.start_pipeline(config)
        
        self.assertEqual(self.tracker.overall_status, StageStatus.IN_PROGRESS)
        self.assertIsNotNone(self.tracker.pipeline_start_time)
        
        # Check config file was saved
        config_file = self.tracker.tracking_dir / "pipeline_config.json"
        self.assertTrue(config_file.exists())
    
    def test_start_stage(self):
        """Test starting a pipeline stage."""
        config = {"epochs": 5}
        self.tracker.start_stage(PipelineStage.SEGMENTATION_TRAINING, config)
        
        stage_info = self.tracker.stages[PipelineStage.SEGMENTATION_TRAINING]
        self.assertEqual(stage_info.status, StageStatus.IN_PROGRESS)
        self.assertIsNotNone(stage_info.metrics.start_time)
        self.assertEqual(stage_info.config, config)
    
    def test_update_stage_metrics(self):
        """Test updating stage metrics."""
        self.tracker.start_stage(PipelineStage.SEGMENTATION_TRAINING, {})
        self.tracker.update_stage_metrics(
            PipelineStage.SEGMENTATION_TRAINING,
            images_processed=100,
            accuracy=0.95
        )
        
        stage_info = self.tracker.stages[PipelineStage.SEGMENTATION_TRAINING]
        self.assertEqual(stage_info.metrics.images_processed, 100)
        self.assertEqual(stage_info.metrics.accuracy, 0.95)
    
    def test_complete_stage(self):
        """Test completing a pipeline stage."""
        self.tracker.start_stage(PipelineStage.SEGMENTATION_TRAINING, {})
        output_paths = ["model.pkl", "results.json"]
        self.tracker.complete_stage(PipelineStage.SEGMENTATION_TRAINING, output_paths)
        
        stage_info = self.tracker.stages[PipelineStage.SEGMENTATION_TRAINING]
        self.assertEqual(stage_info.status, StageStatus.COMPLETED)
        self.assertIsNotNone(stage_info.metrics.end_time)
        self.assertIsNotNone(stage_info.metrics.duration_seconds)
        self.assertEqual(stage_info.output_paths, output_paths)
    
    def test_fail_stage(self):
        """Test failing a pipeline stage."""
        self.tracker.start_stage(PipelineStage.SEGMENTATION_TRAINING, {})
        error = "Model training failed"
        self.tracker.fail_stage(PipelineStage.SEGMENTATION_TRAINING, error)
        
        stage_info = self.tracker.stages[PipelineStage.SEGMENTATION_TRAINING]
        self.assertEqual(stage_info.status, StageStatus.FAILED)
        self.assertIn(error, stage_info.metrics.errors)
    
    def test_get_pipeline_summary(self):
        """Test getting pipeline summary."""
        # Start and complete a stage
        self.tracker.start_pipeline({})
        self.tracker.start_stage(PipelineStage.SEGMENTATION_TRAINING, {})
        self.tracker.complete_stage(PipelineStage.SEGMENTATION_TRAINING)
        
        summary = self.tracker.get_pipeline_summary()
        
        self.assertIn('overall_status', summary)
        self.assertIn('stages_summary', summary)
        self.assertIn('stages', summary)
        self.assertEqual(summary['stages_summary']['completed'], 1)
    
    def test_generate_progression_report(self):
        """Test generating progression report."""
        self.tracker.start_pipeline({})
        self.tracker.start_stage(PipelineStage.SEGMENTATION_TRAINING, {})
        self.tracker.complete_stage(PipelineStage.SEGMENTATION_TRAINING)
        
        report = self.tracker.generate_progression_report()
        
        self.assertIn("Pipeline Progression Report", report)
        self.assertIn("SEGMENTATION TRAINING", report)
        self.assertIn("✅", report)  # Completion emoji


class TestStageReporter(unittest.TestCase):
    """Test stage-specific reporting system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir)
        self.reporter = StageReporter(self.output_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test stage reporter initialization."""
        self.assertTrue(self.reporter.reports_dir.exists())
    
    def test_generate_segmentation_report(self):
        """Test generating segmentation training report."""
        # Create mock stage info
        metrics = StageMetrics(
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_seconds=120.5,
            images_processed=1000,
            accuracy=0.95,
            loss=0.05
        )
        
        stage_info = PipelineStageInfo(
            stage=PipelineStage.SEGMENTATION_TRAINING,
            status=StageStatus.COMPLETED,
            metrics=metrics,
            config={"epochs": 10, "batch_size": 8}
        )
        
        training_history = {
            "train_loss": [0.5, 0.3, 0.2, 0.1, 0.05],
            "val_loss": [0.6, 0.4, 0.25, 0.15, 0.08]
        }
        
        report = self.reporter.generate_segmentation_report(
            stage_info, training_history=training_history
        )
        
        self.assertIn("Segmentation Training Report", report)
        self.assertIn("1000", report)  # Images processed
        self.assertIn("0.95", report)  # Accuracy
        self.assertIn("120.5", report)  # Duration
        self.assertIn("Training History", report)
    
    def test_generate_quantification_report(self):
        """Test generating quantification training report."""
        metrics = StageMetrics(
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_seconds=180.0,
            images_processed=500
        )
        
        stage_info = PipelineStageInfo(
            stage=PipelineStage.QUANTIFICATION_TRAINING,
            status=StageStatus.COMPLETED,
            metrics=metrics,
            config={"epochs": 15}
        )
        
        roi_stats = {
            "total_rois": 1500,
            "avg_roi_size": 2500.5,
            "success_rate": 0.98
        }
        
        report = self.reporter.generate_quantification_report(
            stage_info, roi_stats=roi_stats
        )
        
        self.assertIn("Quantification Training Report", report)
        self.assertIn("1500", report)  # Total ROIs
        self.assertIn("98%", report)  # Success rate
    
    def test_generate_production_report(self):
        """Test generating production inference report."""
        metrics = StageMetrics(
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_seconds=60.0,
            images_processed=200
        )
        
        stage_info = PipelineStageInfo(
            stage=PipelineStage.PRODUCTION_INFERENCE,
            status=StageStatus.COMPLETED,
            metrics=metrics,
            config={"batch_size": 16}
        )
        
        inference_results = {
            "total_predictions": 200,
            "avg_confidence": 0.92,
            "avg_processing_time": 0.3
        }
        
        report = self.reporter.generate_production_report(
            stage_info, inference_results=inference_results
        )
        
        self.assertIn("Production Inference Report", report)
        self.assertIn("200", report)  # Total predictions
        self.assertIn("0.92", report)  # Average confidence


class TestPerformanceMonitor(unittest.TestCase):
    """Test performance metrics collection and monitoring."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir)
        self.monitor = PerformanceMonitor(self.output_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test performance monitor initialization."""
        self.assertTrue(self.monitor.performance_dir.exists())
        self.assertEqual(len(self.monitor.metrics), 0)
        self.assertFalse(self.monitor.monitoring_active)
    
    def test_record_metrics(self):
        """Test recording performance metrics."""
        self.monitor.record_metrics(
            gpu_utilization=85.5,
            processing_rate=10.2,
            epoch=5,
            loss=0.1
        )
        
        self.assertEqual(len(self.monitor.metrics), 1)
        metrics = self.monitor.metrics[0]
        self.assertEqual(metrics.gpu_utilization, 85.5)
        self.assertEqual(metrics.processing_rate, 10.2)
        self.assertEqual(metrics.epoch, 5)
        self.assertEqual(metrics.loss, 0.1)
    
    def test_get_performance_summary(self):
        """Test getting performance summary."""
        # Record some metrics
        self.monitor.record_metrics(cpu_percent=50.0, memory_percent=60.0)
        self.monitor.record_metrics(cpu_percent=70.0, memory_percent=80.0)
        
        summary = self.monitor.get_performance_summary()
        
        self.assertIn('cpu', summary)
        self.assertIn('memory', summary)
        self.assertEqual(summary['cpu']['mean'], 60.0)
        self.assertEqual(summary['memory']['mean_percent'], 70.0)
    
    def test_save_performance_data(self):
        """Test saving performance data."""
        self.monitor.record_metrics(cpu_percent=50.0)
        self.monitor.record_metrics(cpu_percent=70.0)
        
        output_dir = self.monitor.save_performance_data()
        
        metrics_file = output_dir / "performance_metrics.json"
        summary_file = output_dir / "performance_summary.json"
        
        self.assertTrue(metrics_file.exists())
        self.assertTrue(summary_file.exists())


class TestQualityAssessor(unittest.TestCase):
    """Test quality assessment system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir)
        self.assessor = QualityAssessor(self.output_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_assess_segmentation_quality(self):
        """Test segmentation quality assessment."""
        # Create mock masks
        pred_masks = [np.ones((100, 100), dtype=np.uint8)]
        gt_masks = [np.ones((100, 100), dtype=np.uint8)]
        
        metrics = self.assessor.assess_segmentation_quality(pred_masks, gt_masks)
        
        self.assertIsInstance(metrics, SegmentationQualityMetrics)
        self.assertEqual(metrics.dice_coefficient, 1.0)
        self.assertEqual(metrics.iou_score, 1.0)
        self.assertEqual(metrics.precision, 1.0)
        self.assertEqual(metrics.recall, 1.0)
        self.assertEqual(metrics.f1_score, 1.0)
    
    def test_assess_roi_quality(self):
        """Test ROI quality assessment."""
        # Create mock ROIs
        extracted_rois = [np.random.rand(50, 50, 3) for _ in range(5)]
        roi_sizes = [(50, 50), (60, 40), (45, 55), (55, 45), (50, 50)]
        
        metrics = self.assessor.assess_roi_quality(extracted_rois, roi_sizes)
        
        self.assertIsInstance(metrics, ROIQualityMetrics)
        self.assertEqual(metrics.roi_count, 5)
        self.assertEqual(metrics.extraction_success_rate, 1.0)
        self.assertEqual(len(metrics.roi_quality_scores), 5)
    
    def test_assess_feature_quality(self):
        """Test feature quality assessment."""
        # Create mock features
        features = np.random.rand(100, 10)  # 100 samples, 10 features
        feature_names = [f"feature_{i}" for i in range(10)]
        
        metrics = self.assessor.assess_feature_quality(features, feature_names)
        
        self.assertIsInstance(metrics, FeatureQualityMetrics)
        self.assertEqual(metrics.feature_count, 10)
        self.assertEqual(metrics.feature_completeness, 1.0)


class TestErrorReporter(unittest.TestCase):
    """Test error reporting and recovery system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir)
        self.reporter = ErrorReporter(self.output_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test error reporter initialization."""
        self.assertTrue(self.reporter.error_dir.exists())
        self.assertEqual(len(self.reporter.errors), 0)
    
    def test_report_error(self):
        """Test reporting an error."""
        error = FileNotFoundError("Data file not found")
        context = {"stage": "data_loading", "file_path": "/path/to/file"}
        
        error_info = self.reporter.report_error(error, "data_loading", context)
        
        self.assertIsInstance(error_info, ErrorInfo)
        self.assertEqual(error_info.error_type, "FileNotFoundError")
        self.assertEqual(error_info.category, ErrorCategory.DATA_LOADING)
        self.assertEqual(error_info.severity, ErrorSeverity.MEDIUM)
        self.assertGreater(len(error_info.recovery_suggestions), 0)
        self.assertEqual(len(self.reporter.errors), 1)
    
    def test_error_categorization(self):
        """Test automatic error categorization."""
        # Test different error types
        test_cases = [
            (FileNotFoundError("file not found"), ErrorCategory.DATA_LOADING),
            (MemoryError("out of memory"), ErrorCategory.MEMORY),
            (RuntimeError("cuda error"), ErrorCategory.HARDWARE),
        ]
        
        for error, expected_category in test_cases:
            error_info = self.reporter.report_error(error, "test_stage")
            self.assertEqual(error_info.category, expected_category)
    
    def test_get_error_summary(self):
        """Test getting error summary."""
        # Report some errors
        self.reporter.report_error(FileNotFoundError("file1"), "stage1")
        self.reporter.report_error(FileNotFoundError("file2"), "stage2")
        
        summary = self.reporter.get_error_summary()
        
        self.assertEqual(summary['total_errors'], 2)
        self.assertEqual(summary['most_common_category'], 'data_loading')
        self.assertIn('data_loading', summary['category_distribution'])
    
    def test_mark_error_resolved(self):
        """Test marking errors as resolved."""
        error = FileNotFoundError("test error")
        error_info = self.reporter.report_error(error, "test_stage")
        
        self.reporter.mark_error_resolved(error_info, "Fixed by creating file")
        
        self.assertTrue(error_info.resolved)
        self.assertEqual(error_info.resolution_notes, "Fixed by creating file")
        
        summary = self.reporter.get_error_summary()
        self.assertEqual(summary['resolved_errors'], 1)
        self.assertEqual(summary['resolution_rate'], 1.0)


class TestExecutiveSummaryGenerator(unittest.TestCase):
    """Test executive summary generation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir)
        self.generator = ExecutiveSummaryGenerator(self.output_dir)
        self.tracker = PipelineTracker(self.output_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test executive summary generator initialization."""
        self.assertTrue(self.generator.executive_dir.exists())
    
    def test_generate_executive_summary(self):
        """Test generating executive summary."""
        # Set up pipeline state
        self.tracker.start_pipeline({"epochs": 10})
        self.tracker.start_stage(PipelineStage.SEGMENTATION_TRAINING, {"epochs": 5})
        self.tracker.complete_stage(PipelineStage.SEGMENTATION_TRAINING)
        
        summary = self.generator.generate_executive_summary(self.tracker)
        
        self.assertIn("Executive Summary Report", summary)
        self.assertIn("SEGMENTATION TRAINING", summary)
        self.assertIn("✅", summary)  # Completion emoji
        self.assertIn("Key Achievements", summary)
        self.assertIn("Recommendations", summary)
    
    def test_generate_quick_summary(self):
        """Test generating quick summary."""
        # Set up pipeline state
        self.tracker.start_pipeline({})
        self.tracker.start_stage(PipelineStage.SEGMENTATION_TRAINING, {})
        self.tracker.complete_stage(PipelineStage.SEGMENTATION_TRAINING)
        
        summary = self.generator.generate_quick_summary(self.tracker)
        
        self.assertIn("Quick Executive Summary", summary)
        self.assertIn("100.0%", summary)  # Completion percentage
        self.assertIn("✅", summary)  # Completion emoji
    
    def test_create_dashboard_summary(self):
        """Test creating dashboard summary."""
        # Set up pipeline state
        self.tracker.start_pipeline({})
        self.tracker.start_stage(PipelineStage.SEGMENTATION_TRAINING, {})
        self.tracker.complete_stage(PipelineStage.SEGMENTATION_TRAINING)
        
        dashboard_data = self.generator.create_dashboard_summary(self.tracker)
        
        self.assertIn('pipeline_status', dashboard_data)
        self.assertIn('completion_rate', dashboard_data)
        self.assertIn('success_rate', dashboard_data)
        self.assertEqual(dashboard_data['completion_rate'], 100.0)
        self.assertEqual(dashboard_data['success_rate'], 100.0)
    
    def test_save_executive_summary(self):
        """Test saving executive summary."""
        # Set up pipeline state
        self.tracker.start_pipeline({})
        self.tracker.start_stage(PipelineStage.SEGMENTATION_TRAINING, {})
        self.tracker.complete_stage(PipelineStage.SEGMENTATION_TRAINING)
        
        summary_content = self.generator.generate_executive_summary(self.tracker)
        output_file = self.generator.save_executive_summary(summary_content)
        
        self.assertTrue(output_file.exists())
        
        # Check that latest file was created
        latest_file = self.generator.executive_dir / "executive_summary_latest.md"
        self.assertTrue(latest_file.exists())


class TestReportingSystemIntegration(unittest.TestCase):
    """Integration tests for the complete reporting system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir)
        
        # Initialize all reporting components
        self.tracker = PipelineTracker(self.output_dir)
        self.reporter = StageReporter(self.output_dir)
        self.monitor = PerformanceMonitor(self.output_dir)
        self.assessor = QualityAssessor(self.output_dir)
        self.error_reporter = ErrorReporter(self.output_dir)
        self.generator = ExecutiveSummaryGenerator(self.output_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_complete_reporting_workflow(self):
        """Test complete reporting workflow from start to finish."""
        # 1. Start pipeline
        config = {"epochs": 10, "batch_size": 8}
        self.tracker.start_pipeline(config)
        
        # 2. Start and monitor segmentation stage
        self.tracker.start_stage(PipelineStage.SEGMENTATION_TRAINING, config)
        self.monitor.record_metrics(epoch=1, loss=0.5, accuracy=0.8)
        self.monitor.record_metrics(epoch=2, loss=0.3, accuracy=0.9)
        
        # 3. Complete segmentation stage
        self.tracker.complete_stage(PipelineStage.SEGMENTATION_TRAINING)
        
        # 4. Assess quality
        pred_masks = [np.ones((50, 50), dtype=np.uint8)]
        gt_masks = [np.ones((50, 50), dtype=np.uint8)]
        seg_metrics = self.assessor.assess_segmentation_quality(pred_masks, gt_masks)
        
        # 5. Generate stage report
        stage_info = self.tracker.stages[PipelineStage.SEGMENTATION_TRAINING]
        self.reporter.generate_stage_report(
            PipelineStage.SEGMENTATION_TRAINING,
            stage_info,
            training_history={"train_loss": [0.5, 0.3]},
            model_performance={"accuracy": 0.9}
        )
        
        # 6. Report an error (simulated)
        error = FileNotFoundError("Model file not found")
        self.error_reporter.report_error(error, "model_loading")
        
        # 7. Generate executive summary
        summary = self.generator.generate_executive_summary(
            self.tracker,
            performance_monitor=self.monitor,
            error_reporter=self.error_reporter,
            additional_metrics={"segmentation_quality": seg_metrics}
        )
        
        # 8. Verify outputs
        self.assertIn("Executive Summary Report", summary)
        self.assertIn("SEGMENTATION TRAINING", summary)
        self.assertIn("✅", summary)  # Completion emoji
        self.assertIn("FileNotFoundError", summary)  # Error mentioned
        
        # Check that files were created
        self.assertTrue((self.tracker.tracking_dir / "pipeline_state.json").exists())
        self.assertTrue((self.reporter.reports_dir).exists())
        self.assertTrue((self.monitor.performance_dir).exists())
        self.assertTrue((self.error_reporter.error_dir).exists())
        self.assertTrue((self.generator.executive_dir).exists())


if __name__ == '__main__':
    unittest.main()
