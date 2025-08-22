#!/usr/bin/env python3
"""Tests for the output directory management system."""

import json
import tempfile
from datetime import datetime

import pytest

from eq.utils.output_manager import OutputManager, create_output_directories


class TestOutputManager:
    """Test cases for the OutputManager class."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary output directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def output_manager(self, temp_output_dir):
        """Create an OutputManager instance with temporary directory."""
        return OutputManager(temp_output_dir)
    
    def test_create_output_directory_basic(self, output_manager):
        """Test basic output directory creation."""
        data_source = "preeclampsia_data"
        run_type = "production"
        
        output_dirs = output_manager.create_output_directory(data_source, run_type)
        
        # Check that all required directories were created
        assert output_dirs['main'].exists()
        assert output_dirs['models'].exists()
        assert output_dirs['plots'].exists()
        assert output_dirs['results'].exists()
        assert output_dirs['reports'].exists()
        assert output_dirs['logs'].exists()
        assert output_dirs['cache'].exists()
        
        # Check directory name format
        dir_name = output_dirs['main'].name
        assert data_source in dir_name
        assert run_type in dir_name
        assert len(dir_name.split('_')) >= 3  # data_source_timestamp_run_type
        
        # Check metadata file was created
        metadata_file = output_dirs['main'] / "run_metadata.json"
        assert metadata_file.exists()
        
        # Verify metadata content
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        assert metadata['data_source'] == data_source
        assert metadata['run_type'] == run_type
        assert 'timestamp' in metadata
        assert 'created_at' in metadata
    
    def test_create_output_directory_with_custom_timestamp(self, output_manager):
        """Test output directory creation with custom timestamp."""
        data_source = "test_data"
        run_type = "quick"
        custom_timestamp = "2025-01-15_143022"
        
        output_dirs = output_manager.create_output_directory(
            data_source, run_type, timestamp=custom_timestamp
        )
        
        # Check directory name contains custom timestamp
        dir_name = output_dirs['main'].name
        assert custom_timestamp in dir_name
        
        # Verify metadata contains custom timestamp
        metadata_file = output_dirs['main'] / "run_metadata.json"
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        assert metadata['timestamp'] == custom_timestamp
    
    def test_create_output_directory_with_custom_suffix(self, output_manager):
        """Test output directory creation with custom suffix."""
        data_source = "experiment_data"
        run_type = "development"
        custom_suffix = "v2_optimized"
        
        output_dirs = output_manager.create_output_directory(
            data_source, run_type, custom_suffix=custom_suffix
        )
        
        # Check directory name contains custom suffix
        dir_name = output_dirs['main'].name
        assert custom_suffix in dir_name
        
        # Verify metadata contains custom suffix
        metadata_file = output_dirs['main'] / "run_metadata.json"
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        assert metadata['custom_suffix'] == custom_suffix
    
    def test_get_data_source_name_extraction(self, output_manager):
        """Test data source name extraction from various path formats."""
        # Test with data directory name
        assert output_manager.get_data_source_name("data/preeclampsia_data") == "Preeclampsia"
        assert output_manager.get_data_source_name("data/experiment_data") == "Experiment"
        
        # Test with full path
        assert output_manager.get_data_source_name("/path/to/data/study_data") == "Study"
        
        # Test with underscore handling
        assert output_manager.get_data_source_name("data/my_experiment_data") == "My_Experiment"
        
        # Test fallback for empty name
        assert output_manager.get_data_source_name("") == "unknown_data_source"
    
    def test_create_run_summary(self, output_manager):
        """Test run summary creation."""
        data_source = "test_data"
        run_type = "production"
        
        output_dirs = output_manager.create_output_directory(data_source, run_type)
        
        # Create some test files to list
        (output_dirs['models'] / "model.pkl").touch()
        (output_dirs['plots'] / "training_curves.png").touch()
        (output_dirs['results'] / "results.json").touch()
        
        run_info = {
            'data_source': data_source,
            'run_type': run_type,
            'config': {'epochs': 10, 'batch_size': 4},
            'results': {'accuracy': 0.95, 'loss': 0.05}
        }
        
        output_manager.create_run_summary(output_dirs, run_info)
        
        # Check summary file was created
        summary_file = output_dirs['reports'] / "run_summary.md"
        assert summary_file.exists()
        
        # Verify summary content
        with open(summary_file, 'r') as f:
            summary_content = f.read()
        
        assert data_source in summary_content
        assert run_type in summary_content
        assert "model.pkl" in summary_content
        assert "training_curves.png" in summary_content
        assert "results.json" in summary_content
    
    def test_cleanup_old_runs(self, output_manager):
        """Test cleanup of old output directories."""
        # Create some test directories with timestamps
        old_timestamp = "2024-01-01_120000"  # Old
        recent_timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')  # Recent
        
        old_dir = output_manager.base_output_dir / f"old_data_{old_timestamp}_production"
        recent_dir = output_manager.base_output_dir / f"recent_data_{recent_timestamp}_production"
        
        old_dir.mkdir(parents=True, exist_ok=True)
        recent_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a file in each directory
        (old_dir / "test.txt").touch()
        (recent_dir / "test.txt").touch()
        
        # Run cleanup
        output_manager.cleanup_old_runs(max_age_days=30)
        
        # Check that old directory was removed but recent one remains
        assert not old_dir.exists()
        assert recent_dir.exists()
    
    def test_different_run_types(self, output_manager):
        """Test output directory creation with different run types."""
        data_source = "test_data"
        run_types = ["quick", "production", "smoke", "development"]
        
        for run_type in run_types:
            output_dirs = output_manager.create_output_directory(data_source, run_type)
            
            # Check directory name contains run type
            dir_name = output_dirs['main'].name
            assert run_type in dir_name
            
            # Verify metadata contains correct run type
            metadata_file = output_dirs['main'] / "run_metadata.json"
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            assert metadata['run_type'] == run_type


class TestCreateOutputDirectoriesFunction:
    """Test cases for the convenience function."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary output directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    def test_create_output_directories_function(self, temp_output_dir):
        """Test the convenience function for creating output directories."""
        data_source = "experiment_data"
        run_type = "quick"
        
        output_dirs = create_output_directories(
            data_source_name=data_source,
            run_type=run_type,
            base_output_dir=temp_output_dir
        )
        
        # Check that directories were created
        assert output_dirs['main'].exists()
        assert output_dirs['models'].exists()
        assert output_dirs['plots'].exists()
        assert output_dirs['results'].exists()
        assert output_dirs['reports'].exists()
        assert output_dirs['logs'].exists()
        assert output_dirs['cache'].exists()
        
        # Check directory name format
        dir_name = output_dirs['main'].name
        assert data_source in dir_name
        assert run_type in dir_name
    
    def test_create_output_directories_with_all_parameters(self, temp_output_dir):
        """Test the convenience function with all parameters."""
        data_source = "test_data"
        run_type = "production"
        timestamp = "2025-01-15_143022"
        custom_suffix = "experiment_v1"
        
        output_dirs = create_output_directories(
            data_source_name=data_source,
            run_type=run_type,
            timestamp=timestamp,
            custom_suffix=custom_suffix,
            base_output_dir=temp_output_dir
        )
        
        # Check directory name contains all components
        dir_name = output_dirs['main'].name
        assert data_source in dir_name
        assert timestamp in dir_name
        assert run_type in dir_name
        assert custom_suffix in dir_name


class TestOutputDirectoryIntegration:
    """Integration tests for output directory system."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary output directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    def test_full_pipeline_integration(self, temp_output_dir):
        """Test integration with a simulated pipeline run."""
        output_manager = OutputManager(temp_output_dir)
        
        # Simulate pipeline run
        data_source = "preeclampsia_data"
        run_type = "production"
        
        # Create output directories
        output_dirs = output_manager.create_output_directory(data_source, run_type)
        
        # Simulate pipeline outputs
        # Models
        model_file = output_dirs['models'] / "segmentation_model.pkl"
        model_file.write_text("model_data")
        
        # Plots
        plot_file = output_dirs['plots'] / "training_curves.png"
        plot_file.write_text("plot_data")
        
        # Results
        results_file = output_dirs['results'] / "segmentation_results.json"
        results_file.write_text('{"accuracy": 0.95, "dice_score": 0.87}')
        
        # Create run summary
        run_info = {
            'data_source': data_source,
            'run_type': run_type,
            'config': {
                'epochs': 10,
                'batch_size': 4,
                'learning_rate': 0.001
            },
            'results': {
                'final_accuracy': 0.95,
                'final_loss': 0.05,
                'training_time': '2h 15m'
            }
        }
        
        output_manager.create_run_summary(output_dirs, run_info)
        
        # Verify all files exist
        assert model_file.exists()
        assert plot_file.exists()
        assert results_file.exists()
        
        # Verify summary file
        summary_file = output_dirs['reports'] / "run_summary.md"
        assert summary_file.exists()
        
        # Check summary content
        with open(summary_file, 'r') as f:
            summary_content = f.read()
        
        assert "segmentation_model.pkl" in summary_content
        assert "training_curves.png" in summary_content
        assert "segmentation_results.json" in summary_content
        assert "0.95" in summary_content  # accuracy from results
    
    def test_multiple_runs_same_data_source(self, temp_output_dir):
        """Test multiple runs with the same data source."""
        output_manager = OutputManager(temp_output_dir)
        data_source = "experiment_data"
        
        # Create multiple runs
        run1_dirs = output_manager.create_output_directory(data_source, "quick")
        run2_dirs = output_manager.create_output_directory(data_source, "production")
        run3_dirs = output_manager.create_output_directory(data_source, "development")
        
        # Verify all directories exist and are different
        assert run1_dirs['main'] != run2_dirs['main']
        assert run2_dirs['main'] != run3_dirs['main']
        assert run1_dirs['main'] != run3_dirs['main']
        
        # Verify all contain the data source name
        assert data_source in run1_dirs['main'].name
        assert data_source in run2_dirs['main'].name
        assert data_source in run3_dirs['main'].name
        
        # Verify run types are different
        assert "quick" in run1_dirs['main'].name
        assert "production" in run2_dirs['main'].name
        assert "development" in run3_dirs['main'].name


if __name__ == "__main__":
    pytest.main([__file__])
