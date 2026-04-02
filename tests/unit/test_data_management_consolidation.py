"""
Tests for data management consolidation.

This module tests that the data management consolidation:
1. Files are moved to correct locations
2. Imports work properly after consolidation
3. Data management directory contains all consolidated functionality
4. Moved functions work in their new locations
"""

import pytest
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

class TestDataManagementConsolidation:
    """Test data management consolidation."""
    
    def test_data_management_directory_structure(self):
        """Test that data_management directory contains all moved files."""
        data_mgmt_dir = Path(__file__).parent.parent.parent / 'src' / 'eq' / 'data_management'
        
        # Should contain all moved files
        expected_files = {
            '__init__.py',
            'data_loading.py',
            'model_loading.py',
            'loaders.py',
            'config.py',
            'organize_lucchi_dataset.py',
            'output_manager.py',
            'metadata_processor.py'
        }
        
        actual_files = {f.name for f in data_mgmt_dir.iterdir() if f.is_file() and not f.name.startswith('.')}
        
        for expected_file in expected_files:
            assert expected_file in actual_files, f"data_management directory should contain {expected_file}"
    
    def test_data_directory_removed(self):
        """Test that data directory has been completely removed."""
        data_dir = Path(__file__).parent.parent.parent / 'src' / 'eq' / 'data'
        
        # Data directory should not exist anymore
        assert not data_dir.exists(), "data directory should be completely removed"
    
    def test_utils_directory_cleaned(self):
        """Test that utils directory no longer contains moved files."""
        utils_dir = Path(__file__).parent.parent.parent / 'src' / 'eq' / 'utils'
        
        # Should not contain moved files
        moved_files = {
            'organize_lucchi_dataset.py',
            'output_manager.py',
            'metadata_processor.py'
        }
        
        actual_files = {f.name for f in utils_dir.iterdir() if f.is_file() and not f.name.startswith('.')}
        
        for moved_file in moved_files:
            assert moved_file not in actual_files, f"utils directory should not contain {moved_file}"
    
    def test_data_loading_imports_work(self):
        """Test that data loading functions can be imported from new location."""
        try:
            from eq.data_management.data_loading import get_glom_mask_file, get_glom_y, n_glom_codes
            assert callable(get_glom_mask_file), "get_glom_mask_file should be callable"
            assert callable(get_glom_y), "get_glom_y should be callable"
            assert callable(n_glom_codes), "n_glom_codes should be callable"
        except ImportError as e:
            pytest.fail(f"Failed to import data_loading functions: {e}")
    
    def test_model_loading_imports_work(self):
        """Test that model loading functions can be imported from new location."""
        try:
            from eq.data_management.model_loading import load_model_with_historical_support, get_model_info
            assert callable(load_model_with_historical_support), "load_model_with_historical_support should be callable"
            assert callable(get_model_info), "get_model_info should be callable"
        except ImportError as e:
            pytest.fail(f"Failed to import model_loading functions: {e}")
    
    def test_loaders_imports_work(self):
        """Test that loaders functions can be imported from new location."""
        try:
            from eq.data_management.loaders import load_glomeruli_data, load_mitochondria_patches, UnifiedDataLoader
            assert callable(load_glomeruli_data), "load_glomeruli_data should be callable"
            assert callable(load_mitochondria_patches), "load_mitochondria_patches should be callable"
            assert UnifiedDataLoader is not None, "UnifiedDataLoader should be defined"
        except ImportError as e:
            pytest.fail(f"Failed to import loaders functions: {e}")
    
    def test_config_imports_work(self):
        """Test that config classes can be imported from new location."""
        try:
            from eq.data_management.config import DataConfig, AugmentationConfig
            assert DataConfig is not None, "DataConfig should be defined"
            assert AugmentationConfig is not None, "AugmentationConfig should be defined"
        except ImportError as e:
            pytest.fail(f"Failed to import config classes: {e}")
    
    def test_output_manager_imports_work(self):
        """Test that output manager can be imported from new location."""
        try:
            from eq.data_management.output_manager import OutputManager, create_output_directories
            assert OutputManager is not None, "OutputManager should be defined"
            assert callable(create_output_directories), "create_output_directories should be callable"
        except ImportError as e:
            pytest.fail(f"Failed to import output_manager: {e}")
    
    def test_organize_lucchi_dataset_imports_work(self):
        """Test that organize_lucchi_dataset functions can be imported from new location."""
        try:
            from eq.data_management.organize_lucchi_dataset import organize_lucchi_dataset, extract_tif_stack
            assert callable(organize_lucchi_dataset), "organize_lucchi_dataset should be callable"
            assert callable(extract_tif_stack), "extract_tif_stack should be callable"
        except ImportError as e:
            pytest.fail(f"Failed to import organize_lucchi_dataset functions: {e}")
    
    def test_metadata_processor_imports_work(self):
        """Test that metadata_processor can be imported from new location."""
        try:
            from eq.data_management.metadata_processor import MetadataProcessor, process_metadata_file
            assert MetadataProcessor is not None, "MetadataProcessor should be defined"
            assert callable(process_metadata_file), "process_metadata_file should be callable"
        except ImportError as e:
            pytest.fail(f"Failed to import metadata_processor: {e}")
    
    def test_data_management_module_imports_work(self):
        """Test that the entire data_management module can be imported."""
        try:
            from eq.data_management import (
                load_glomeruli_data,
                OutputManager,
                organize_lucchi_dataset,
                DataConfig,
                MetadataProcessor
            )
            assert callable(load_glomeruli_data), "load_glomeruli_data should be callable"
            assert OutputManager is not None, "OutputManager should be defined"
            assert callable(organize_lucchi_dataset), "organize_lucchi_dataset should be callable"
            assert DataConfig is not None, "DataConfig should be defined"
            assert MetadataProcessor is not None, "MetadataProcessor should be defined"
        except ImportError as e:
            pytest.fail(f"Failed to import data_management module: {e}")
    
    def test_data_management_imports_work(self):
        """Test that data_management imports work (replacing legacy data imports)."""
        try:
            from eq.data_management import load_glomeruli_data, load_mitochondria_patches, DataConfig
            assert callable(load_glomeruli_data), "load_glomeruli_data should be callable"
            assert callable(load_mitochondria_patches), "load_mitochondria_patches should be callable"
            assert DataConfig is not None, "DataConfig should be defined"
        except ImportError as e:
            pytest.fail(f"Failed to import data_management functions: {e}")
    
    def test_moved_functions_work_correctly(self):
        """Test that moved functions work correctly in their new locations."""
        try:
            from eq.data_management.data_loading import get_glom_mask_file
            from eq.data_management.loaders import load_glomeruli_data
            from eq.data_management.output_manager import OutputManager
            from eq.data_management.organize_lucchi_dataset import organize_lucchi_dataset
            from eq.data_management.metadata_processor import MetadataProcessor
            
            # These should not raise exceptions (actual functionality tested elsewhere)
            assert callable(get_glom_mask_file), "get_glom_mask_file should be callable"
            assert callable(load_glomeruli_data), "load_glomeruli_data should be callable"
            assert OutputManager is not None, "OutputManager should be defined"
            assert callable(organize_lucchi_dataset), "organize_lucchi_dataset should be callable"
            assert MetadataProcessor is not None, "MetadataProcessor should be defined"
            
        except Exception as e:
            pytest.fail(f"Moved functions should work correctly: {e}")
