"""
Tests for core directory reorganization.

This module tests that the core directory reorganization:
1. Files are moved to correct locations
2. Imports work properly after reorganization
3. Core directory contains only constants, types, and abstract interfaces
4. Moved functions work in their new locations
"""

import pytest
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

class TestCoreReorganization:
    """Test core directory reorganization."""
    
    def test_core_directory_structure(self):
        """Test that core directory contains only expected files."""
        core_dir = Path(__file__).parent.parent.parent / 'src' / 'eq' / 'core'
        
        # Core should only contain these files
        expected_files = {
            '__init__.py',
            'constants.py', 
            'types.py'  # New file to be created
        }
        
        actual_files = {f.name for f in core_dir.iterdir() if f.is_file() and not f.name.startswith('.')}
        
        # Should not contain moved files
        moved_files = {
            'data_loading.py',
            'preprocessing.py',
            'model_loading.py'
        }
        
        for moved_file in moved_files:
            assert moved_file not in actual_files, f"Core directory should not contain {moved_file}"
        
        # Should contain expected files
        for expected_file in expected_files:
            assert expected_file in actual_files, f"Core directory should contain {expected_file}"
    
    def test_data_management_directory_structure(self):
        """Test that data_management directory contains moved files."""
        data_mgmt_dir = Path(__file__).parent.parent.parent / 'src' / 'eq' / 'data_management'
        
        # Should contain moved files
        expected_files = {
            '__init__.py',
            'data_loading.py',
            'model_loading.py'
        }
        
        actual_files = {f.name for f in data_mgmt_dir.iterdir() if f.is_file() and not f.name.startswith('.')}
        
        for expected_file in expected_files:
            assert expected_file in actual_files, f"data_management directory should contain {expected_file}"
    
    def test_processing_directory_structure(self):
        """Test that processing directory contains moved files."""
        processing_dir = Path(__file__).parent.parent.parent / 'src' / 'eq' / 'processing'
        
        # Should contain moved files
        expected_files = {
            '__init__.py',
            'preprocessing.py'
        }
        
        actual_files = {f.name for f in processing_dir.iterdir() if f.is_file() and not f.name.startswith('.')}
        
        for expected_file in expected_files:
            assert expected_file in actual_files, f"processing directory should contain {expected_file}"
    
    def test_data_loading_imports_work(self):
        """Test that data_loading functions can be imported from new location."""
        try:
            from eq.data_management.data_loading import get_glom_mask_file, get_glom_y, n_glom_codes
            assert callable(get_glom_mask_file), "get_glom_mask_file should be callable"
            assert callable(get_glom_y), "get_glom_y should be callable"
            assert callable(n_glom_codes), "n_glom_codes should be callable"
        except ImportError as e:
            pytest.fail(f"Failed to import data_loading functions: {e}")
    
    def test_preprocessing_imports_work(self):
        """Test that preprocessing functions can be imported from new location."""
        try:
            from eq.processing.preprocessing import preprocess_image_for_model, normalize_image_array
            assert callable(preprocess_image_for_model), "preprocess_image_for_model should be callable"
            assert callable(normalize_image_array), "normalize_image_array should be callable"
        except ImportError as e:
            pytest.fail(f"Failed to import preprocessing functions: {e}")
    
    def test_model_loading_imports_work(self):
        """Test that model_loading functions can be imported from new location."""
        try:
            from eq.data_management.model_loading import load_model_with_historical_support, get_model_info
            assert callable(load_model_with_historical_support), "load_model_with_historical_support should be callable"
            assert callable(get_model_info), "get_model_info should be callable"
        except ImportError as e:
            pytest.fail(f"Failed to import model_loading functions: {e}")
    
    def test_core_constants_imports_work(self):
        """Test that core constants can still be imported."""
        try:
            from eq.core.constants import BINARY_P2C, DEFAULT_MASK_THRESHOLD
            assert BINARY_P2C == [0, 1], "BINARY_P2C should be [0, 1]"
            assert DEFAULT_MASK_THRESHOLD == 127, "DEFAULT_MASK_THRESHOLD should be 127"
        except ImportError as e:
            pytest.fail(f"Failed to import core constants: {e}")
    
    def test_core_types_imports_work(self):
        """Test that core types can be imported from new location."""
        try:
            from eq.core.types import DataLoaderInterface, ModelLoaderInterface, PreprocessorInterface
            # These should be abstract base classes or type hints
            assert DataLoaderInterface is not None, "DataLoaderInterface should be defined"
            assert ModelLoaderInterface is not None, "ModelLoaderInterface should be defined"
            assert PreprocessorInterface is not None, "PreprocessorInterface should be defined"
        except ImportError as e:
            pytest.fail(f"Failed to import core types: {e}")
    
    def test_no_import_errors_in_main_modules(self):
        """Test that main modules can import without errors after reorganization."""
        try:
            # Test that main package can be imported
            import eq
            assert eq is not None, "eq package should be importable"
            
            # Test that core can be imported
            from eq import core
            assert core is not None, "eq.core should be importable"
            
        except ImportError as e:
            pytest.fail(f"Failed to import main modules: {e}")
    
    def test_moved_functions_work_correctly(self):
        """Test that moved functions work correctly in their new locations."""
        try:
            from eq.data_management.data_loading import get_glom_mask_file
            from eq.processing.preprocessing import preprocess_image_for_model
            from eq.data_management.model_loading import load_model_with_historical_support
            
            # These should not raise exceptions (actual functionality tested elsewhere)
            assert callable(get_glom_mask_file), "get_glom_mask_file should be callable"
            assert callable(preprocess_image_for_model), "preprocess_image_for_model should be callable"
            assert callable(load_model_with_historical_support), "load_model_with_historical_support should be callable"
            
        except Exception as e:
            pytest.fail(f"Moved functions should work correctly: {e}")
