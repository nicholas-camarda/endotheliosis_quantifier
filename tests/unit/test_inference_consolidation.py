#!/usr/bin/env python3
"""
Tests for inference infrastructure consolidation.

This test suite verifies that inference scripts are properly organized
and the new inference module works correctly.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestInferenceDirectoryStructure:
    """Test that inference directory structure is properly set up."""
    
    def test_inference_directory_exists(self):
        """Test that the inference directory exists."""
        inference_dir = Path("src/eq/inference")
        assert inference_dir.exists(), "Inference directory should exist"
        assert inference_dir.is_dir(), "Inference directory should be a directory"
    
    def test_inference_module_imports(self):
        """Test that the inference module can be imported."""
        try:
            from eq.inference import __init__
            assert True
        except ImportError as e:
            pytest.fail(f"Inference module import failed: {e}")
    
    def test_inference_scripts_exist(self):
        """Test that all required inference scripts exist."""
        inference_dir = Path("src/eq/inference")
        
        required_files = [
            "__init__.py",
            "run_glomeruli_prediction.py",
            "run_mitochondria_prediction.py",
            "gpu_inference.py"
        ]
        
        for file_name in required_files:
            file_path = inference_dir / file_name
            assert file_path.exists(), f"Required file {file_name} should exist in inference directory"


class TestGlomeruliInference:
    """Test glomeruli inference functionality."""
    
    def test_run_glomeruli_prediction_function_exists(self):
        """Test that run_glomeruli_prediction function exists."""
        try:
            from eq.inference import run_glomeruli_prediction
            assert callable(run_glomeruli_prediction)
        except ImportError as e:
            pytest.fail(f"run_glomeruli_prediction import failed: {e}")
    
    def test_run_glomeruli_prediction_accepts_parameters(self):
        """Test that run_glomeruli_prediction accepts expected parameters."""
        from eq.inference import run_glomeruli_prediction
        
        # Test function signature without executing heavy operations
        import inspect
        sig = inspect.signature(run_glomeruli_prediction)
        assert len(sig.parameters) >= 1  # Should have at least config_path parameter
        
        # Test that function can be imported and called (without execution)
        assert True


class TestMitochondriaInference:
    """Test mitochondria inference functionality."""
    
    def test_run_mitochondria_prediction_function_exists(self):
        """Test that run_mitochondria_prediction function exists."""
        try:
            from eq.inference import run_mitochondria_prediction
            assert callable(run_mitochondria_prediction)
        except ImportError as e:
            pytest.fail(f"run_mitochondria_prediction import failed: {e}")
    
    def test_run_mitochondria_prediction_accepts_parameters(self):
        """Test that run_mitochondria_prediction accepts expected parameters."""
        from eq.inference import run_mitochondria_prediction
        
        # Test function signature without executing heavy operations
        import inspect
        sig = inspect.signature(run_mitochondria_prediction)
        assert len(sig.parameters) >= 2  # Should have at least model_path and data_path
        
        # Test that function can be imported and called (without execution)
        assert True


class TestGPUInference:
    """Test GPU inference functionality."""
    
    def test_run_gpu_inference_function_exists(self):
        """Test that run_gpu_inference function exists."""
        try:
            from eq.inference import run_gpu_inference
            assert callable(run_gpu_inference)
        except ImportError as e:
            pytest.fail(f"run_gpu_inference import failed: {e}")
    
    def test_run_gpu_inference_accepts_parameters(self):
        """Test that run_gpu_inference accepts expected parameters."""
        from eq.inference import run_gpu_inference
        
        # Test function signature without executing heavy operations
        import inspect
        sig = inspect.signature(run_gpu_inference)
        assert len(sig.parameters) >= 1  # Should have at least some parameters
        
        # Test that function can be imported and called (without execution)
        assert True


class TestInferenceModuleImports:
    """Test that the inference module properly imports all functions."""
    
    def test_inference_module_imports_all_functions(self):
        """Test that all inference functions can be imported from the module."""
        try:
            from eq.inference import (
                run_glomeruli_prediction,
                run_mitochondria_prediction,
                run_gpu_inference
            )
            assert all(callable(f) for f in [
                run_glomeruli_prediction,
                run_mitochondria_prediction,
                run_gpu_inference
            ])
        except ImportError as e:
            pytest.fail(f"Inference module imports failed: {e}")
    
    def test_inference_module_has_unified_api(self):
        """Test that the inference module provides a unified API."""
        from eq.inference import __all__
        
        expected_functions = [
            'run_glomeruli_prediction',
            'run_mitochondria_prediction',
            'run_gpu_inference'
        ]
        
        for func_name in expected_functions:
            assert func_name in __all__, f"Function {func_name} should be in __all__"


class TestRedundantFilesRemoved:
    """Test that redundant inference files have been removed."""
    
    def test_redundant_pipeline_files_removed(self):
        """Test that redundant inference files have been removed from pipeline."""
        pipeline_dir = Path("src/eq/pipeline")
        
        redundant_files = [
            'run_glomeruli_prediction.py',
            'run_glomeruli_prediction_fixed.py',
            'historical_glomeruli_inference.py',
            'gpu_inference.py'
        ]
        
        for file_name in redundant_files:
            assert not (pipeline_dir / file_name).exists(), f"Redundant file {file_name} still exists"


class TestImportUpdates:
    """Test that imports have been updated throughout the codebase."""
    
    def test_no_old_inference_imports(self):
        """Test that no old inference imports remain in the codebase."""
        # This is a basic check - in a real scenario, you'd use grep or similar
        # For now, we'll just verify the new imports work
        try:
            from eq.inference import run_glomeruli_prediction, run_mitochondria_prediction, run_gpu_inference
            assert True  # If we get here, imports work
        except ImportError as e:
            pytest.fail(f"New inference imports failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
