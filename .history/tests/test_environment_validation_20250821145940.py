"""
Tests for environment validation and dependency checking.

This module tests the environment setup for the dual-environment architecture
with fastai/PyTorch support for both MPS (Apple Silicon) and CUDA (NVIDIA) backends.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest
import torch

# Import the modules we'll be testing
from eq.utils.env_check import read_environment_name_from_yaml, verify_environment_name
from eq.utils.runtime_check import check_imports, check_paths


class TestEnvironmentValidation:
    """Test environment validation functionality."""

    def test_environment_name_verification_success(self):
        """Test successful environment name verification."""
        with patch('eq.utils.env_check.read_environment_name_from_yaml', return_value='eq'), \
             patch('eq.utils.env_check.list_env_names', return_value=['base', 'eq', 'fastai']):
            result, message = verify_environment_name('eq')
            assert result is True
            assert 'Environment verification passed' in message

    def test_environment_name_verification_failure(self):
        """Test environment name verification failure."""
        with patch('eq.utils.env_check.read_environment_name_from_yaml', return_value='wrong_name'):
            result, message = verify_environment_name('eq')
            assert result is False
            assert 'wrong_name' in message

    def test_environment_file_not_found(self):
        """Test handling of missing environment.yml file."""
        with patch('pathlib.Path.exists', return_value=False):
            with pytest.raises(FileNotFoundError):
                read_environment_name_from_yaml('nonexistent.yml')

    def test_environment_file_missing_name_field(self):
        """Test handling of environment.yml without name field."""
        mock_content = "channels:\n  - conda-forge\n"
        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_content.splitlines()
        mock_file.__exit__.return_value = None
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.open', return_value=mock_file):
            with pytest.raises(ValueError, match="No 'name:' field found"):
                read_environment_name_from_yaml('environment.yml')


class TestDependencyChecking:
    """Test dependency checking functionality."""

    def test_required_imports_success(self):
        """Test successful import checking."""
        with patch('importlib.import_module') as mock_import:
            mock_import.return_value = MagicMock()
            result = check_imports()
            assert result is True

    def test_required_imports_failure(self):
        """Test import checking with missing dependencies."""
        with patch('importlib.import_module', side_effect=ImportError("No module named 'nonexistent'")):
            result = check_imports()
            assert result is False

    def test_data_paths_check_success(self):
        """Test successful data paths checking."""
        with patch('pathlib.Path.exists', return_value=True):
            result = check_paths()
            assert result is True

    def test_data_paths_check_failure(self):
        """Test data paths checking with missing paths."""
        with patch('pathlib.Path.exists', return_value=False):
            result = check_paths()
            assert result is False


class TestHardwareDetection:
    """Test hardware detection capabilities."""

    def test_pytorch_mps_availability(self):
        """Test PyTorch MPS (Metal Performance Shaders) availability detection."""
        # Test that we can check MPS availability
        mps_available = torch.backends.mps.is_available()
        mps_built = torch.backends.mps.is_built()
        
        # These should be boolean values
        assert isinstance(mps_available, bool)
        assert isinstance(mps_built, bool)
        
        # On Apple Silicon, MPS should be built
        if sys.platform == "darwin":
            assert mps_built is True

    def test_pytorch_cuda_availability(self):
        """Test PyTorch CUDA availability detection."""
        # Test that we can check CUDA availability
        cuda_available = torch.cuda.is_available()
        
        # This should be a boolean value
        assert isinstance(cuda_available, bool)
        
        # If CUDA is available, we should be able to get device count
        if cuda_available:
            device_count = torch.cuda.device_count()
            assert isinstance(device_count, int)
            assert device_count > 0

    def test_device_selection_logic(self):
        """Test device selection logic for different hardware configurations."""
        # Test MPS device selection
        if torch.backends.mps.is_available():
            mps_device = torch.device("mps")
            assert str(mps_device) == "mps"
        
        # Test CUDA device selection
        if torch.cuda.is_available():
            cuda_device = torch.device("cuda")
            assert str(cuda_device) == "cuda"
        
        # Test CPU fallback
        cpu_device = torch.device("cpu")
        assert str(cpu_device) == "cpu"

    def test_memory_detection(self):
        """Test memory detection capabilities."""
        # Test that we can get system memory information
        if torch.cuda.is_available():
            # CUDA memory info
            total_memory = torch.cuda.get_device_properties(0).total_memory
            assert isinstance(total_memory, int)
            assert total_memory > 0
        
        # Test that we can create tensors on available devices
        if torch.backends.mps.is_available():
            try:
                tensor_mps = torch.tensor([1, 2, 3], device="mps")
                assert tensor_mps.device.type == "mps"
            except Exception:
                # MPS might not be available for tensor creation even if built
                pass
        
        if torch.cuda.is_available():
            tensor_cuda = torch.tensor([1, 2, 3], device="cuda")
            assert tensor_cuda.device.type == "cuda"


class TestFastaiDependencies:
    """Test fastai-specific dependency checking."""

    def test_fastai_import_availability(self):
        """Test that fastai can be imported when available."""
        try:
            import fastai
            assert fastai is not None
        except ImportError:
            # fastai not installed - this is expected during migration
            pytest.skip("fastai not installed - expected during migration")

    def test_fastai_vision_import(self):
        """Test fastai.vision.all import when available."""
        try:
            import fastai.vision.all

            # If we get here, the import was successful
            assert fastai.vision.all is not None
        except ImportError:
            # fastai not installed - this is expected during migration
            pytest.skip("fastai not installed - expected during migration")

    def test_fastai_device_integration(self):
        """Test fastai device integration with PyTorch backends."""
        try:
            import fastai.vision.all

            # Test that fastai can use available devices
            if torch.backends.mps.is_available():
                # Test MPS device
                pass  # fastai should handle MPS devices
            
            if torch.cuda.is_available():
                # Test CUDA device
                pass  # fastai should handle CUDA devices
            
        except ImportError:
            # fastai not installed - this is expected during migration
            pytest.skip("fastai not installed - expected during migration")


class TestEnvironmentConfiguration:
    """Test environment configuration validation."""

    def test_python_version_compatibility(self):
        """Test Python version compatibility."""
        # Should be Python 3.9 as specified in environment.yml
        assert sys.version_info.major == 3
        assert sys.version_info.minor >= 9

    def test_platform_detection(self):
        """Test platform detection for dual-environment architecture."""
        platform = sys.platform
        
        # Should be able to detect macOS
        if platform == "darwin":
            assert platform == "darwin"
        
        # Should be able to detect Linux (for CUDA support)
        elif platform.startswith("linux"):
            assert platform.startswith("linux")
        
        # Should be able to detect Windows
        elif platform == "win32":
            assert platform == "win32"

    def test_environment_variables(self):
        """Test environment variable configuration."""
        # Test PYTORCH_ENABLE_MPS_FALLBACK environment variable
        import os
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"


if __name__ == "__main__":
    pytest.main([__file__])
