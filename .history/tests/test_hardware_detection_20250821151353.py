"""
Tests for hardware detection utilities.

This module tests the hardware detection and capability reporting functionality
for the dual-environment architecture with MPS/CUDA support.
"""

import pytest
import platform
from unittest.mock import patch, MagicMock
import torch

from eq.utils.hardware_detection import (
    HardwareDetector, HardwareCapabilities, BackendType, HardwareTier,
    get_hardware_capabilities, get_device_recommendation, get_capability_report,
    get_optimal_batch_size
)


class TestHardwareDetector:
    """Test the HardwareDetector class."""

    def test_initialization(self):
        """Test HardwareDetector initialization."""
        detector = HardwareDetector()
        assert detector._capabilities is None

    @patch('torch.backends.mps.is_available', return_value=True)
    @patch('torch.backends.mps.is_built', return_value=True)
    @patch('torch.cuda.is_available', return_value=False)
    @patch('torch.cuda.device_count', return_value=0)
    @patch('psutil.cpu_count', return_value=8)
    @patch('psutil.virtual_memory')
    @patch('platform.system', return_value='Darwin')
    @patch('platform.machine', return_value='arm64')
    def test_detect_capabilities_mps(self, mock_machine, mock_system, mock_memory, 
                                   mock_cpu_count, mock_cuda_count, mock_cuda_available,
                                   mock_mps_built, mock_mps_available):
        """Test capability detection with MPS available."""
        # Mock memory info
        mock_memory.return_value.total = 16 * 1024**3  # 16 GB
        mock_memory.return_value.available = 8 * 1024**3  # 8 GB
        
        detector = HardwareDetector()
        capabilities = detector.detect_capabilities()
        
        assert capabilities.platform == 'Darwin'
        assert capabilities.architecture == 'arm64'
        assert capabilities.cpu_count == 8
        assert capabilities.total_memory_gb == 16.0
        assert capabilities.available_memory_gb == 8.0
        assert capabilities.backend_type == BackendType.MPS
        assert capabilities.mps_available is True
        assert capabilities.mps_built is True
        assert capabilities.cuda_available is False
        assert capabilities.cuda_device_count == 0
        assert capabilities.hardware_tier == HardwareTier.POWERFUL

    @patch('torch.backends.mps.is_available', return_value=False)
    @patch('torch.backends.mps.is_built', return_value=False)
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=1)
    @patch('torch.cuda.get_device_name', return_value='NVIDIA RTX 3080')
    @patch('torch.cuda.get_device_properties')
    @patch('psutil.cpu_count', return_value=12)
    @patch('psutil.virtual_memory')
    @patch('platform.system', return_value='Linux')
    @patch('platform.machine', return_value='x86_64')
    def test_detect_capabilities_cuda(self, mock_machine, mock_system, mock_memory,
                                    mock_cpu_count, mock_properties, mock_device_name, 
                                    mock_cuda_count, mock_cuda_available, mock_mps_built, 
                                    mock_mps_available):
        """Test capability detection with CUDA available."""
        # Mock memory info
        mock_memory.return_value.total = 32 * 1024**3  # 32 GB
        mock_memory.return_value.available = 16 * 1024**3  # 16 GB
        
        # Mock GPU properties
        mock_prop = MagicMock()
        mock_prop.total_memory = 10 * 1024**3  # 10 GB
        mock_properties.return_value = mock_prop
        
        detector = HardwareDetector()
        capabilities = detector.detect_capabilities()
        
        assert capabilities.platform == 'Linux'
        assert capabilities.architecture == 'x86_64'
        assert capabilities.cpu_count == 12
        assert capabilities.total_memory_gb == 32.0
        assert capabilities.available_memory_gb == 16.0
        assert capabilities.backend_type == BackendType.CUDA
        assert capabilities.gpu_name == 'NVIDIA RTX 3080'
        assert capabilities.gpu_memory_gb == 10.0
        assert capabilities.mps_available is False
        assert capabilities.mps_built is False
        assert capabilities.cuda_available is True
        assert capabilities.cuda_device_count == 1
        assert capabilities.hardware_tier == HardwareTier.POWERFUL

    @patch('torch.backends.mps.is_available', return_value=False)
    @patch('torch.backends.mps.is_built', return_value=False)
    @patch('torch.cuda.is_available', return_value=False)
    @patch('psutil.cpu_count', return_value=4)
    @patch('psutil.virtual_memory')
    @patch('platform.system', return_value='Windows')
    @patch('platform.machine', return_value='AMD64')
    def test_detect_capabilities_cpu_only(self, mock_machine, mock_system, mock_memory,
                                        mock_cpu_count, mock_cuda_available, mock_mps_built, 
                                        mock_mps_available):
        """Test capability detection with CPU only."""
        # Mock memory info
        mock_memory.return_value.total = 8 * 1024**3  # 8 GB
        mock_memory.return_value.available = 4 * 1024**3  # 4 GB
        
        detector = HardwareDetector()
        capabilities = detector.detect_capabilities()
        
        assert capabilities.platform == 'Windows'
        assert capabilities.architecture == 'AMD64'
        assert capabilities.cpu_count == 4
        assert capabilities.total_memory_gb == 8.0
        assert capabilities.available_memory_gb == 4.0
        assert capabilities.backend_type == BackendType.CPU
        assert capabilities.gpu_name is None
        assert capabilities.gpu_memory_gb is None
        assert capabilities.mps_available is False
        assert capabilities.mps_built is False
        assert capabilities.cuda_available is False
        assert capabilities.cuda_device_count == 0
        assert capabilities.hardware_tier == HardwareTier.BASIC

    def test_determine_primary_backend(self):
        """Test primary backend determination logic."""
        detector = HardwareDetector()
        
        # CUDA should be preferred over MPS
        backend = detector._determine_primary_backend(True, True)
        assert backend == BackendType.CUDA
        
        # MPS should be used when CUDA is not available
        backend = detector._determine_primary_backend(True, False)
        assert backend == BackendType.MPS
        
        # CPU should be used when neither is available
        backend = detector._determine_primary_backend(False, False)
        assert backend == BackendType.CPU

    @patch('platform.system', return_value='Darwin')
    @patch('platform.processor', return_value='Apple M1')
    def test_infer_apple_gpu_name_m1(self, mock_processor, mock_system):
        """Test Apple GPU name inference for M1."""
        detector = HardwareDetector()
        gpu_name = detector._infer_apple_gpu_name()
        assert gpu_name == 'Apple M1 GPU'

    @patch('platform.system', return_value='Darwin')
    @patch('platform.processor', return_value='Apple M2')
    def test_infer_apple_gpu_name_m2(self, mock_processor, mock_system):
        """Test Apple GPU name inference for M2."""
        detector = HardwareDetector()
        gpu_name = detector._infer_apple_gpu_name()
        assert gpu_name == 'Apple M2 GPU'

    @patch('platform.system', return_value='Darwin')
    @patch('platform.processor', return_value='Apple M3')
    def test_infer_apple_gpu_name_m3(self, mock_processor, mock_system):
        """Test Apple GPU name inference for M3."""
        detector = HardwareDetector()
        gpu_name = detector._infer_apple_gpu_name()
        assert gpu_name == 'Apple M3 GPU'

    @patch('psutil.virtual_memory')
    def test_estimate_apple_gpu_memory(self, mock_memory):
        """Test Apple GPU memory estimation."""
        mock_memory.return_value.total = 16 * 1024**3  # 16 GB
        
        detector = HardwareDetector()
        gpu_memory = detector._estimate_apple_gpu_memory()
        assert gpu_memory == 4.8  # 30% of 16 GB

    def test_classify_hardware_tier(self):
        """Test hardware tier classification."""
        detector = HardwareDetector()
        
        # Basic tier: CPU-only
        tier = detector._classify_hardware_tier(4, 8.0, BackendType.CPU, None)
        assert tier == HardwareTier.BASIC
        
        # Basic tier: Low memory
        tier = detector._classify_hardware_tier(8, 4.0, BackendType.MPS, None)
        assert tier == HardwareTier.BASIC
        
        # Powerful tier: High-end CUDA GPU
        tier = detector._classify_hardware_tier(8, 16.0, BackendType.CUDA, 12.0)
        assert tier == HardwareTier.POWERFUL
        
        # Powerful tier: High-end MPS with lots of memory
        tier = detector._classify_hardware_tier(8, 32.0, BackendType.MPS, None)
        assert tier == HardwareTier.POWERFUL
        
        # Powerful tier: High-end CPU system
        tier = detector._classify_hardware_tier(16, 32.0, BackendType.CPU, None)
        assert tier == HardwareTier.POWERFUL
        
        # Standard tier: Everything else
        tier = detector._classify_hardware_tier(8, 16.0, BackendType.MPS, None)
        assert tier == HardwareTier.STANDARD

    @patch('torch.backends.mps.is_available', return_value=True)
    @patch('torch.backends.mps.is_built', return_value=True)
    @patch('torch.cuda.is_available', return_value=False)
    @patch('psutil.cpu_count', return_value=8)
    @patch('psutil.virtual_memory')
    @patch('platform.system', return_value='Darwin')
    @patch('platform.machine', return_value='arm64')
    def test_get_device_recommendation_development(self, mock_machine, mock_system, mock_memory,
                                                 mock_cpu_count, mock_cuda_available,
                                                 mock_mps_built, mock_mps_available):
        """Test device recommendation for development mode."""
        mock_memory.return_value.total = 16 * 1024**3
        mock_memory.return_value.available = 8 * 1024**3
        
        detector = HardwareDetector()
        backend, explanation = detector.get_device_recommendation("development")
        
        assert backend == BackendType.MPS
        assert "Development mode" in explanation
        assert "MPS" in explanation

    @patch('torch.backends.mps.is_available', return_value=False)
    @patch('torch.backends.mps.is_built', return_value=False)
    @patch('torch.cuda.is_available', return_value=True)
    @patch('psutil.cpu_count', return_value=8)
    @patch('psutil.virtual_memory')
    @patch('platform.system', return_value='Linux')
    @patch('platform.machine', return_value='x86_64')
    def test_get_device_recommendation_production(self, mock_machine, mock_system, mock_memory,
                                                mock_cpu_count, mock_cuda_available,
                                                mock_mps_built, mock_mps_available):
        """Test device recommendation for production mode."""
        mock_memory.return_value.total = 16 * 1024**3
        mock_memory.return_value.available = 8 * 1024**3
        
        detector = HardwareDetector()
        backend, explanation = detector.get_device_recommendation("production")
        
        assert backend == BackendType.CUDA
        assert "Production mode" in explanation
        assert "CUDA" in explanation

    @patch('torch.backends.mps.is_available', return_value=True)
    @patch('torch.backends.mps.is_built', return_value=True)
    @patch('torch.cuda.is_available', return_value=False)
    @patch('psutil.cpu_count', return_value=8)
    @patch('psutil.virtual_memory')
    @patch('platform.system', return_value='Darwin')
    @patch('platform.machine', return_value='arm64')
    def test_get_device_recommendation_auto(self, mock_machine, mock_system, mock_memory,
                                          mock_cpu_count, mock_cuda_available,
                                          mock_mps_built, mock_mps_available):
        """Test device recommendation for auto mode."""
        mock_memory.return_value.total = 16 * 1024**3
        mock_memory.return_value.available = 8 * 1024**3
        
        detector = HardwareDetector()
        backend, explanation = detector.get_device_recommendation("auto")
        
        assert backend == BackendType.MPS
        assert "Auto mode" in explanation
        assert "MPS" in explanation

    @patch('torch.backends.mps.is_available', return_value=True)
    @patch('torch.backends.mps.is_built', return_value=True)
    @patch('torch.cuda.is_available', return_value=False)
    @patch('psutil.cpu_count', return_value=8)
    @patch('psutil.virtual_memory')
    @patch('platform.system', return_value='Darwin')
    @patch('platform.machine', return_value='arm64')
    def test_get_capability_report(self, mock_machine, mock_system, mock_memory,
                                 mock_cpu_count, mock_cuda_available,
                                 mock_mps_built, mock_mps_available):
        """Test capability report generation."""
        mock_memory.return_value.total = 16 * 1024**3
        mock_memory.return_value.available = 8 * 1024**3
        
        detector = HardwareDetector()
        report = detector.get_capability_report()
        
        assert "Hardware Capability Report" in report
        assert "Platform: Darwin" in report
        assert "Architecture: arm64" in report
        assert "CPU Cores: 8" in report
        assert "MPS Available: True" in report
        assert "CUDA Available: False" in report
        assert "Development Mode" in report
        assert "Production Mode" in report

    @patch('torch.backends.mps.is_available', return_value=True)
    @patch('torch.backends.mps.is_built', return_value=True)
    @patch('torch.cuda.is_available', return_value=False)
    @patch('psutil.cpu_count', return_value=8)
    @patch('psutil.virtual_memory')
    @patch('platform.system', return_value='Darwin')
    @patch('platform.machine', return_value='arm64')
    def test_get_optimal_batch_size_mps(self, mock_machine, mock_system, mock_memory,
                                      mock_cpu_count, mock_cuda_available,
                                      mock_mps_built, mock_mps_available):
        """Test optimal batch size calculation for MPS."""
        mock_memory.return_value.total = 16 * 1024**3
        mock_memory.return_value.available = 8 * 1024**3
        
        detector = HardwareDetector()
        batch_size = detector.get_optimal_batch_size("production")
        
        assert batch_size == 8  # 16GB total memory should give batch size 8

    @patch('torch.backends.mps.is_available', return_value=False)
    @patch('torch.backends.mps.is_built', return_value=False)
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.get_device_properties')
    @patch('psutil.cpu_count', return_value=8)
    @patch('psutil.virtual_memory')
    @patch('platform.system', return_value='Linux')
    @patch('platform.machine', return_value='x86_64')
    def test_get_optimal_batch_size_cuda(self, mock_machine, mock_system, mock_memory,
                                       mock_cpu_count, mock_properties, mock_cuda_available,
                                       mock_mps_built, mock_mps_available):
        """Test optimal batch size calculation for CUDA."""
        mock_memory.return_value.total = 16 * 1024**3
        mock_memory.return_value.available = 8 * 1024**3
        mock_properties.return_value.total_memory = 12 * 1024**3  # 12 GB GPU
        
        detector = HardwareDetector()
        batch_size = detector.get_optimal_batch_size("production")
        
        assert batch_size == 16  # 12GB GPU memory should give batch size 16

    @patch('torch.backends.mps.is_available', return_value=False)
    @patch('torch.backends.mps.is_built', return_value=False)
    @patch('torch.cuda.is_available', return_value=False)
    @patch('psutil.cpu_count', return_value=4)
    @patch('psutil.virtual_memory')
    @patch('platform.system', return_value='Windows')
    @patch('platform.machine', return_value='AMD64')
    def test_get_optimal_batch_size_cpu(self, mock_machine, mock_system, mock_memory,
                                      mock_cpu_count, mock_cuda_available,
                                      mock_mps_built, mock_mps_available):
        """Test optimal batch size calculation for CPU."""
        mock_memory.return_value.total = 8 * 1024**3
        mock_memory.return_value.available = 4 * 1024**3
        
        detector = HardwareDetector()
        batch_size = detector.get_optimal_batch_size("production")
        
        assert batch_size == 2  # CPU should give batch size 2


class TestGlobalFunctions:
    """Test the global convenience functions."""

    @patch('eq.utils.hardware_detection.hardware_detector')
    def test_get_hardware_capabilities(self, mock_detector):
        """Test get_hardware_capabilities function."""
        mock_capabilities = MagicMock()
        mock_detector.detect_capabilities.return_value = mock_capabilities
        
        result = get_hardware_capabilities()
        assert result == mock_capabilities
        mock_detector.detect_capabilities.assert_called_once()

    @patch('eq.utils.hardware_detection.hardware_detector')
    def test_get_device_recommendation(self, mock_detector):
        """Test get_device_recommendation function."""
        mock_detector.get_device_recommendation.return_value = (BackendType.MPS, "test")
        
        result = get_device_recommendation("development")
        assert result == (BackendType.MPS, "test")
        mock_detector.get_device_recommendation.assert_called_once_with("development")

    @patch('eq.utils.hardware_detection.hardware_detector')
    def test_get_capability_report(self, mock_detector):
        """Test get_capability_report function."""
        mock_detector.get_capability_report.return_value = "test report"
        
        result = get_capability_report()
        assert result == "test report"
        mock_detector.get_capability_report.assert_called_once()

    @patch('eq.utils.hardware_detection.hardware_detector')
    def test_get_optimal_batch_size(self, mock_detector):
        """Test get_optimal_batch_size function."""
        mock_detector.get_optimal_batch_size.return_value = 16
        
        result = get_optimal_batch_size("production")
        assert result == 16
        mock_detector.get_optimal_batch_size.assert_called_once_with("production")


class TestIntegration:
    """Integration tests for hardware detection."""

    def test_real_hardware_detection(self):
        """Test hardware detection with real system (if possible)."""
        # This test will run on the actual system
        capabilities = get_hardware_capabilities()
        
        # Basic assertions that should work on any system
        assert capabilities.platform in ['Darwin', 'Linux', 'Windows']
        assert capabilities.cpu_count > 0
        assert capabilities.total_memory_gb > 0
        assert capabilities.available_memory_gb > 0
        assert capabilities.hardware_tier in [HardwareTier.BASIC, HardwareTier.STANDARD, HardwareTier.POWERFUL]
        
        # Backend availability should be consistent
        if capabilities.mps_available:
            assert capabilities.mps_built
        if capabilities.cuda_available:
            assert capabilities.cuda_device_count > 0

    def test_device_recommendation_consistency(self):
        """Test that device recommendations are consistent."""
        capabilities = get_hardware_capabilities()
        
        # Get recommendations for all modes
        dev_backend, dev_explanation = get_device_recommendation("development")
        prod_backend, prod_explanation = get_device_recommendation("production")
        auto_backend, auto_explanation = get_device_recommendation("auto")
        
        # All recommendations should be valid backend types
        assert dev_backend in [BackendType.MPS, BackendType.CUDA, BackendType.CPU]
        assert prod_backend in [BackendType.MPS, BackendType.CUDA, BackendType.CPU]
        assert auto_backend in [BackendType.MPS, BackendType.CUDA, BackendType.CPU]
        
        # Explanations should be non-empty
        assert len(dev_explanation) > 0
        assert len(prod_explanation) > 0
        assert len(auto_explanation) > 0

    def test_batch_size_consistency(self):
        """Test that batch size recommendations are consistent."""
        # Batch sizes should be positive integers
        dev_batch_size = get_optimal_batch_size("development")
        prod_batch_size = get_optimal_batch_size("production")
        auto_batch_size = get_optimal_batch_size("auto")
        
        assert dev_batch_size > 0
        assert prod_batch_size > 0
        assert auto_batch_size > 0
        assert isinstance(dev_batch_size, int)
        assert isinstance(prod_batch_size, int)
        assert isinstance(auto_batch_size, int)


if __name__ == "__main__":
    pytest.main([__file__])
