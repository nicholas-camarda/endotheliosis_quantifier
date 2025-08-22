"""
Hardware detection utilities for dual-environment architecture.

This module provides comprehensive hardware detection and capability reporting
for the dual-environment architecture supporting both MPS (Apple Silicon) and
CUDA (NVIDIA) backends.
"""

import platform
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import psutil
import torch


class BackendType(Enum):
    """Enumeration of available backend types."""
    MPS = "mps"
    CUDA = "cuda"
    CPU = "cpu"


class HardwareTier(Enum):
    """Enumeration of hardware capability tiers."""
    BASIC = "basic"
    STANDARD = "standard"
    POWERFUL = "powerful"


@dataclass
class HardwareCapabilities:
    """Data class for hardware capability information."""
    platform: str
    architecture: str
    cpu_count: int
    total_memory_gb: float
    available_memory_gb: float
    backend_type: Optional[BackendType]
    gpu_name: Optional[str]
    gpu_memory_gb: Optional[float]
    hardware_tier: HardwareTier
    mps_available: bool
    mps_built: bool
    cuda_available: bool
    cuda_device_count: int


class HardwareDetector:
    """Hardware detection and capability assessment utility."""
    
    def __init__(self):
        """Initialize the hardware detector."""
        self._capabilities: Optional[HardwareCapabilities] = None
    
    def detect_capabilities(self) -> HardwareCapabilities:
        """
        Detect and assess hardware capabilities.
        
        Returns:
            HardwareCapabilities: Comprehensive hardware capability information.
        """
        if self._capabilities is None:
            self._capabilities = self._perform_detection()
        return self._capabilities
    
    def _perform_detection(self) -> HardwareCapabilities:
        """Perform the actual hardware detection."""
        # Basic system information
        platform_name = platform.system()
        architecture = platform.machine()
        cpu_count = psutil.cpu_count()
        
        # Memory information
        memory = psutil.virtual_memory()
        total_memory_gb = memory.total / (1024**3)
        available_memory_gb = memory.available / (1024**3)
        
        # PyTorch backend detection
        mps_available = torch.backends.mps.is_available()
        mps_built = torch.backends.mps.is_built()
        cuda_available = torch.cuda.is_available()
        cuda_device_count = torch.cuda.device_count() if cuda_available else 0
        
        # Determine primary backend
        backend_type = self._determine_primary_backend(mps_available, cuda_available)
        
        # GPU information
        gpu_name = None
        gpu_memory_gb = None
        
        if backend_type == BackendType.CUDA and cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        elif backend_type == BackendType.MPS and mps_available:
            # For MPS, we can't easily get GPU name/memory, but we can infer from system
            gpu_name = self._infer_apple_gpu_name()
            gpu_memory_gb = self._estimate_apple_gpu_memory()
        
        # Determine hardware tier
        hardware_tier = self._classify_hardware_tier(
            cpu_count, total_memory_gb, backend_type, gpu_memory_gb
        )
        
        return HardwareCapabilities(
            platform=platform_name,
            architecture=architecture,
            cpu_count=cpu_count,
            total_memory_gb=total_memory_gb,
            available_memory_gb=available_memory_gb,
            backend_type=backend_type,
            gpu_name=gpu_name,
            gpu_memory_gb=gpu_memory_gb,
            hardware_tier=hardware_tier,
            mps_available=mps_available,
            mps_built=mps_built,
            cuda_available=cuda_available,
            cuda_device_count=cuda_device_count
        )
    
    def _determine_primary_backend(self, mps_available: bool, cuda_available: bool) -> Optional[BackendType]:
        """Determine the primary backend based on availability."""
        if cuda_available:
            return BackendType.CUDA
        elif mps_available:
            return BackendType.MPS
        else:
            return BackendType.CPU
    
    def _infer_apple_gpu_name(self) -> str:
        """Infer Apple GPU name from system information."""
        # This is a simplified approach - in practice, you might want to use
        # more sophisticated detection methods
        if platform.system() == "Darwin":
            if "Apple M1" in platform.processor():
                return "Apple M1 GPU"
            elif "Apple M2" in platform.processor():
                return "Apple M2 GPU"
            elif "Apple M3" in platform.processor():
                return "Apple M3 GPU"
            else:
                return "Apple Silicon GPU"
        return "Unknown Apple GPU"
    
    def _estimate_apple_gpu_memory(self) -> Optional[float]:
        """Estimate Apple GPU memory (unified memory architecture)."""
        # Apple Silicon uses unified memory, so we can estimate based on total system memory
        memory = psutil.virtual_memory()
        total_memory_gb = memory.total / (1024**3)
        
        # Rough estimation: GPU typically gets 30-50% of unified memory
        # This is a conservative estimate
        return total_memory_gb * 0.3
    
    def _classify_hardware_tier(self, cpu_count: int, total_memory_gb: float, 
                               backend_type: Optional[BackendType], 
                               gpu_memory_gb: Optional[float]) -> HardwareTier:
        """Classify hardware into capability tiers."""
        # Basic tier: CPU-only with minimal resources
        if backend_type == BackendType.CPU and total_memory_gb < 8:
            return HardwareTier.BASIC
        
        # Powerful tier: High-end GPU or significant resources
        if (backend_type == BackendType.CUDA and gpu_memory_gb and gpu_memory_gb >= 8) or \
           (backend_type == BackendType.MPS and total_memory_gb >= 16) or \
           (cpu_count >= 16 and total_memory_gb >= 32):
            return HardwareTier.POWERFUL
        
        # Standard tier: Everything else
        return HardwareTier.STANDARD
    
    def get_device_recommendation(self, mode: str = "auto") -> Tuple[BackendType, str]:
        """
        Get device recommendation based on hardware capabilities and mode.
        
        Args:
            mode: "development", "production", or "auto"
            
        Returns:
            Tuple of (BackendType, explanation)
        """
        capabilities = self.detect_capabilities()
        
        if mode == "development":
            # Development mode: Prefer MPS for Mac, CPU for others
            if capabilities.mps_available:
                return BackendType.MPS, "Development mode: Using MPS for Apple Silicon"
            else:
                return BackendType.CPU, "Development mode: Using CPU (no GPU available)"
        
        elif mode == "production":
            # Production mode: Prefer CUDA, fallback to MPS, then CPU
            if capabilities.cuda_available:
                return BackendType.CUDA, "Production mode: Using CUDA for maximum performance"
            elif capabilities.mps_available:
                return BackendType.MPS, "Production mode: Using MPS (CUDA not available)"
            else:
                return BackendType.CPU, "Production mode: Using CPU (no GPU available)"
        
        else:  # auto mode
            # Auto mode: Use best available backend
            if capabilities.cuda_available:
                return BackendType.CUDA, "Auto mode: CUDA available, using for best performance"
            elif capabilities.mps_available:
                return BackendType.MPS, "Auto mode: MPS available, using for Apple Silicon"
            else:
                return BackendType.CPU, "Auto mode: Using CPU (no GPU available)"
    
    def get_capability_report(self) -> str:
        """
        Generate a comprehensive capability report.
        
        Returns:
            Formatted capability report string.
        """
        capabilities = self.detect_capabilities()
        
        report = []
        report.append("=== Hardware Capability Report ===")
        report.append(f"Platform: {capabilities.platform}")
        report.append(f"Architecture: {capabilities.architecture}")
        report.append(f"CPU Cores: {capabilities.cpu_count}")
        report.append(f"Total Memory: {capabilities.total_memory_gb:.1f} GB")
        report.append(f"Available Memory: {capabilities.available_memory_gb:.1f} GB")
        report.append(f"Hardware Tier: {capabilities.hardware_tier.value.upper()}")
        report.append("")
        
        report.append("=== Backend Availability ===")
        report.append(f"MPS Built: {capabilities.mps_built}")
        report.append(f"MPS Available: {capabilities.mps_available}")
        report.append(f"CUDA Available: {capabilities.cuda_available}")
        if capabilities.cuda_available:
            report.append(f"CUDA Devices: {capabilities.cuda_device_count}")
        report.append("")
        
        if capabilities.backend_type:
            report.append("=== Primary Backend ===")
            report.append(f"Type: {capabilities.backend_type.value.upper()}")
            if capabilities.gpu_name:
                report.append(f"GPU: {capabilities.gpu_name}")
            if capabilities.gpu_memory_gb:
                report.append(f"GPU Memory: {capabilities.gpu_memory_gb:.1f} GB")
            report.append("")
        
        # Mode recommendations
        report.append("=== Mode Recommendations ===")
        dev_backend, dev_explanation = self.get_device_recommendation("development")
        prod_backend, prod_explanation = self.get_device_recommendation("production")
        
        report.append(f"Development Mode: {dev_backend.value.upper()} - {dev_explanation}")
        report.append(f"Production Mode: {prod_backend.value.upper()} - {prod_explanation}")
        
        return "\n".join(report)
    
    def get_optimal_batch_size(self, mode: str = "auto") -> int:
        """
        Get optimal batch size based on hardware capabilities and mode.
        
        Args:
            mode: "development", "production", or "auto"
            
        Returns:
            Recommended batch size.
        """
        capabilities = self.detect_capabilities()
        backend, _ = self.get_device_recommendation(mode)
        
        if backend == BackendType.CUDA and capabilities.gpu_memory_gb:
            # CUDA batch size based on GPU memory
            if capabilities.gpu_memory_gb >= 16:
                return 32
            elif capabilities.gpu_memory_gb >= 8:
                return 16
            else:
                return 8
        elif backend == BackendType.MPS:
            # MPS batch size based on unified memory
            if capabilities.total_memory_gb >= 32:
                return 16
            elif capabilities.total_memory_gb >= 16:
                return 8
            else:
                return 4
        else:
            # CPU batch size
            return 2


# Global instance for convenience
hardware_detector = HardwareDetector()


def get_hardware_capabilities() -> HardwareCapabilities:
    """Get hardware capabilities using the global detector."""
    return hardware_detector.detect_capabilities()


def get_device_recommendation(mode: str = "auto") -> Tuple[BackendType, str]:
    """Get device recommendation using the global detector."""
    return hardware_detector.get_device_recommendation(mode)


def get_capability_report() -> str:
    """Get capability report using the global detector."""
    return hardware_detector.get_capability_report()


def get_optimal_batch_size(mode: str = "auto") -> int:
    """Get optimal batch size using the global detector."""
    return hardware_detector.get_optimal_batch_size(mode)


if __name__ == "__main__":
    # Example usage
    print(get_capability_report())
    print("\n" + "="*50 + "\n")
    
    backend, explanation = get_device_recommendation("development")
    print(f"Development recommendation: {backend.value} - {explanation}")
    
    backend, explanation = get_device_recommendation("production")
    print(f"Production recommendation: {backend.value} - {explanation}")
    
    batch_size = get_optimal_batch_size("production")
    print(f"Optimal batch size: {batch_size}")
