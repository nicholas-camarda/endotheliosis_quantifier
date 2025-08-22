"""
Backend management for dual-environment architecture.

This module provides backend abstraction layer for MPS/CUDA switching
and device management across different hardware configurations.
"""

import torch
import logging
from enum import Enum
from typing import Optional, Dict, Any, Tuple, Union
from dataclasses import dataclass

from eq.utils.hardware_detection import BackendType, get_device_recommendation, get_optimal_batch_size

logger = logging.getLogger(__name__)


class BackendStatus(Enum):
    """Status of backend availability."""
    
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    FALLBACK = "fallback"


@dataclass
class BackendInfo:
    """Information about a specific backend."""
    
    backend_type: BackendType
    status: BackendStatus
    device: Optional[torch.device] = None
    memory_gb: Optional[float] = None
    device_name: Optional[str] = None
    error_message: Optional[str] = None


class BackendManager:
    """
    Manages backend abstraction layer for MPS/CUDA switching.
    
    This class provides a unified interface for backend management,
    automatic device selection, and fallback mechanisms.
    """
    
    def __init__(self, preferred_backend: Optional[BackendType] = None):
        """
        Initialize the backend manager.
        
        Args:
            preferred_backend: Preferred backend type (defaults to auto-detection)
        """
        self.preferred_backend = preferred_backend
        self._backends = self._detect_available_backends()
        self._current_backend = self._select_optimal_backend()
        
        logger.info(f"BackendManager initialized. Current backend: {self._current_backend.backend_type.value}")
    
    @property
    def current_backend(self) -> BackendInfo:
        """Get the current backend information."""
        return self._current_backend
    
    @property
    def available_backends(self) -> Dict[BackendType, BackendInfo]:
        """Get all available backends."""
        return self._backends
    
    def get_device(self) -> torch.device:
        """
        Get the current device for PyTorch operations.
        
        Returns:
            PyTorch device
        """
        return self._current_backend.device
    
    def switch_backend(self, backend_type: BackendType) -> bool:
        """
        Switch to a different backend.
        
        Args:
            backend_type: Backend type to switch to
            
        Returns:
            True if switch was successful, False otherwise
        """
        if backend_type not in self._backends:
            logger.error(f"Backend {backend_type.value} is not available")
            return False
        
        backend_info = self._backends[backend_type]
        if backend_info.status == BackendStatus.UNAVAILABLE:
            logger.error(f"Backend {backend_type.value} is unavailable: {backend_info.error_message}")
            return False
        
        self._current_backend = backend_info
        logger.info(f"Switched to backend: {backend_type.value}")
        return True
    
    def get_optimal_batch_size(self, mode: str = "auto") -> int:
        """
        Get optimal batch size for current backend.
        
        Args:
            mode: Mode for batch size calculation
            
        Returns:
            Optimal batch size
        """
        return get_optimal_batch_size(mode)
    
    def get_backend_summary(self) -> str:
        """
        Get a summary of all backends and their status.
        
        Returns:
            Formatted summary string
        """
        summary = f"Current Backend: {self._current_backend.backend_type.value}\n"
        summary += f"Device: {self._current_backend.device}\n"
        
        if self._current_backend.device_name:
            summary += f"Device Name: {self._current_backend.device_name}\n"
        
        if self._current_backend.memory_gb:
            summary += f"Memory: {self._current_backend.memory_gb:.1f} GB\n"
        
        summary += "\nAvailable Backends:\n"
        for backend_type, backend_info in self._backends.items():
            status_icon = "✅" if backend_info.status == BackendStatus.AVAILABLE else "❌"
            summary += f"  {status_icon} {backend_type.value}: {backend_info.status.value}"
            
            if backend_info.error_message:
                summary += f" ({backend_info.error_message})"
            
            summary += "\n"
        
        return summary.strip()
    
    def validate_backend(self, backend_type: BackendType) -> Tuple[bool, str]:
        """
        Validate if a backend is suitable for the current task.
        
        Args:
            backend_type: Backend type to validate
            
        Returns:
            Tuple of (is_valid, reason)
        """
        if backend_type not in self._backends:
            return False, f"Backend {backend_type.value} is not available"
        
        backend_info = self._backends[backend_type]
        
        if backend_info.status == BackendStatus.UNAVAILABLE:
            return False, f"Backend {backend_type.value} is unavailable: {backend_info.error_message}"
        
        # Check memory requirements for GPU backends
        if backend_type in [BackendType.MPS, BackendType.CUDA]:
            if backend_info.memory_gb and backend_info.memory_gb < 4.0:
                return False, f"Backend {backend_type.value} has insufficient memory: {backend_info.memory_gb:.1f} GB"
        
        return True, f"Backend {backend_type.value} is suitable"
    
    def _detect_available_backends(self) -> Dict[BackendType, BackendInfo]:
        """
        Detect all available backends.
        
        Returns:
            Dictionary of backend information
        """
        backends = {}
        
        # Detect MPS (Apple Silicon)
        backends[BackendType.MPS] = self._detect_mps_backend()
        
        # Detect CUDA (NVIDIA)
        backends[BackendType.CUDA] = self._detect_cuda_backend()
        
        # Detect CPU (always available as fallback)
        backends[BackendType.CPU] = self._detect_cpu_backend()
        
        return backends
    
    def _detect_mps_backend(self) -> BackendInfo:
        """Detect MPS backend availability."""
        try:
            if not torch.backends.mps.is_available():
                return BackendInfo(
                    backend_type=BackendType.MPS,
                    status=BackendStatus.UNAVAILABLE,
                    error_message="MPS not available"
                )
            
            if not torch.backends.mps.is_built():
                return BackendInfo(
                    backend_type=BackendType.MPS,
                    status=BackendStatus.UNAVAILABLE,
                    error_message="MPS not built"
                )
            
            device = torch.device("mps")
            
            # Test device availability
            try:
                test_tensor = torch.ones(1, device=device)
                del test_tensor
            except Exception as e:
                return BackendInfo(
                    backend_type=BackendType.MPS,
                    status=BackendStatus.UNAVAILABLE,
                    error_message=f"Device test failed: {str(e)}"
                )
            
            # Get device information
            device_name = "Apple Silicon GPU"
            memory_gb = self._estimate_mps_memory()
            
            return BackendInfo(
                backend_type=BackendType.MPS,
                status=BackendStatus.AVAILABLE,
                device=device,
                device_name=device_name,
                memory_gb=memory_gb
            )
            
        except Exception as e:
            return BackendInfo(
                backend_type=BackendType.MPS,
                status=BackendStatus.UNAVAILABLE,
                error_message=str(e)
            )
    
    def _detect_cuda_backend(self) -> BackendInfo:
        """Detect CUDA backend availability."""
        try:
            if not torch.cuda.is_available():
                return BackendInfo(
                    backend_type=BackendType.CUDA,
                    status=BackendStatus.UNAVAILABLE,
                    error_message="CUDA not available"
                )
            
            device = torch.device("cuda")
            
            # Get device information
            device_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            return BackendInfo(
                backend_type=BackendType.CUDA,
                status=BackendStatus.AVAILABLE,
                device=device,
                device_name=device_name,
                memory_gb=memory_gb
            )
            
        except Exception as e:
            return BackendInfo(
                backend_type=BackendType.CUDA,
                status=BackendStatus.UNAVAILABLE,
                error_message=str(e)
            )
    
    def _detect_cpu_backend(self) -> BackendInfo:
        """Detect CPU backend availability."""
        try:
            device = torch.device("cpu")
            
            return BackendInfo(
                backend_type=BackendType.CPU,
                status=BackendStatus.AVAILABLE,
                device=device,
                device_name="CPU"
            )
            
        except Exception as e:
            return BackendInfo(
                backend_type=BackendType.CPU,
                status=BackendStatus.UNAVAILABLE,
                error_message=str(e)
            )
    
    def _select_optimal_backend(self) -> BackendInfo:
        """
        Select the optimal backend based on availability and preferences.
        
        Returns:
            Optimal backend information
        """
        # If preferred backend is specified and available, use it
        if self.preferred_backend and self.preferred_backend in self._backends:
            backend_info = self._backends[self.preferred_backend]
            if backend_info.status == BackendStatus.AVAILABLE:
                return backend_info
        
        # Priority order: CUDA > MPS > CPU
        priority_order = [BackendType.CUDA, BackendType.MPS, BackendType.CPU]
        
        for backend_type in priority_order:
            if backend_type in self._backends:
                backend_info = self._backends[backend_type]
                if backend_info.status == BackendStatus.AVAILABLE:
                    return backend_info
        
        # Fallback to CPU if nothing else is available
        return self._backends[BackendType.CPU]
    
    def _estimate_mps_memory(self) -> Optional[float]:
        """
        Estimate MPS memory (unified memory on Apple Silicon).
        
        Returns:
            Estimated memory in GB
        """
        try:
            import psutil
            # MPS uses unified memory, so we estimate based on system memory
            total_memory_gb = psutil.virtual_memory().total / (1024**3)
            # Assume 70% of system memory is available for MPS
            return total_memory_gb * 0.7
        except ImportError:
            return None
    
    def get_memory_info(self) -> Dict[str, Any]:
        """
        Get memory information for the current backend.
        
        Returns:
            Dictionary with memory information
        """
        info = {
            "backend": self._current_backend.backend_type.value,
            "device": str(self._current_backend.device),
            "memory_gb": self._current_backend.memory_gb
        }
        
        if self._current_backend.backend_type == BackendType.CUDA:
            try:
                info["allocated_gb"] = torch.cuda.memory_allocated() / (1024**3)
                info["cached_gb"] = torch.cuda.memory_reserved() / (1024**3)
            except:
                pass
        
        return info
    
    def clear_memory(self) -> None:
        """Clear memory for the current backend."""
        if self._current_backend.backend_type == BackendType.CUDA:
            try:
                torch.cuda.empty_cache()
                logger.info("Cleared CUDA memory cache")
            except:
                pass
        elif self._current_backend.backend_type == BackendType.MPS:
            # MPS memory management is handled by the system
            logger.info("MPS memory is managed by the system")
    
    def get_backend_recommendation(self, mode: str = "auto") -> Tuple[BackendType, str]:
        """
        Get backend recommendation for a specific mode.
        
        Args:
            mode: Mode for recommendation
            
        Returns:
            Tuple of (recommended_backend, explanation)
        """
        backend, explanation = get_device_recommendation(mode)
        
        # Convert string to BackendType enum
        if backend == "mps":
            return BackendType.MPS, explanation
        elif backend == "cuda":
            return BackendType.CUDA, explanation
        else:
            return BackendType.CPU, explanation
