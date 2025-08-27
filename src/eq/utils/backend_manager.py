"""
Backend management for dual-environment architecture.

This module provides backend abstraction layer for MPS/CUDA switching
and device management across different hardware configurations.
"""

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

import torch

from eq.utils.hardware_detection import BackendType

logger = logging.getLogger(__name__)


class BackendStatus(Enum):
    """Status of backend availability."""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


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
    """Manages backend abstraction layer for MPS/CUDA switching."""
    
    def __init__(self, preferred_backend: Optional[BackendType] = None):
        """Initialize the backend manager."""
        self.preferred_backend = preferred_backend
        self._backends = self._detect_available_backends()
        self._current_backend = self._select_optimal_backend()
        
        # Automatically configure MPS fallback for Mac systems
        self._configure_mps_fallback()
        
        logger.info(f"BackendManager initialized. Current backend: {self._current_backend.backend_type.value}")
    
    def _configure_mps_fallback(self):
        """Automatically configure MPS fallback for Mac systems."""
        if self._current_backend.backend_type == BackendType.MPS:
            # Set MPS fallback environment variable for unsupported operations
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            logger.info("✅ Automatically enabled MPS fallback for unsupported operations")
            
            # Log MPS capabilities
            logger.info(f"MPS Backend: {self._current_backend.device_name}")
            if self._current_backend.memory_gb:
                logger.info(f"MPS Memory: {self._current_backend.memory_gb:.1f}GB")
    
    @property
    def current_backend(self) -> BackendInfo:
        """Get the current backend information."""
        return self._current_backend
    
    def get_device(self) -> torch.device:
        """Get the current device for PyTorch operations."""
        return self._current_backend.device
    
    def switch_backend(self, backend_type: BackendType) -> bool:
        """Switch to a different backend."""
        if backend_type not in self._backends:
            logger.error(f"Backend {backend_type.value} is not available")
            return False
        
        backend_info = self._backends[backend_type]
        if backend_info.status == BackendStatus.UNAVAILABLE:
            logger.error(f"Backend {backend_type.value} is unavailable: {backend_info.error_message}")
            return False
        
        self._current_backend = backend_info
        
        # Reconfigure MPS fallback if switching to MPS
        if backend_type == BackendType.MPS:
            self._configure_mps_fallback()
        
        logger.info(f"Switched to backend: {backend_type.value}")
        return True
    
    def _detect_available_backends(self) -> Dict[BackendType, BackendInfo]:
        """Detect all available backends."""
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
            
            device = torch.device("mps")
            return BackendInfo(
                backend_type=BackendType.MPS,
                status=BackendStatus.AVAILABLE,
                device=device,
                device_name="Apple Silicon GPU"
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
            return BackendInfo(
                backend_type=BackendType.CUDA,
                status=BackendStatus.AVAILABLE,
                device=device,
                device_name="NVIDIA GPU"
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
        """Select the optimal backend based on availability."""
        # Priority order: CUDA > MPS > CPU
        priority_order = [BackendType.CUDA, BackendType.MPS, BackendType.CPU]
        
        for backend_type in priority_order:
            if backend_type in self._backends:
                backend_info = self._backends[backend_type]
                if backend_info.status == BackendStatus.AVAILABLE:
                    return backend_info
        
        # Fallback to CPU
        return self._backends[BackendType.CPU]
