#!/usr/bin/env python3
"""
Inference Module

Minimal, explicit exports. No stubs or fallbacks.
"""

from .gpu_inference import GPUGlomeruliInference
from .prediction_core import PredictionCore, create_prediction_core


# Create a convenience function for run_gpu_inference
def run_gpu_inference(*args, **kwargs):
    """Convenience wrapper for GPUGlomeruliInference."""
    inference = GPUGlomeruliInference(*args, **kwargs)
    return inference

__all__ = [
    'run_gpu_inference',
    'GPUGlomeruliInference',
    'PredictionCore',
    'create_prediction_core',
]
