#!/usr/bin/env python3
"""
Core Model Loading Functions

This module provides unified model loading functionality with support for
historical model compatibility. Consolidates model loading logic that was
previously scattered across multiple files.
"""

import os
from pathlib import Path
from typing import Optional, Union

from .constants import MPS_FALLBACK_ENV_VAR
from .data_loading import setup_global_functions

# Import FastAI components (with fallback handling)
try:
    from fastai.vision.all import Learner, load_learner
except ImportError:
    load_learner = None
    Learner = None


def setup_model_loading_environment():
    """
    Set up the environment for proper model loading.
    
    This function:
    1. Sets up MPS fallback for Mac compatibility
    2. Injects required functions into global namespace
    3. Prepares environment for FastAI model loading
    """
    # Set MPS fallback for Mac compatibility (critical for model loading)
    os.environ[MPS_FALLBACK_ENV_VAR] = '1'
    
    # Set up global functions required for model loading
    setup_global_functions()


def load_model_with_historical_support(model_path: Union[str, Path], 
                                     setup_environment: bool = True) -> Optional[object]:
    """
    Load a FastAI model with historical function support.
    
    This function handles the complexities of loading models that were trained
    with custom functions that need to be available in the global namespace.
    
    Args:
        model_path: Path to the model file (.pkl)
        setup_environment: Whether to set up the environment (default: True)
        
    Returns:
        Loaded FastAI Learner object or None if loading fails
    """
    if load_learner is None:
        raise ImportError("FastAI not available. Please install fastai to use this function.")
    
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if setup_environment:
        setup_model_loading_environment()
    
    try:
        # Load the model with historical functions available
        learner = load_learner(model_path)
        return learner
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {e}")


def get_model_info(learner: object) -> dict:
    """
    Get information about a loaded model.
    
    Args:
        learner: FastAI Learner object
        
    Returns:
        Dictionary with model information
    """
    if learner is None:
        return {"error": "No model provided"}
    
    info = {
        "model_type": type(learner).__name__,
        "has_model": hasattr(learner, 'model'),
        "has_dls": hasattr(learner, 'dls'),
    }
    
    try:
        if hasattr(learner, 'model') and learner.model is not None:
            info["model_class"] = type(learner.model).__name__
            info["device"] = str(next(learner.model.parameters()).device)
        
        if hasattr(learner, 'dls') and learner.dls is not None:
            info["has_vocab"] = hasattr(learner.dls, 'vocab')
            if hasattr(learner.dls, 'vocab'):
                info["vocab_size"] = len(learner.dls.vocab) if learner.dls.vocab else 0
        
    except Exception as e:
        info["info_error"] = str(e)
    
    return info


def validate_model_compatibility(model_path: Union[str, Path]) -> dict:
    """
    Validate if a model can be loaded with current setup.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Dictionary with validation results
    """
    model_path = Path(model_path)
    
    results = {
        "file_exists": model_path.exists(),
        "file_size": model_path.stat().st_size if model_path.exists() else 0,
        "is_pickle": model_path.suffix == '.pkl',
        "can_load": False,
        "error": None
    }
    
    if not results["file_exists"]:
        results["error"] = "Model file does not exist"
        return results
    
    if not results["is_pickle"]:
        results["error"] = "Model file is not a pickle file (.pkl)"
        return results
    
    # Try to load the model
    try:
        learner = load_model_with_historical_support(model_path)
        if learner is not None:
            results["can_load"] = True
            results["model_info"] = get_model_info(learner)
    except Exception as e:
        results["error"] = str(e)
    
    return results
