#!/usr/bin/env python3
"""
Comprehensive Model Loading Module

This module provides unified model loading functionality with support for
historical model compatibility. Consolidates ALL model loading logic that was
previously scattered across multiple files.

Features:
- Safe model loading with proper function definitions
- Automatic model type detection
- Historical function compatibility
- MPS/CUDA/CPU hardware awareness
- Comprehensive error handling and logging
"""

import os
from pathlib import Path
from typing import Optional, Union

from eq.core.constants import MPS_FALLBACK_ENV_VAR
from eq.utils.logger import get_logger

# Import FastAI components (with fallback handling)
try:
    from fastai.vision.all import Learner, load_learner
except ImportError:
    load_learner = None
    Learner = None


# Define the functions that were used during model training
def get_y(x):
    """Get mask path for image path - used in mitochondria training."""
    return str(x).replace('.jpg', '.png').replace('img_', 'mask_')


def get_items_func(x):
    """Get items function - used in glomeruli transfer learning."""
    return x


def get_y_func(x):
    """Get y function - used in glomeruli transfer learning."""
    return str(x).replace('.jpg', '_mask.png')


def get_glom_y(o):
    """Get glomeruli mask for a given image file - used in glomeruli training."""
    # This is the actual function used during glomeruli training
    # Import here to avoid circular imports
    from eq.data_management.data_loading import get_glom_mask_file
    p2c = [0, 1]  # Default binary mask codes
    return get_glom_mask_file(o, p2c)


def _patch_main_module():
    """Patch the __main__ module with required functions."""
    import __main__

    # Add the required functions to __main__
    __main__.get_y = get_y
    __main__.get_items_func = get_items_func
    __main__.get_y_func = get_y_func
    __main__.get_glom_y = get_glom_y
    
    # Also add to globals for safety
    globals()['get_y'] = get_y
    globals()['get_items_func'] = get_items_func
    globals()['get_y_func'] = get_y_func
    globals()['get_glom_y'] = get_glom_y


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
    _patch_main_module()


def load_mitochondria_model(model_path: str) -> Learner:
    """
    Load the pre-trained mitochondria segmentation model.
    
    Args:
        model_path: Path to the mitochondria model file
        
    Returns:
        Loaded FastAI learner
    """
    logger = get_logger("eq.model_loading")
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Mitochondria model not found at: {model_path}")
    
    logger.info(f"Loading mitochondria model from: {model_path}")
    
    try:
        # Patch the main module with required functions
        _patch_main_module()
        
        # Load the model
        learn = load_learner(model_path)
        logger.info("✅ Mitochondria model loaded successfully")
        return learn
        
    except Exception as e:
        logger.error(f"Failed to load mitochondria model: {e}")
        raise


def load_glomeruli_model(model_path: str) -> Learner:
    """
    Load the pre-trained glomeruli transfer learning model.
    
    Args:
        model_path: Path to the glomeruli model file
        
    Returns:
        Loaded FastAI learner
    """
    logger = get_logger("eq.model_loading")
    
    try:
        # Patch the main module with required functions
        _patch_main_module()
        
        # Load the model
        learn = load_learner(model_path)
        logger.info("✅ Glomeruli model loaded successfully")
        return learn
        
    except Exception as e:
        logger.error(f"Failed to load glomeruli model: {e}")
        raise


def load_model_safely(model_path: str, model_type: str = "auto") -> Learner:
    """
    Safely load a model by automatically detecting the type.
    
    Args:
        model_path: Path to the model file
        model_type: Type of model ("mito", "glomeruli", or "auto")
        
    Returns:
        Loaded FastAI learner
    """
    logger = get_logger("eq.model_loading")
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found at: {model_path}")
    
    if model_type == "auto":
        # Auto-detect based on filename
        if "mito" in Path(model_path).name.lower():
            model_type = "mito"
        elif "glomerulus" in Path(model_path).name.lower():
            model_type = "glomeruli"
        else:
            # Default to glomeruli for backward compatibility
            model_type = "glomeruli"
    
    logger.info(f"Auto-detected model type: {model_type}")
    
    if model_type == "mito":
        return load_mitochondria_model(model_path)
    elif model_type == "glomeruli":
        return load_glomeruli_model(model_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


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
    Get comprehensive information about a loaded model.
    
    Args:
        learner: FastAI Learner object
        
    Returns:
        Dictionary with detailed model information
    """
    logger = get_logger("eq.model_loading")
    
    if learner is None:
        return {"error": "No model provided"}
    
    info = {
        "model_type": type(learner).__name__,
        "has_model": hasattr(learner, 'model'),
        "has_dls": hasattr(learner, 'dls'),
        "has_loss_func": hasattr(learner, 'loss_func'),
        "has_metrics": hasattr(learner, 'metrics'),
        "has_device": hasattr(learner, 'device'),
    }
    
    try:
        if hasattr(learner, 'model') and learner.model is not None:
            info["model_class"] = type(learner.model).__name__
            
            # Try to get device information
            try:
                if hasattr(learner.model, 'parameters'):
                    params = list(learner.model.parameters())
                    if params:
                        info["device"] = str(params[0].device)
            except:
                pass
            
            # Try to get model architecture details
            if hasattr(learner.model, 'encoder'):
                info['encoder'] = type(learner.model.encoder).__name__
            if hasattr(learner.model, 'decoder'):
                info['decoder'] = type(learner.model.decoder).__name__
        
        if hasattr(learner, 'dls') and learner.dls is not None:
            info["dls_type"] = type(learner.dls).__name__
            info["has_vocab"] = hasattr(learner.dls, 'vocab')
            if hasattr(learner.dls, 'vocab'):
                info["vocab_size"] = len(learner.dls.vocab) if learner.dls.vocab else 0
        
        if hasattr(learner, 'loss_func') and learner.loss_func is not None:
            info["loss_func"] = type(learner.loss_func).__name__
        
        if hasattr(learner, 'metrics') and learner.metrics is not None:
            info["metrics"] = [type(m).__name__ for m in learner.metrics]
        
        if hasattr(learner, 'device') and learner.device is not None:
            info["device"] = str(learner.device)
        
    except Exception as e:
        logger.warning(f"Error getting detailed model info: {e}")
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


# Note: legacy aliases removed to minimize duplication
