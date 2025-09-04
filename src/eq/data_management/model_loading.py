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
from typing import Any, Union

from eq.core.constants import MPS_FALLBACK_ENV_VAR
from eq.utils.logger import get_logger

# FastAI is required by design
from fastai.vision.all import Learner, load_learner  # type: ignore


# Define the functions that were used during model training
def get_y(x: Any) -> str:
    """Get mask path for image path - used in mitochondria training."""
    return str(x).replace('.jpg', '.png').replace('img_', 'mask_')


def get_items_func(x: Any) -> Any:
    """Get items function - used in glomeruli transfer learning."""
    return x


def get_y_func(x: Any) -> str:
    """Get y function - used in glomeruli transfer learning."""
    return str(x).replace('.jpg', '_mask.png')


def get_glom_y(o: Any) -> Any:
    """Get glomeruli mask for a given image file - used in glomeruli training."""
    # Import the actual implementation from datablock_loader
    from eq.data_management.datablock_loader import default_get_y
    return default_get_y(o)


def _patch_main_module() -> None:
    """Patch the __main__ module with required functions."""
    import __main__

    # Add the required functions to __main__ (use setattr to satisfy type checkers)
    setattr(__main__, 'get_y', get_y)
    setattr(__main__, 'get_items_func', get_items_func)
    setattr(__main__, 'get_y_func', get_y_func)
    setattr(__main__, 'get_glom_y', get_glom_y)
    
    # Also add to globals for safety
    globals()['get_y'] = get_y
    globals()['get_items_func'] = get_items_func
    globals()['get_y_func'] = get_y_func
    globals()['get_glom_y'] = get_glom_y


def setup_model_loading_environment() -> None:
    """
    Set up the environment for proper model loading.
    """
    os.environ[MPS_FALLBACK_ENV_VAR] = '1'
    _patch_main_module()


def load_mitochondria_model(model_path: str) -> Learner:
    """Load the pre-trained mitochondria segmentation model."""
    logger = get_logger("eq.model_loading")
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Mitochondria model not found at: {model_path}")
    
    logger.info(f"Loading mitochondria model from: {model_path}")
    
    # Patch the main module with required functions
    _patch_main_module()
    
    # Load the model
    learn: Learner = load_learner(model_path)  # type: ignore[call-arg]
    logger.info("✅ Mitochondria model loaded successfully")
    return learn


def load_glomeruli_model(model_path: str) -> Learner:
    """Load the pre-trained glomeruli transfer learning model."""
    logger = get_logger("eq.model_loading")
    
    # Patch the main module with required functions
    _patch_main_module()
    
    # Load the model
    learn: Learner = load_learner(model_path)  # type: ignore[call-arg]
    logger.info("✅ Glomeruli model loaded successfully")
    return learn


def load_model_safely(model_path: str, model_type: str = "auto") -> Learner:
    """
    Safely load a model by automatically detecting the type.
    """
    logger = get_logger("eq.model_loading")
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found at: {model_path}")
    
    if model_type == "auto":
        # Auto-detect based on filename
        name_lower = Path(model_path).name.lower()
        if "mito" in name_lower:
            model_type = "mito"
        elif "glomerulus" in name_lower:
            model_type = "glomeruli"
        else:
            model_type = "glomeruli"
    
    logger.info(f"Auto-detected model type: {model_type}")
    
    if model_type == "mito":
        return load_mitochondria_model(model_path)
    elif model_type == "glomeruli":
        return load_glomeruli_model(model_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def load_model_with_historical_support(model_path: Union[str, Path], 
                                     setup_environment: bool = True) -> Learner:
    """
    Load a FastAI model with historical function support.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if setup_environment:
        setup_model_loading_environment()
    learner: Learner = load_learner(model_path)  # type: ignore[call-arg]
    return learner


def get_model_info(learner: Learner) -> dict:
    """Get comprehensive information about a loaded model."""
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
        if hasattr(learner, 'model') and getattr(learner, 'model') is not None:
            info["model_class"] = type(learner.model).__name__  # type: ignore[attr-defined]
            try:
                if hasattr(learner.model, 'parameters'):
                    params = list(learner.model.parameters())  # type: ignore[attr-defined]
                    if params:
                        info["device"] = str(params[0].device)
            except Exception:
                pass
            if hasattr(learner.model, 'encoder'):
                info['encoder'] = type(learner.model.encoder).__name__  # type: ignore[attr-defined]
            if hasattr(learner.model, 'decoder'):
                info['decoder'] = type(learner.model.decoder).__name__  # type: ignore[attr-defined]
        
        if hasattr(learner, 'dls') and getattr(learner, 'dls') is not None:
            info["dls_type"] = type(learner.dls).__name__  # type: ignore[attr-defined]
            info["has_vocab"] = hasattr(learner.dls, 'vocab')  # type: ignore[attr-defined]
            if hasattr(learner.dls, 'vocab'):
                vocab = getattr(learner.dls, 'vocab')  # type: ignore[attr-defined]
                info["vocab_size"] = len(vocab) if vocab else 0
        
        if hasattr(learner, 'loss_func') and getattr(learner, 'loss_func') is not None:
            info["loss_func"] = type(learner.loss_func).__name__  # type: ignore[attr-defined]
        
        if hasattr(learner, 'metrics') and getattr(learner, 'metrics') is not None:
            metrics = getattr(learner, 'metrics')
            try:
                info["metrics"] = [type(m).__name__ for m in metrics]  # type: ignore[assignment]
            except Exception:
                pass
        
        if hasattr(learner, 'device') and getattr(learner, 'device') is not None:
            info["device"] = str(getattr(learner, 'device'))
        
    except Exception as e:
        logger.warning(f"Error getting detailed model info: {e}")
        info["info_error"] = str(e)
    
    return info


def validate_model_compatibility(model_path: Union[str, Path]) -> dict:
    """Validate if a model can be loaded with current setup."""
    model_path = Path(model_path)
    
    results: dict = {
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
    
    try:
        learner = load_model_with_historical_support(model_path)
        if learner is not None:
            results["can_load"] = True
            results["model_info"] = get_model_info(learner)
    except Exception as e:
        results["error"] = str(e)
    
    return results


# Note: legacy aliases removed to minimize duplication
