#!/usr/bin/env python3
"""
Model loading utilities for endotheliosis quantifier.

This module provides safe loading of pre-trained models with proper
function definitions in the namespace.
"""

from pathlib import Path

from fastai.vision.all import *

from eq.utils.logger import get_logger


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
    from eq.utils.model_loader import get_glom_mask_file
    p2c = [0, 1]  # Default binary mask codes
    return get_glom_mask_file(o, p2c)


def get_glom_mask_file(image_file, p2c, thresh=127):
    """Get mask file path for a given image file - extracted from __main__.py."""
    # this is the base path
    base_path = image_file.parent.parent.parent
    first_name = image_file.parent.name
    # get training or testing from here
    import re
    full_name = re.findall(string=image_file.name, pattern=r"^[A-Za-z]*[0-9]+[_|-]+[A-Za-z]*[0-9]+")[0]
    
    # put the whole thing together
    str_name = f'{full_name}_mask' + image_file.suffix
    # attach it to the correct path
    mask_path = (base_path / 'masks' / first_name / str_name)
    
    # convert to an array (mask)
    msk = np.array(PILMask.create(mask_path))
    # convert the image to binary if it isn't already (tends to happen when working with .jpg files)
    msk[msk <= thresh] = 0
    msk[msk > thresh] = 1
    
    # find all the possible values in the mask (0,255)
    for i, val in enumerate(p2c):
        msk[msk == p2c[i]] = val
    return PILMask.create(msk)


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


def load_mitochondria_model(model_path: str) -> Learner:
    """
    Load the pre-trained mitochondria segmentation model.
    
    Args:
        model_path: Path to the mitochondria model file
        
    Returns:
        Loaded FastAI learner
    """
    logger = get_logger("eq.model_loader")
    
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
    logger = get_logger("eq.model_loader")
    
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
    logger = get_logger("eq.model_loader")
    
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


def get_model_info(learn: Learner) -> dict:
    """
    Get information about a loaded model.
    
    Args:
        learn: Loaded FastAI learner
        
    Returns:
        Dictionary with model information
    """
    info = {
        'model_type': type(learn.model).__name__,
        'dls_type': type(learn.dls).__name__ if hasattr(learn, 'dls') else 'Unknown',
        'loss_func': type(learn.loss_func).__name__ if hasattr(learn, 'loss_func') else 'Unknown',
        'metrics': [type(m).__name__ for m in learn.metrics] if hasattr(learn, 'metrics') else [],
        'device': str(learn.device) if hasattr(learn, 'device') else 'Unknown'
    }
    
    # Try to get model architecture details
    try:
        if hasattr(learn.model, 'encoder'):
            info['encoder'] = type(learn.model.encoder).__name__
        if hasattr(learn.model, 'decoder'):
            info['decoder'] = type(learn.model.decoder).__name__
    except:
        pass
    
    return info

