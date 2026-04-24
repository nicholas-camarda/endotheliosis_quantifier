#!/usr/bin/env python3
"""Current-namespace FastAI model loading utilities."""

from pathlib import Path
from typing import Union

from eq.utils.logger import get_logger

# FastAI is required by design
from fastai.vision.all import Learner, load_learner  # type: ignore


def load_mitochondria_model(model_path: str) -> Learner:
    """Load a current-namespace mitochondria segmentation model."""
    logger = get_logger("eq.model_loading")
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Mitochondria model not found at: {model_path}")
    
    logger.info(f"Loading mitochondria model from: {model_path}")

    learn: Learner = load_learner(model_path)  # type: ignore[call-arg]
    logger.info("✅ Mitochondria model loaded successfully")
    return learn


def load_glomeruli_model(model_path: str) -> Learner:
    """Load a current-namespace glomeruli segmentation model."""
    logger = get_logger("eq.model_loading")

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Glomeruli model not found at: {model_path}")

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
        name_lower = Path(model_path).name.lower()
        if "mito" in name_lower:
            model_type = "mito"
        elif "glomerulus" in name_lower or "glomeruli" in name_lower:
            model_type = "glomeruli"
        else:
            raise ValueError(
                f"Could not auto-detect model type from filename: {model_path}. "
                "Pass model_type='mito' or model_type='glomeruli'."
            )
    
    logger.info(f"Auto-detected model type: {model_type}")
    
    if model_type == "mito":
        return load_mitochondria_model(model_path)
    elif model_type == "glomeruli":
        return load_glomeruli_model(model_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


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
        learner = load_model_safely(str(model_path), model_type="auto")
        if learner is not None:
            results["can_load"] = True
            results["model_info"] = get_model_info(learner)
    except Exception as e:
        results["error"] = str(e)
    
    return results


# Note: legacy aliases removed to minimize duplication
