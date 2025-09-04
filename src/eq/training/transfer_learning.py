#!/usr/bin/env python3
"""
Transfer Learning Utilities for FastAI v2

This module provides utilities for transfer learning between segmentation models,
specifically handling the namespace issues with custom functions.
"""

from pathlib import Path
from typing import Optional, Union
import torch
from fastai.vision.all import *
# BCEWithLogitsLossFlat import removed - using FastAI v2 automatic loss selection

from eq.data_management.standard_getters import get_y_universal
from eq.utils.logger import get_logger

logger = get_logger("eq.transfer_learning")

def _format_run_suffix(epochs: int, batch_size: int, learning_rate: float, image_size: int, tag: str = "") -> str:
    lr_str = (f"{learning_rate:.0e}" if learning_rate < 1e-2 else f"{learning_rate:.3f}").replace("-0", "-")
    parts = [f"e{epochs}", f"b{batch_size}", f"lr{lr_str}", f"sz{image_size}"]
    if tag:
        parts.insert(0, tag)
    return "_".join(parts)


def load_model_for_transfer_learning(
    model_path: Union[str, Path],
    target_data_dir: Union[str, Path],
    batch_size: int = 8,
    num_workers: int = 0
) -> Learner:
    """
    Load a pretrained model for transfer learning, handling namespace issues.
    
    This function creates a new learner with the same architecture as the
    pretrained model but with fresh data loaders for the target task.
    
    Args:
        model_path: Path to the pretrained model (.pkl file)
        target_data_dir: Directory containing target task data
        batch_size: Batch size for new data loaders
        num_workers: Number of workers for data loading
        
    Returns:
        Learner: New learner ready for transfer learning
    """
    logger.info(f"Loading pretrained model from: {model_path}")
    
    # Make get_y function available in global namespace for model loading
    import sys
    from typing import Any
    current_module = sys.modules[__name__]
    if not hasattr(current_module, 'get_y'):
        # Expose getter for FastAI's pickled learner; silence type checker
        setattr(current_module, 'get_y', get_y_universal)  # type: ignore[attr-defined]
    
    # Create new data loaders for target task
    from eq.data_management.datablock_loader import build_segmentation_dls
    target_dls = build_segmentation_dls(target_data_dir, bs=batch_size, num_workers=num_workers)
    
    logger.info(f"Created target data loaders: {len(target_dls.train_ds)} train, {len(target_dls.valid_ds)} val")
    
    # Create new learner with same architecture as mitochondria model (FastAI v2 approach)
    learn = unet_learner(
        target_dls,
        resnet34,
        n_out=2,  # 2 classes: background (0) + glomeruli (1) - matches mitochondria model
        metrics=Dice,  # Standard Dice metric works with multiclass!
    )
    # FastAI automatically sets CrossEntropyLossFlat for n_out=2, don't override
    print(f"Transfer learning using loss function: {learn.loss_func}")
    
    # Load the pretrained weights
    try:
        # Method 1: Try to load the full learner and extract weights
        pretrained_learn = load_learner(model_path)
        logger.info("Successfully loaded pretrained learner")
        
        # Copy the model weights (only compatible layers)
        pretrained_state = pretrained_learn.model.state_dict()
        current_state = learn.model.state_dict()
        
        # Only copy weights for layers that exist in both models
        compatible_weights = {}
        for key, value in pretrained_state.items():
            if key in current_state and current_state[key].shape == value.shape:
                compatible_weights[key] = value
            else:
                logger.debug(f"Skipping incompatible layer: {key}")
        
        current_state.update(compatible_weights)
        learn.model.load_state_dict(current_state)
        logger.info(f"Successfully loaded {len(compatible_weights)} compatible layers")
        
    except Exception as e:
        logger.warning(f"Could not load full learner: {e}")
        logger.info("Attempting to load just the model weights...")
        
        try:
            # Method 2: Try to load just the model state dict
            checkpoint = torch.load(model_path, map_location='cpu')
            if 'model' in checkpoint:
                pretrained_state = checkpoint['model']
                current_state = learn.model.state_dict()
                
                # Only copy compatible weights
                compatible_weights = {}
                for key, value in pretrained_state.items():
                    if key in current_state and current_state[key].shape == value.shape:
                        compatible_weights[key] = value
                
                current_state.update(compatible_weights)
                learn.model.load_state_dict(current_state)
                logger.info(f"Successfully loaded {len(compatible_weights)} compatible layers from checkpoint")
            else:
                logger.warning("No 'model' key found in checkpoint, using random initialization")
        except Exception as e2:
            logger.warning(f"Could not load model weights: {e2}")
            logger.info("Proceeding with random initialization")
    
    return learn


def transfer_learn_glomeruli(
    base_model_path: Union[str, Path],
    glomeruli_data_dir: Union[str, Path],
    output_dir: Union[str, Path],
    model_name: str = "glomeruli_transfer_model",
    epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 1e-4,  # Lower LR for fine-tuning
    freeze_encoder: bool = False
) -> Learner:
    """
    Perform transfer learning from mitochondria to glomeruli segmentation.
    
    Args:
        base_model_path: Path to pretrained mitochondria model
        glomeruli_data_dir: Directory containing glomeruli data
        output_dir: Directory to save the trained model
        model_name: Name for the output model
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for fine-tuning
        freeze_encoder: Whether to freeze the encoder during training
        
    Returns:
        Learner: Trained glomeruli model
    """
    logger.info("Starting transfer learning from mitochondria to glomeruli")
    
    # Create transfer learning output directory
    output_path = Path(output_dir) / "transfer"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load pretrained model for transfer learning
    learn = load_model_for_transfer_learning(
        base_model_path, 
        glomeruli_data_dir, 
        batch_size=batch_size
    )
    
    # Optionally freeze encoder
    if freeze_encoder:
        logger.info("Freezing encoder for fine-tuning")
        learn.freeze()
    else:
        logger.info("Training full model (unfrozen)")
        learn.unfreeze()
    
    # Train with lower learning rate for fine-tuning
    logger.info(f"Training for {epochs} epochs with LR={learning_rate}")
    learn.fit_one_cycle(epochs, lr_max=learning_rate)
    
    # Save training plots similar to mito
    try:
        learn.recorder.plot_loss()
        plt.savefig(output_path / f"{model_name}_training_loss.png")
        plt.close()
    except Exception as _e:
        logger.warning(f"Could not save training loss plot: {_e}")

    try:
        learn.show_results(max_n=8, figsize=(8, 8))
        plt.savefig(output_path / f"{model_name}_validation_predictions.png")
        plt.close()
    except Exception as _e:
        logger.warning(f"Could not save validation predictions plot: {_e}")

    # Save the model (include params in name)
    model_tag = _format_run_suffix(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, image_size=learn.dls.one_batch()[0].shape[-1], tag="xfer")
    model_path = output_path / f"{model_name}-{model_tag}.pkl"
    learn.export(model_path)
    logger.info(f"Model saved to: {model_path}")
    
    return learn
