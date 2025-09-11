#!/usr/bin/env python3
"""
Transfer Learning Utilities for FastAI v2

This module provides utilities for transfer learning between segmentation models,
specifically handling the namespace issues with custom functions.
"""

from pathlib import Path
from typing import Optional, Union
import torch
import matplotlib.pyplot as plt
from fastai.vision.all import *
# BCEWithLogitsLossFlat import removed - using FastAI v2 automatic loss selection

from eq.data_management.standard_getters import get_y
from eq.training.losses import make_loss
from eq.utils.logger import get_logger
from eq.utils.run_io import (
    save_splits, attach_best_model_callback, save_plots, 
    save_training_history, save_run_metadata, export_final_model
)

logger = get_logger("eq.transfer_learning")


def _get_encoder_module(model: torch.nn.Module) -> Optional[torch.nn.Module]:
    """Attempt to retrieve the encoder/body module from a DynamicUNet-like model."""
    # Common attribute on some implementations
    enc = getattr(model, 'encoder', None)
    if enc is not None:
        return enc
    # FastAI DynamicUnet is typically Sequential-like with encoder as first child
    try:
        first_child = getattr(model, '0', None)
        if first_child is not None and isinstance(first_child, torch.nn.Module):
            return first_child
    except Exception:
        pass
    try:
        children = list(model.children()) if hasattr(model, 'children') else []
        if len(children) > 0:
            return children[0]
    except Exception:
        pass
    return None


def _get_decoder_module(model: torch.nn.Module) -> Optional[torch.nn.Module]:
    """Attempt to retrieve the decoder/head module(s) from a DynamicUNet-like model."""
    # Common attribute on some implementations
    dec = getattr(model, 'decoder', None)
    if dec is not None:
        return dec
    # FastAI DynamicUnet is typically Sequential-like with decoder as second child
    try:
        second_child = getattr(model, '1', None)
        if second_child is not None and isinstance(second_child, torch.nn.Module):
            return second_child
    except Exception:
        pass
    try:
        children = list(model.children()) if hasattr(model, 'children') else []
        if len(children) > 1:
            # Group all but the first child as decoder-side
            return torch.nn.Sequential(*children[1:])
    except Exception:
        pass
    return None

def _format_run_suffix(epochs: int, batch_size: int, learning_rate: float, image_size: int, tag: str = "") -> str:
    """Format run parameters into a descriptive suffix (directory-safe)."""
    # Use scientific notation for learning rate to avoid decimal points
    lr_str = f"{learning_rate:.0e}".replace("-0", "-")
    parts = [f"e{epochs}", f"b{batch_size}", f"lr{lr_str}", f"sz{image_size}"]
    if tag:
        parts.insert(0, tag)
    return "_".join(parts)


def load_model_for_transfer_learning(
    model_path: Union[str, Path],
    target_data_dir: Union[str, Path],
    image_size: int = 256,
    crop_size: Optional[int] = None,
    batch_size: int = 8,
    num_workers: int = 0,
    output_path: Optional[Union[str, Path]] = None,
    use_dynamic_patching: bool = True,
    loss_name: Optional[str] = None,
    positive_focus_p: float = 0.6,
    min_pos_pixels: int = 64,
    pos_crop_attempts: int = 10,
    load_encoder_only: bool = True,
    reinit_decoder: bool = True,
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
        setattr(current_module, 'get_y', get_y)  # type: ignore[attr-defined]
    
    # Create new data loaders for target task with positive-aware cropping
    from eq.data_management.datablock_loader import build_segmentation_dls, build_segmentation_dls_dynamic_patching
    if use_dynamic_patching:
        target_dls = build_segmentation_dls_dynamic_patching(
            target_data_dir, 
            bs=batch_size, 
            num_workers=num_workers,
            crop_size=(crop_size if crop_size is not None else image_size),
            output_size=image_size,
            positive_focus_p=positive_focus_p,
            min_pos_pixels=min_pos_pixels,
            pos_crop_attempts=pos_crop_attempts,
        )
    else:
        target_dls = build_segmentation_dls(
            target_data_dir, 
            bs=batch_size, 
            num_workers=num_workers
        )
    
    logger.info(f"Created target data loaders: {len(target_dls.train_ds)} train, {len(target_dls.valid_ds)} val")
    
    # Create new learner with same architecture as mitochondria model (FastAI v2 approach)
    custom_loss = make_loss(loss_name or "")

    if output_path is not None:
        output_path = Path(output_path)
        learn = unet_learner(
            target_dls,
            resnet34,
            n_out=2,  # 2 classes: background (0) + glomeruli (1) - matches mitochondria model
            metrics=Dice,  # Standard Dice metric works with multiclass!
            loss_func=custom_loss if custom_loss else None,
            path=output_path,  # Save artifacts directly under the model output directory
            model_dir='.'  # Ensure callbacks/save go inside output_path
        )
    else:
        learn = unet_learner(
            target_dls,
            resnet34,
            n_out=2,  # 2 classes: background (0) + glomeruli (1) - matches mitochondria model
            metrics=Dice,  # Standard Dice metric works with multiclass!
            loss_func=custom_loss if custom_loss else None
        )
    
    # Enable mixed precision to significantly reduce VRAM usage
    try:
        learn = learn.to_fp16()
        logger.info("Enabled mixed-precision (fp16) training for reduced GPU memory usage")
    except Exception:
        logger.warning("Could not enable mixed-precision; proceeding in fp32")

    if custom_loss is not None:
        logger.info(f"Using custom loss: {custom_loss.__class__.__name__}")
    else:
        logger.info(f"Using default loss function: {learn.loss_func}")
    
    # Load the pretrained weights
    try:
        # Method 1: Try to load the full learner and extract weights
        pretrained_learn = load_learner(model_path)
        logger.info("Successfully loaded pretrained learner")
        
        if load_encoder_only:
            # Strict encoder-only: copy by submodule state_dict
            src_enc = _get_encoder_module(pretrained_learn.model)
            dst_enc = _get_encoder_module(learn.model)
            if src_enc is None or dst_enc is None:
                raise RuntimeError("Could not resolve encoder modules for strict encoder-only loading")
            src_sd = src_enc.state_dict()
            dst_sd = dst_enc.state_dict()
            copied = 0
            for k, v in src_sd.items():
                if k in dst_sd and dst_sd[k].shape == v.shape:
                    dst_sd[k] = v
                    copied += 1
            dst_enc.load_state_dict(dst_sd)
            logger.info(f"Strict encoder-only weights loaded: {copied} params")
        else:
            # Copy all compatible layers (full model)
            pretrained_state = pretrained_learn.model.state_dict()
            current_state = learn.model.state_dict()
            compatible_weights = {}
            for key, value in pretrained_state.items():
                if key in current_state and current_state[key].shape == value.shape:
                    compatible_weights[key] = value
            current_state.update(compatible_weights)
            learn.model.load_state_dict(current_state)
            logger.info(f"Successfully loaded {len(compatible_weights)} compatible layers (full model)")
        
    except Exception as e:
        logger.warning(f"Could not load full learner: {e}")
        logger.info("Attempting to load just the model weights...")
        
        try:
            # Method 2: Try to load just the model state dict
            checkpoint = torch.load(model_path, map_location='cpu')
            if 'model' in checkpoint:
                pretrained_state = checkpoint['model']
                if load_encoder_only:
                    # Strict encoder-only via submodule state_dicts
                    src_learn = load_learner(model_path)
                    src_enc = _get_encoder_module(src_learn.model)
                    dst_enc = _get_encoder_module(learn.model)
                    if src_enc is None or dst_enc is None:
                        raise RuntimeError("Could not resolve encoder modules for strict encoder-only loading (checkpoint)")
                    src_sd = src_enc.state_dict()
                    dst_sd = dst_enc.state_dict()
                    copied = 0
                    for k, v in src_sd.items():
                        if k in dst_sd and dst_sd[k].shape == v.shape:
                            dst_sd[k] = v
                            copied += 1
                    dst_enc.load_state_dict(dst_sd)
                    logger.info(f"Strict encoder-only weights loaded from checkpoint: {copied} params")
                else:
                    current_state = learn.model.state_dict()
                    compatible_weights = {}
                    for key, value in pretrained_state.items():
                        if key in current_state and current_state[key].shape == value.shape:
                            compatible_weights[key] = value
                    current_state.update(compatible_weights)
                    learn.model.load_state_dict(current_state)
                    logger.info(f"Successfully loaded {len(compatible_weights)} compatible layers from checkpoint (full model)")
            else:
                logger.warning("No 'model' key found in checkpoint, using random initialization")
        except Exception as e2:
            logger.warning(f"Could not load model weights: {e2}")
            logger.info("Proceeding with random initialization")
    
    # Optionally reinitialize decoder to avoid negative transfer from source task
    if reinit_decoder:
        try:
            model = learn.model
            # Try to directly access encoder if available
            encoder_module = getattr(model, 'encoder', None)
            if encoder_module is None:
                # Heuristic fallbacks
                encoder_module = getattr(model, '0', None)
                if encoder_module is None and hasattr(model, 'children'):
                    children = list(model.children())
                    if len(children) > 0:
                        encoder_module = children[0]

            # Build a set of parameter ids that belong to the encoder to avoid resetting them
            encoder_param_ids = set()
            if encoder_module is not None:
                try:
                    for p in encoder_module.parameters():
                        encoder_param_ids.add(id(p))
                except Exception:
                    pass

            reset_count = 0
            visited = set()
            for submodule in model.modules():
                # Skip the top-level model itself and the encoder subtree
                if submodule is model:
                    continue
                # Determine if this submodule belongs to the encoder by checking its parameters
                belongs_to_encoder = False
                try:
                    for p in submodule.parameters(recurse=False):
                        if id(p) in encoder_param_ids:
                            belongs_to_encoder = True
                            break
                except Exception:
                    pass
                if belongs_to_encoder:
                    continue
                if hasattr(submodule, 'reset_parameters') and callable(getattr(submodule, 'reset_parameters')):
                    # Avoid resetting the same module twice
                    if id(submodule) in visited:
                        continue
                    try:
                        submodule.reset_parameters()
                        visited.add(id(submodule))
                        reset_count += 1
                    except Exception:
                        pass

            if reset_count > 0:
                logger.info(f"Decoder/head reinitialized (layers reset: {reset_count})")
            else:
                logger.info("No non-encoder layers were reinitialized (structure may differ)")
        except Exception as _e:
            logger.warning(f"Failed to reinitialize decoder/head: {_e}")
    
    return learn


def transfer_learn_glomeruli(
    base_model_path: Union[str, Path],
    glomeruli_data_dir: Union[str, Path],
    output_dir: Union[str, Path],
    model_name: str = "glomeruli_transfer_model",
    epochs: int = 30,  # Total epochs (stage1 + stage2)
    batch_size: int = 16,  # Increased from 8 for more stable gradients
    learning_rate: float = 1e-3,  # Reduced from 5e-4 for proper transfer learning
    freeze_encoder: bool = False,  # Will be handled by two-stage approach
    image_size: int = 256,
    stage1_epochs: Optional[int] = None,  # Auto-calculate if not provided
    stage2_epochs: Optional[int] = None,  # Auto-calculate if not provided
    stage1_lr: float = 1e-3,  # New: higher LR for head training
    stage2_lr: Optional[float] = None,  # New: lower LR for fine-tuning
    use_lr_find: bool = True,
    config_path: Optional[str] = None,
    use_dynamic_patching: bool = True,
    positive_focus_p: float = 0.6,
    min_pos_pixels: int = 64,
    pos_crop_attempts: int = 10,
    loss_name: Optional[str] = None,
    crop_size: Optional[int] = None,
    encoder_only: bool = True,
    reinit_decoder: bool = True,
) -> Learner:
    """
    Perform transfer learning from mitochondria to glomeruli segmentation using FastAI v2 best practices.
    
    This implements the recommended two-stage approach:
    1. Stage 1: Freeze encoder, train only the head with higher learning rate
    2. Stage 2: Unfreeze all layers, fine-tune with lower learning rate
    
    Args:
        base_model_path: Path to pretrained mitochondria model
        glomeruli_data_dir: Directory containing glomeruli data
        output_dir: Directory to save the trained model
        model_name: Name for the output model
        epochs: Total training epochs (stage1 + stage2)
        batch_size: Training batch size
        learning_rate: Base learning rate (used for stage1)
        freeze_encoder: Deprecated - handled by two-stage approach
        stage1_epochs: Epochs for frozen encoder training (default: 3)
        stage2_epochs: Epochs for unfrozen fine-tuning (default: 27)
        stage1_lr: Learning rate for stage 1 (default: 1e-3)
        stage2_lr: Learning rate for stage 2 (default: 1e-5)
        
    Returns:
        Learner: Trained glomeruli model
    """
    # Set up logging for transfer learning
    from eq.utils.logger import setup_logging
    logger = setup_logging(verbose=True)
    logger.info("Starting transfer learning from mitochondria to glomeruli")
    
    # Calculate stage epochs if not provided
    if stage1_epochs is None:
        stage1_epochs = max(1, epochs // 10)  # 10% of total epochs for stage 1
    if stage2_epochs is None:
        stage2_epochs = epochs - stage1_epochs  # Remaining epochs for stage 2
    
    # If stage2_epochs is 0 or negative, adjust stage1_epochs to ensure both stages get at least 1 epoch
    if stage2_epochs <= 0:
        stage1_epochs = max(1, epochs - 1)  # Leave at least 1 epoch for stage 2
        stage2_epochs = epochs - stage1_epochs
    
    logger.info(f"Training schedule: {stage1_epochs} epochs (frozen) + {stage2_epochs} epochs (unfrozen) = {epochs} total")
    
    # Create transfer learning output directory with descriptive model name
    # Include loss in tag for traceability (e.g., transfer_loss-dice)
    _loss_key = None
    try:
        if loss_name:
            k = (loss_name or "").strip().lower()
            if k in ("dice", "bcedice", "tversky"):
                _loss_key = k
            else:
                _loss_key = "custom"
    except Exception:
        _loss_key = None
    # Compose a more truthful tag: include stage-1 LR and the intended stage-2 LR source
    s1lr_str = f"s1lr{stage1_lr:.0e}".replace("-0", "-")
    # If stage2_lr <= 0 we will use lr_find; reflect that in the tag now
    if stage2_lr is None or stage2_lr <= 0:
        s2lr_str = "s2lr_lrfind"
    else:
        s2lr_str = f"s2lr{stage2_lr:.0e}".replace("-0", "-")
    _tag = "transfer" + (f"_loss-{_loss_key}" if _loss_key else "") + f"_{s1lr_str}_{s2lr_str}"
    model_tag = _format_run_suffix(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, image_size=image_size, tag=_tag)
    model_folder_name = f"{model_name}-{model_tag}"
    
    # Create the transfer subdirectory
    transfer_dir = Path(output_dir) / "transfer"
    transfer_dir.mkdir(parents=True, exist_ok=True)
    output_path = transfer_dir / model_folder_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load pretrained model for transfer learning with weighted loss
    
    learn = load_model_for_transfer_learning(
        base_model_path, 
        glomeruli_data_dir, 
        image_size=image_size,
        crop_size=crop_size,
        batch_size=batch_size,
        output_path=output_path,
        use_dynamic_patching=use_dynamic_patching,
        loss_name=loss_name,
        positive_focus_p=positive_focus_p,
        min_pos_pixels=min_pos_pixels,
        pos_crop_attempts=pos_crop_attempts,
        load_encoder_only=encoder_only,
        reinit_decoder=reinit_decoder,
    )
    
    # Save data splits manifest
    save_splits(output_path, model_folder_name, {
        "stage": "glomeruli_transfer",
        "train_items": getattr(learn.dls.train_ds, 'items', []),
        "valid_items": getattr(learn.dls.valid_ds, 'items', [])
    })
    
    # STAGE 1: Train with frozen encoder (FastAI v2 best practice)
    logger.info("STAGE 1: Training with frozen encoder")
    learn.freeze()
    logger.info(f"Training frozen encoder for {stage1_epochs} epochs with LR={stage1_lr}")
    # In stage 1 optimize for validation loss (head warmup)
    save_callback = attach_best_model_callback(model_folder_name, monitor='valid_loss')
    learn.fit_one_cycle(stage1_epochs, lr_max=stage1_lr, cbs=[save_callback])
    
    # STAGE 2: Unfreeze and fine-tune (FastAI v2 best practice)
    logger.info("STAGE 2: Unfreezing encoder for fine-tuning")
    learn.unfreeze()
    
    # Determine learning rate for fine-tuning
    if use_lr_find and (stage2_lr is None or stage2_lr <= 0):
        logger.info("Finding optimal learning rate for fine-tuning...")
        try:
            plt.close('all')
        except Exception:
            pass
        lr_find_results = learn.lr_find()
        try:
            fig = lr_find_results.plot(suggestion=True)
            fig.savefig(output_path / f"{model_folder_name}_lr_finder.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
        except Exception:
            try:
                plt.close('all')
            except Exception:
                pass
        try:
            valley_lr = float(getattr(lr_find_results, 'valley', 0.0) or 0.0)
        except Exception:
            valley_lr = 0.0
        fine_tune_lr = float(valley_lr) if valley_lr > 0 else 1e-4
        logger.info(f"Using lr_find valley for fine-tune: {fine_tune_lr}")
    else:
        fine_tune_lr = float(stage2_lr if stage2_lr and stage2_lr > 0 else 1e-4)
        logger.info(f"Skipping lr_find; using provided fine-tune LR: {fine_tune_lr}")
    # Use conservative discriminative learning rates to stabilize Dice
    # These are standard in the FastAI v2 documentation
    low_lr = max(fine_tune_lr / 30.0, fine_tune_lr * (1.0/30.0))
    high_lr = max(fine_tune_lr / 3.0, fine_tune_lr * (1.0/3.0))
    logger.info(f"Using discriminative LRs for fine-tuning: slice({low_lr}, {high_lr})")
    
    # In stage 2, monitor validation loss for checkpointing and early stopping
    from fastai.callback.tracker import EarlyStoppingCallback
    loss_save_cb = attach_best_model_callback(model_folder_name, monitor='valid_loss')
    early_stop_cb = EarlyStoppingCallback(monitor='valid_loss', patience=10)
    logger.info(f"Fine-tuning for {stage2_epochs} epochs with LR slice (monitor=valid_loss, early stopping)")
    learn.fit_one_cycle(stage2_epochs, lr_max=slice(low_lr, high_lr), cbs=[loss_save_cb, early_stop_cb])
    
    # Save training history BEFORE any plotting/predictions that may alter recorder state
    save_training_history(learn, output_path, model_folder_name, {
        'total_epochs': epochs,
        'stage1_epochs': stage1_epochs,
        'stage2_epochs': stage2_epochs,
        'batch_size': batch_size,
        'stage1_learning_rate': stage1_lr,
        'stage2_learning_rate': stage2_lr,
        'stage2_learning_rate_used': float(fine_tune_lr),
        'crop_size': int(crop_size) if crop_size is not None else int(image_size),
        'output_size': int(image_size),
        'base_model_path': str(base_model_path),
        'training_approach': 'two_stage_transfer_learning'
    })
    
    # Generate training visualizations (may call get_preds etc.)
    logger.info("Generating training visualizations...")
    save_plots(learn, output_path, model_folder_name)

    # Save the model
    model_path = export_final_model(learn, output_path, model_folder_name)
    
    # Save run metadata
    save_run_metadata(output_path, model_folder_name, config_path)
    
    # Also save a small text file with the LR used for quick inspection
    try:
        with open(output_path / 'fine_tune_lr.txt', 'w') as lr_f:
            lr_f.write(str(float(fine_tune_lr)))
    except Exception:
        pass
    
    return learn
