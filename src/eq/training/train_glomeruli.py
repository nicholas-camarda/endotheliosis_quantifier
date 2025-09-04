#!/usr/bin/env python3
"""
Train Glomeruli Segmentation Model with Transfer Learning - FastAI v2 Compatible

This script trains a glomeruli segmentation model using transfer learning from a 
pretrained mitochondria segmentation model. The approach:
1. Loads a pretrained mitochondria model
2. Fine-tunes it on glomeruli data using transfer learning
3. Saves the trained glomeruli model

This is the second stage of the two-stage training pipeline.
"""

import random
import re
import sys
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

import matplotlib.pyplot as plt
from fastai.vision.all import *

from eq.utils.logger import get_logger
from eq.utils.config_manager import ConfigManager
from eq.data_management.datablock_loader import build_segmentation_dls
from eq.data_management.standard_getters import get_y_glomeruli, get_y_universal
from eq.training.transfer_learning import transfer_learn_glomeruli
from fastai.losses import BCEWithLogitsLossFlat
from eq.core.constants import (
    DEFAULT_IMAGE_SIZE, DEFAULT_MASK_THRESHOLD, DEFAULT_PREDICTION_THRESHOLD, 
    DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, DEFAULT_LEARNING_RATE,
    DEFAULT_FLIP_VERT, DEFAULT_MAX_ROTATE, DEFAULT_MIN_ZOOM, DEFAULT_MAX_ZOOM,
    DEFAULT_MAX_WARP, DEFAULT_MAX_LIGHTING, DEFAULT_RANDOM_ERASING_P,
    DEFAULT_RANDOM_ERASING_SL, DEFAULT_RANDOM_ERASING_SH, 
    DEFAULT_RANDOM_ERASING_MIN_ASPECT, DEFAULT_RANDOM_ERASING_MAX_COUNT,
    DEFAULT_GLOMERULI_MODEL_DIR
)

logger = get_logger("eq.retrain_glomeruli_original")

def _format_run_suffix(epochs: int, batch_size: int, learning_rate: float, image_size: int, tag: str = "") -> str:
    """Create a concise, filesystem-safe suffix describing training params."""
    lr_str = (f"{learning_rate:.0e}" if learning_rate < 1e-2 else f"{learning_rate:.3f}").replace("-0", "-")
    parts = [f"e{epochs}", f"b{batch_size}", f"lr{lr_str}", f"sz{image_size}"]
    if tag:
        parts.insert(0, tag)
    return "_".join(parts)

def get_all_paths(directory_path):
    """Get all file paths from a directory recursively."""
    directory = Path(directory_path)
    paths = []
    for path in directory.glob('**/*'):
        if path.is_file():
            paths.append(path)
    return paths

def n_glom_codes(fnames, is_partial=True):
    """Gather the codes from a list of fnames, full file paths."""
    vals = set()
    if is_partial:
        random.shuffle(fnames)
        fnames = fnames[:10]
    for fname in fnames:
        msk = np.array(PILMask.create(fname))
        for val in np.unique(msk):
            if val not in vals:
                vals.add(val)
    vals = list(vals)
    p2c = dict()
    for i, val in enumerate(vals):
        p2c[i] = vals[i]
    return p2c

def get_glom_mask_file(image_file, p2c, thresh=DEFAULT_MASK_THRESHOLD):
    """Get glomeruli mask file with color mapping."""
    # For derived data, mask is in the mask_patches directory with '_mask' suffix
    mask_path = image_file.parent.parent / "mask_patches" / f"{image_file.stem}_mask{image_file.suffix}"
    
    # Convert to an array (mask)
    msk = np.array(PILMask.create(mask_path))
    # Derived data should already be binary, but ensure it's 0/1
    msk = (msk > DEFAULT_MASK_THRESHOLD).astype(np.uint8)
    return PILMask.create(msk)

# Use standardized getter function for compatibility
get_y = get_y_glomeruli


def train_glomeruli_with_transfer_learning(
    data_dir: str,
    output_dir: str,
    model_name: str,
    base_model_path: Optional[str] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = 1e-4,  # Lower LR for transfer learning
    image_size: int = DEFAULT_IMAGE_SIZE
) -> Learner:
    """
    Train glomeruli model using transfer learning from mitochondria model.
    
    Args:
        data_dir: Directory containing glomeruli data
        output_dir: Directory to save trained model
        model_name: Name for the model
        base_model_path: Path to pretrained mitochondria model
        batch_size: Training batch size
        epochs: Number of training epochs
        learning_rate: Learning rate for fine-tuning
        image_size: Input image size
        
    Returns:
        Learner: Trained glomeruli model
    """
    logger.info("Starting glomeruli training with transfer learning")
    
    if base_model_path is None:
        # Default path to mitochondria model
        base_model_path = "models/segmentation/mitochondria/mitochondria_model.pkl"
    
    if not Path(base_model_path).exists():
        logger.warning(f"Base model not found at {base_model_path}, training from scratch")
        return train_glomeruli_with_datablock(
            data_dir=data_dir,
            output_dir=output_dir,
            model_name=model_name,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            image_size=image_size,
        )
    
    # Use transfer learning
    learn = transfer_learn_glomeruli(
        base_model_path=base_model_path,
        glomeruli_data_dir=data_dir,
        output_dir=output_dir,
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
    
    return learn


def train_glomeruli_with_datablock(
    data_dir: str,
    output_dir: str,
    model_name: str,
    base_model_path: Optional[str] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    image_size: int = DEFAULT_IMAGE_SIZE
):
    """
    Train glomeruli segmentation model using FastAI v2 DataBlock approach with transfer learning.
    
    Args:
        data_dir: Directory containing image_patches/ and mask_patches/
        output_dir: Directory to save model and results
        model_name: Name for the model
        base_model_path: Path to pretrained mitochondria model for transfer learning
        batch_size: Training batch size
        epochs: Number of training epochs
        learning_rate: Learning rate
        image_size: Input image size
        
    Returns:
        Trained learner
    """
    logger = get_logger("eq.glomeruli_training")
    logger.info("Starting glomeruli model training with DataBlock and transfer learning...")
    
    # Create output directory
    output_path = Path(output_dir) / model_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading data from: {data_dir}")
    
    # Build DataLoaders using DataBlock approach
    dls = build_segmentation_dls(data_dir, bs=batch_size, num_workers=0)
    
    logger.info(f"Data loaded: {len(dls.train_ds)} train, {len(dls.valid_ds)} val samples")
    
    # Create learner
    learn = unet_learner(dls, resnet34, n_out=2, metrics=DiceMulti())
    
    # Load pretrained model if provided
    if base_model_path and Path(base_model_path).exists():
        logger.info(f"Loading pretrained model from: {base_model_path}")
        # For transfer learning, we need to load the exported learner and extract the model
        pretrained_learn = load_learner(base_model_path)
        # Copy the pretrained model weights to our new learner
        learn.model.load_state_dict(pretrained_learn.model.state_dict())
        logger.info("✅ Pretrained model loaded successfully")
    else:
        logger.info("No pretrained model provided, training from scratch")
    
    # Train the model
    logger.info(f"Training for {epochs} epochs...")
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
    model_tag = _format_run_suffix(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, image_size=image_size, tag="scratch")
    model_path = output_path / f"{model_name}-{model_tag}.pkl"
    learn.export(model_path)
    logger.info(f"Model saved to: {model_path}")
    
    # Return the learner for now (can be wrapped later if needed)
    return learn


def _removed_legacy_transfer():
    """Placeholder to mark removal of legacy duplicate function."""
    return None

def main():
    """CLI interface for glomeruli training."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train glomeruli segmentation model')
    parser.add_argument('--config', help='Optional YAML config file to override defaults')
    parser.add_argument('--data-dir', required=True, help='Directory containing derived_data (from eq process-data)')
    parser.add_argument('--model-dir', default=DEFAULT_GLOMERULI_MODEL_DIR, help='Directory to save trained model')
    parser.add_argument('--model-name', default='glomeruli_model', help='Base name for saved model files')
    parser.add_argument('--base-model', help='Path to base model for transfer learning')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE, help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=DEFAULT_LEARNING_RATE, help='Learning rate')
    parser.add_argument('--image-size', type=int, default=DEFAULT_IMAGE_SIZE, help='Input image size')
    
    args = parser.parse_args()

    # Optional: load YAML config and overlay onto args
    if args.config:
        try:
            import yaml  # type: ignore
            with open(args.config, 'r') as f:
                cfg_yaml = yaml.safe_load(f) or {}
            # Map common fields from YAML if present
            # pretrained model
            base_model = (
                cfg_yaml.get('pretrained_model', {}).get('path')
                if isinstance(cfg_yaml.get('pretrained_model'), dict) else None
            )
            if base_model and not args.base_model:
                args.base_model = base_model
            # training hyperparams
            model_cfg = cfg_yaml.get('model', {}) if isinstance(cfg_yaml.get('model'), dict) else {}
            training_cfg = model_cfg.get('training', {}) if isinstance(model_cfg.get('training'), dict) else {}
            if 'epochs' in training_cfg and not parser.get_default('epochs') == args.epochs:
                args.epochs = int(training_cfg['epochs'])
            if 'batch_size' in training_cfg and not parser.get_default('batch_size') == args.batch_size:
                args.batch_size = int(training_cfg['batch_size'])
            if 'learning_rate' in training_cfg and not parser.get_default('learning_rate') == args.learning_rate:
                args.learning_rate = float(training_cfg['learning_rate'])
            # image size
            if 'input_size' in model_cfg and isinstance(model_cfg['input_size'], (list, tuple)) and len(model_cfg['input_size']) >= 1:
                args.image_size = int(model_cfg['input_size'][0])
            # output model dir and name from checkpoint_path
            if 'checkpoint_path' in model_cfg:
                from pathlib import Path as _P
                ckpt = _P(model_cfg['checkpoint_path'])
                args.model_dir = str(ckpt.parent)
                # Only set model_name from YAML if CLI did not provide it
                if parser.get_default('model_name') == args.model_name:
                    args.model_name = ckpt.stem
        except Exception as _e:  # pragma: no cover
            print(f"⚠️  Failed to load config {args.config}: {_e}")
    
    try:
        logger = get_logger("eq.glomeruli_training")
        logger.info("🚀 Starting glomeruli model training...")
        logger.info(f"📁 Data directory: {args.data_dir}")
        logger.info(f"📁 Model directory: {args.model_dir}")
        logger.info(f"🧾 Model name: {args.model_name}")
        logger.info(f"⚙️  Epochs: {args.epochs}, Batch size: {args.batch_size}")
        
        if args.base_model:
            logger.info(f"🔄 Using base model: {args.base_model}")
        
        # Train the model using transfer learning if base model provided, otherwise from scratch
        if args.base_model:
            model = train_glomeruli_with_transfer_learning(
                data_dir=args.data_dir,
                output_dir=args.model_dir,
                model_name=args.model_name,
                base_model_path=args.base_model,
                batch_size=args.batch_size,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                image_size=args.image_size
            )
        else:
            model = train_glomeruli_with_datablock(
                data_dir=args.data_dir,
                output_dir=args.model_dir,
                model_name=args.model_name,
                base_model_path=args.base_model,
                batch_size=args.batch_size,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                image_size=args.image_size
            )
        
        logger.info("🎉 Glomeruli training completed successfully!")
        print(f"✅ Model saved to: {args.model_dir}/glomeruli_model")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        print(f"❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
