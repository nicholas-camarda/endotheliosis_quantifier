#!/usr/bin/env python3
"""
Train Mitochondria Segmentation Model - FastAI v2 Compatible

This script trains a mitochondria segmentation model on EM data without a
domain-specific base artifact.
This is the first stage of the two-stage training pipeline:
1. Train mitochondria model from an ImageNet-initialized ResNet34 encoder (this script)
2. Use mitochondria model as base for glomeruli transfer learning

The trained mitochondria model serves as a pretrained base for transfer learning
to glomeruli segmentation in the second stage.
"""

import sys
from pathlib import Path
from typing import Optional

from fastai.callback.all import *
from fastai.vision.all import *

from eq.utils.logger import get_logger
from eq.utils.run_io import (
    save_splits, attach_best_model_callback, save_plots, 
    save_training_history, save_run_metadata, export_final_model
)
from eq.core.constants import (
    DEFAULT_IMAGE_SIZE, 
    DEFAULT_EPOCHS, 
    DEFAULT_LEARNING_RATE,
    DEFAULT_MITOCHONDRIA_MODEL_DIR,
    DEFAULT_POSITIVE_FOCUS_P,
    DEFAULT_MIN_POS_PIXELS,
    DEFAULT_POS_CROP_ATTEMPTS,
)
from eq.data_management.datablock_loader import (
    TRAINING_MODE_DYNAMIC_FULL_IMAGE,
    build_segmentation_dls_dynamic_patching,
    validate_supported_segmentation_training_root,
)
from eq.utils.hardware_detection import get_segmentation_training_batch_size
from eq.utils.paths import resolve_runtime_path
from eq.training.segmentation_validation_audit import build_mitochondria_training_provenance

MITOCHONDRIA_ENCODER_INITIALIZATION = "imagenet_pretrained_resnet34"


def _format_run_suffix(epochs, batch_size, learning_rate, image_size, tag=""):
    """Format run parameters into a descriptive suffix (directory-safe)."""
    # Use scientific notation for learning rate to avoid decimal points
    lr_str = f"{learning_rate:.0e}".replace("-0", "-")
    parts = [f"e{epochs}", f"b{batch_size}", f"lr{lr_str}", f"sz{image_size}"]
    if tag:
        parts.insert(0, tag)
    return "_".join(parts)




def train_mitochondria_with_datablock(
    data_dir: str,
    output_dir: str,
    model_name: str,
    batch_size: Optional[int] = None,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    image_size: int = DEFAULT_IMAGE_SIZE,
    config_path: Optional[str] = None,
    positive_focus_p: float = DEFAULT_POSITIVE_FOCUS_P,
    min_pos_pixels: int = DEFAULT_MIN_POS_PIXELS,
    pos_crop_attempts: int = DEFAULT_POS_CROP_ATTEMPTS,
    device: Optional[str] = None,
):
    """
    Train mitochondria segmentation model using FastAI v2 DataBlock approach.
    
    Args:
        data_dir: Full-image training root containing images/ and masks/
        output_dir: Directory to save model and results
        model_name: Name for the model
        batch_size: Training batch size
        epochs: Number of training epochs
        learning_rate: Learning rate
        image_size: Input image size
        
    Returns:
        Tuple of (Trained FastAI learner instance, model_path)
    """
    logger = get_logger("eq.mitochondria_training")
    logger.info("🔬 Starting mitochondria model training with DataBlock...")
    data_root = validate_supported_segmentation_training_root(data_dir, stage="mitochondria")
    batch_size = get_segmentation_training_batch_size(
        "mitochondria",
        image_size=image_size,
        requested_batch_size=batch_size,
    )
    
    # Create output directory with descriptive model name
    model_tag = _format_run_suffix(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, image_size=image_size, tag="pretrain")
    model_folder_name = f"{model_name}-{model_tag}"
    output_path = Path(output_dir) / model_folder_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"📁 Output directory: {output_path}")
    logger.info(f"📊 Training parameters: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}, image_size={image_size}")
    logger.info(f"🔄 Training mode: {TRAINING_MODE_DYNAMIC_FULL_IMAGE}")
    logger.info(f"🎯 Positive-aware cropping: p={positive_focus_p}, min_pos={min_pos_pixels}, attempts={pos_crop_attempts}")
    logger.info(f"📂 Loading data from: {data_root}")
    
    # Build DataLoaders from full images using dynamic patching.
    logger.info("🔧 Building DataLoaders with dynamic patching...")
    dls = build_segmentation_dls_dynamic_patching(
        data_root,
        bs=batch_size,
        num_workers=0,
        crop_size=image_size,
        positive_focus_p=positive_focus_p,
        min_pos_pixels=min_pos_pixels,
        pos_crop_attempts=pos_crop_attempts,
        device=device,
    )

    # Mask validation handled centrally in datablock loader
    train_items = list(getattr(dls.train_ds, 'items', []))
    valid_items = list(getattr(dls.valid_ds, 'items', []))
    training_provenance = build_mitochondria_training_provenance(
        data_root=data_root,
        train_items=train_items,
        valid_items=valid_items,
        image_size=image_size,
        crop_size=image_size,
        output_size=image_size,
        positive_focus_p=positive_focus_p,
        min_pos_pixels=min_pos_pixels,
        pos_crop_attempts=pos_crop_attempts,
        command=" ".join(sys.argv),
    )
    training_provenance["training_device"] = str(dls.device)

    # Save data splits manifest
    save_splits(output_path, model_folder_name, {
        "stage": "mitochondria",
        "training_mode": TRAINING_MODE_DYNAMIC_FULL_IMAGE,
        "data_root": str(data_root),
        "train_items": train_items,
        "valid_items": valid_items,
        "split_seed": 42,
        "splitter_name": "RandomSplitter",
        **training_provenance,
    })
    
    logger.info(f"✅ Data loaded: {len(dls.train_ds)} train, {len(dls.valid_ds)} val samples")
    
    # Create learner (binary segmentation with n_out=2 and proper [0,1] normalization)
    logger.info("🏗️  Creating UNet learner with ResNet34 encoder...")
    learn = unet_learner(
        dls,
        resnet34,
        n_out=2,  # 2 classes: background (0) + mitochondria (1)
        pretrained=True,
        metrics=[Dice, JaccardCoeff()],  # Track both Dice and IoU for segmentation quality
        path=output_path,  # Save artifacts directly under the model output directory
        model_dir='.'  # Ensure callbacks/save go inside output_path
    )
    
    # FastAI automatically sets CrossEntropyLossFlat for n_out=2, don't override
    print(f"Using default loss function: {learn.loss_func}")
    
    # Train the model with callbacks for visualization
    logger.info(f"Training for {epochs} epochs...")
    
    # Add callbacks for training visualization
    logger.info("📋 Attaching training callbacks...")
    save_callback = attach_best_model_callback(model_folder_name)
    
    # Train with callbacks
    logger.info(f"🚀 Starting training for {epochs} epochs with learning rate {learning_rate}...")
    learn.fit_one_cycle(epochs, lr_max=learning_rate, cbs=[save_callback])
    logger.info("✅ Training completed!")
    
    # Save training history BEFORE any plotting/predictions that may alter recorder state
    save_training_history(learn, output_path, model_folder_name, {
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'image_size': image_size,
        'training_approach': 'no_domain_base',
        'candidate_family': 'mitochondria_no_domain_base',
        'encoder_initialization': MITOCHONDRIA_ENCODER_INITIALIZATION,
        'training_mode': TRAINING_MODE_DYNAMIC_FULL_IMAGE,
        'data_root': str(data_root),
        **training_provenance,
    })

    # Generate training visualizations
    logger.info("📊 Generating training visualizations...")
    learn.eq_validation_trace_context = {
        "stage": "mitochondria",
        "candidate_family": "mitochondria_no_domain_base",
        "resize_policy": training_provenance.get("resize_policy"),
        "threshold": None,
    }
    save_plots(learn, output_path, model_folder_name)
    
    # Print training summary
    logger.info("=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)
    
    # Safely access recorder data using FastAI v2 attributes
    if hasattr(learn, 'recorder') and learn.recorder:
        if hasattr(learn.recorder, 'log') and learn.recorder.log:
            logger.info(f"Training log entries: {len(learn.recorder.log)}")
        
        if hasattr(learn.recorder, 'values') and learn.recorder.values:
            final_values = learn.recorder.values[-1] if learn.recorder.values else []
            if len(final_values) >= 1:
                logger.info(f"Final training loss: {final_values[0]:.6f}")
            if len(final_values) >= 2:
                logger.info(f"Final validation loss: {final_values[1]:.6f}")
        
        if hasattr(learn.recorder, 'losses') and learn.recorder.losses:
            logger.info(f"Training losses recorded: {len(learn.recorder.losses)}")
            if learn.recorder.losses:
                logger.info(f"Final training loss: {learn.recorder.losses[-1]:.6f}")
        
        if hasattr(learn.recorder, 'metric_names') and learn.recorder.metric_names:
            logger.info(f"Metric names: {learn.recorder.metric_names}")
    else:
        logger.info("Training completed - recorder data not available")
    
    logger.info("=" * 60)
    
    # Save the model
    model_path = export_final_model(learn, output_path, model_folder_name)
    
    # Save run metadata
    save_run_metadata(
        output_path,
        model_folder_name,
        config_path,
        extra_metadata={
            "stage": "mitochondria",
            "artifact_status": "supported_runtime",
            "scientific_promotion_status": "not_evaluated",
            "training_mode": TRAINING_MODE_DYNAMIC_FULL_IMAGE,
            "data_root": str(data_root),
            "model_path": str(model_path),
            "candidate_family": "mitochondria_no_domain_base",
            "encoder_initialization": MITOCHONDRIA_ENCODER_INITIALIZATION,
            **training_provenance,
            "invocation": {
                "data_dir": str(data_root),
                "output_dir": str(output_dir),
                "model_name": model_name,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "image_size": image_size,
            },
        },
    )
    
    # Return the learner for now (can be wrapped later if needed)
    return learn, model_path


def main():
    """CLI interface for mitochondria training."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train mitochondria segmentation model from the ImageNet-initialized ResNet34 baseline')
    parser.add_argument('--config', help='Optional YAML config file to override defaults')
    parser.add_argument('--data-dir', required=True, help='Full-image training root containing images/ and masks/')
    parser.add_argument('--model-dir', default=DEFAULT_MITOCHONDRIA_MODEL_DIR, help='Directory to save trained model')
    parser.add_argument('--model-name', default='mitochondria_model', help='Base name for saved model files')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None, help='Training batch size (default: machine-aware recommendation)')
    parser.add_argument('--learning-rate', type=float, default=DEFAULT_LEARNING_RATE, help='Learning rate')
    parser.add_argument('--image-size', type=int, default=DEFAULT_IMAGE_SIZE, help='Input image size')
    parser.add_argument(
        '--device',
        choices=["mps", "cuda", "cpu"],
        help='Training device. Omit to auto-select cuda, then mps, then cpu.',
    )
    
    args = parser.parse_args()

    # Optional: load YAML config and overlay onto args
    config_path = None
    if args.config:
        try:
            import yaml  # type: ignore
            with open(args.config, 'r') as f:
                cfg_yaml = yaml.safe_load(f) or {}
            # training hyperparams
            model_cfg = cfg_yaml.get('model', {}) if isinstance(cfg_yaml.get('model'), dict) else {}
            training_cfg = model_cfg.get('training', {}) if isinstance(model_cfg.get('training'), dict) else {}
            if 'epochs' in training_cfg and parser.get_default('epochs') == args.epochs:
                args.epochs = int(training_cfg['epochs'])
            if 'batch_size' in training_cfg and args.batch_size is None:
                args.batch_size = int(training_cfg['batch_size'])
            if 'learning_rate' in training_cfg and parser.get_default('learning_rate') == args.learning_rate:
                args.learning_rate = float(training_cfg['learning_rate'])
            # output model dir from checkpoint_path
            if 'checkpoint_path' in cfg_yaml.get('model', {}):
                ckpt = resolve_runtime_path(cfg_yaml['model']['checkpoint_path'])
                args.model_dir = str(ckpt.parent)
                # Only take name from YAML if CLI didn't override
                if parser.get_default('model_name') == args.model_name:
                    args.model_name = ckpt.stem
            config_path = args.config
        except Exception as _e:  # pragma: no cover
            print(f"⚠️  Failed to load config {args.config}: {_e}")
    
    try:
        from eq.utils.logger import setup_logging
        logger = setup_logging(verbose=True)
        args.batch_size = get_segmentation_training_batch_size(
            "mitochondria",
            image_size=args.image_size,
            requested_batch_size=args.batch_size,
        )
        logger.info("🚀 Starting mitochondria model training...")
        logger.info(f"📁 Data directory: {args.data_dir}")
        logger.info(f"📁 Model directory: {args.model_dir}")
        logger.info(f"🧾 Model name: {args.model_name}")
        logger.info(f"⚙️  Epochs: {args.epochs}, Batch size: {args.batch_size}")
        logger.info(f"🎯 Learning rate: {args.learning_rate}")
        logger.info(f"📐 Image size: {args.image_size}")
        logger.info(f"🔄 Training mode: {TRAINING_MODE_DYNAMIC_FULL_IMAGE}")
        
        # Train the model using DataBlock approach
        model, model_path = train_mitochondria_with_datablock(
            data_dir=args.data_dir,
            output_dir=args.model_dir,
            model_name=args.model_name,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            image_size=args.image_size,
            config_path=config_path,
            device=args.device,
        )
        
        logger.info("🎉 Mitochondria training completed successfully!")
        print(f"✅ Model saved to: {model_path}")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        print(f"❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
