#!/usr/bin/env python3
"""
Train Mitochondria Segmentation Model - FastAI v2 Compatible

This script trains a mitochondria segmentation model from scratch on EM data.
This is the first stage of the two-stage training pipeline:
1. Train mitochondria model from scratch (this script)
2. Use mitochondria model as base for glomeruli transfer learning

The trained mitochondria model serves as a pretrained base for transfer learning
to glomeruli segmentation in the second stage.
"""

import pickle
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from fastai.callback.all import *
from fastai.vision.all import *

from eq.utils.logger import get_logger
from eq.core.constants import (
    DEFAULT_IMAGE_SIZE, 
    DEFAULT_MASK_THRESHOLD, 
    DEFAULT_PREDICTION_THRESHOLD, 
    DEFAULT_BATCH_SIZE, 
    DEFAULT_EPOCHS, 
    DEFAULT_LEARNING_RATE,
    DEFAULT_MITOCHONDRIA_MODEL_DIR
)
from eq.data_management.datablock_loader import build_segmentation_dls
from eq.data_management.standard_getters import get_y_mitochondria
from fastai.losses import CrossEntropyLossFlat




def train_mitochondria_with_datablock(
    data_dir: str,
    output_dir: str,
    model_name: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    image_size: int = DEFAULT_IMAGE_SIZE
):
    """
    Train mitochondria segmentation model using FastAI v2 DataBlock approach.
    
    Args:
        data_dir: Directory containing image_patches/ and mask_patches/
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
    logger.info("Starting mitochondria model training with DataBlock...")
    
    # Create output directory - create subfolder named after the model
    output_path = Path(output_dir) / model_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading data from: {data_dir}")
    
    # Build DataLoaders using DataBlock approach
    dls = build_segmentation_dls(data_dir, bs=batch_size, num_workers=0)

    # Mask validation handled centrally in datablock loader

    # Use standardized getter function for compatibility
    get_y = get_y_mitochondria

    # Minimal split manifest for audit (write to model output folder)
    try:
        import json
        from datetime import datetime as _dt
        splits_dir = output_path
        splits_dir.mkdir(parents=True, exist_ok=True)
        split_manifest = {
            "stage": "mitochondria",
            "generated_at": _dt.now().isoformat(),
            "train_images": [str(p) for p in getattr(dls.train_ds, 'items', [])],
            "valid_images": [str(p) for p in getattr(dls.valid_ds, 'items', [])],
            "counts": {
                "train": int(len(getattr(dls.train_ds, 'items', []))),
                "valid": int(len(getattr(dls.valid_ds, 'items', [])))
            }
        }
        with open(splits_dir / "splits.json", 'w') as f:
            json.dump(split_manifest, f, indent=2)
        logger.info(f"Wrote split manifest to {splits_dir / 'splits.json'}")
    except Exception as _e:
        logger.warning(f"Could not write split manifest: {_e}")
    
    logger.info(f"Data loaded: {len(dls.train_ds)} train, {len(dls.valid_ds)} val samples")
    
    # Create learner (binary segmentation with n_out=1 and proper [0,1] normalization)
    learn = unet_learner(
        dls,
        resnet34,
        n_out=2,  # 2 classes: background (0) + mitochondria (1)
        metrics=Dice,  # Standard Dice metric works with multiclass!
    )
    
    # FastAI automatically sets CrossEntropyLossFlat for n_out=2, don't override
    print(f"Using default loss function: {learn.loss_func}")
    
    # Train the model with callbacks for visualization
    logger.info(f"Training for {epochs} epochs...")
    
    # Add callbacks for training visualization
    from fastai.callback.tracker import SaveModelCallback
    
    # Save best model during training (in the output directory with descriptive name)
    save_callback = SaveModelCallback(monitor='valid_loss', fname=f'{model_name}_best_model', with_opt=False)
    
    # Train with callbacks
    learn.fit_one_cycle(epochs, lr_max=learning_rate, cbs=[save_callback])
    
    # Generate training visualizations
    logger.info("Generating training visualizations...")
    
    # Plot training history
    try:
        learn.recorder.plot_loss()
        plt.savefig(output_path / "training_loss.png", dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Training loss plot saved to: {output_path / 'training_loss.png'}")
    except Exception as e:
        logger.warning(f"Could not save loss plot: {e}")
    
    # Plot learning rate schedule (FastAI v2 compatible)
    try:
        if hasattr(learn.recorder, 'lrs') and learn.recorder.lrs:
            plt.figure(figsize=(10, 6))
            plt.plot(learn.recorder.lrs)
            plt.title('Learning Rate Schedule')
            plt.xlabel('Batch')
            plt.ylabel('Learning Rate')
            plt.grid(True)
            plt.savefig(output_path / "learning_rate_schedule.png", dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"Learning rate plot saved to: {output_path / 'learning_rate_schedule.png'}")
        else:
            logger.info("Learning rate data not available for plotting")
    except Exception as e:
        logger.warning(f"Could not save LR plot: {e}")
    
    # Plot metrics (FastAI v2 compatible)
    try:
        if hasattr(learn.recorder, 'values') and learn.recorder.values:
            plt.figure(figsize=(12, 8))
            
            # Extract metrics data
            values = learn.recorder.values
            epoch_range = range(1, len(values) + 1)
            
            # Plot loss
            plt.subplot(2, 2, 1)
            train_losses = [v[0] for v in values]
            val_losses = [v[1] for v in values] if len(values[0]) > 1 else []
            plt.plot(epoch_range, train_losses, label='Training Loss', color='blue')
            if val_losses:
                plt.plot(epoch_range, val_losses, label='Validation Loss', color='red')
            plt.title('Training and Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            
            # Plot metrics if available
            if len(values[0]) > 2:
                plt.subplot(2, 2, 2)
                metrics = [v[2] for v in values]
                plt.plot(epoch_range, metrics, label='Dice Score', color='green')
                plt.title('Dice Score')
                plt.xlabel('Epoch')
                plt.ylabel('Score')
                plt.legend()
                plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(output_path / "training_metrics.png", dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"Training metrics plot saved to: {output_path / 'training_metrics.png'}")
        else:
            logger.info("Metrics data not available for plotting")
    except Exception as e:
        logger.warning(f"Could not save metrics plot: {e}")
    
    # Show some predictions on validation set
    try:
        # Get a few validation samples
        val_dl = learn.dls.valid
        batch = next(iter(val_dl))
        images, masks = batch
        
        # Make predictions
        with learn.no_bar():
            preds = learn.get_preds(dl=val_dl.new(shuffle=False, drop_last=False))
        
        # Plot first few predictions
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        
        # ImageNet normalization stats for denormalization
        from fastai.vision.all import imagenet_stats
        imagenet_mean = np.array(imagenet_stats[0])
        imagenet_std = np.array(imagenet_stats[1])
        
        for i in range(min(3, len(images))):
            # Original image - denormalize from ImageNet normalization
            img_denorm = images[i].permute(1, 2, 0).cpu().numpy()
            img_denorm = img_denorm * imagenet_std + imagenet_mean
            img_denorm = np.clip(img_denorm, 0, 1)  # Ensure valid range for matplotlib
            axes[i, 0].imshow(img_denorm)
            axes[i, 0].set_title(f'Image {i+1}')
            axes[i, 0].axis('off')
            
            # Ground truth mask
            axes[i, 1].imshow(masks[i].cpu().numpy(), cmap='gray')
            axes[i, 1].set_title(f'Ground Truth {i+1}')
            axes[i, 1].axis('off')
            
            # Prediction
            pred_mask = preds[0][i].argmax(dim=0).cpu().numpy()
            axes[i, 2].imshow(pred_mask, cmap='gray')
            axes[i, 2].set_title(f'Prediction {i+1}')
            axes[i, 2].axis('off')
            
            # Overlay - use denormalized image
            overlay = img_denorm.copy()
            pred_overlay = pred_mask > 0
            overlay[pred_overlay] = [1, 0, 0]  # Red overlay for predictions
            axes[i, 3].imshow(overlay)
            axes[i, 3].set_title(f'Overlay {i+1}')
            axes[i, 3].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path / "validation_predictions.png", dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Validation predictions saved to: {output_path / 'validation_predictions.png'}")
    except Exception as e:
        logger.warning(f"Could not save validation predictions: {e}")
    
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
    
    # Save the model (parameterized name)
    def _fmt_suffix2(e, b, lr, sz, tag=""):
        lr_str = (f"{lr:.0e}" if lr < 1e-2 else f"{lr:.3f}").replace("-0", "-")
        parts = [f"e{e}", f"b{b}", f"lr{lr_str}", f"sz{sz}"]
        if tag:
            parts.insert(0, tag)
        return "_".join(parts)
    run_tag2 = _fmt_suffix2(epochs, batch_size, learning_rate, image_size, tag="pretrain")
    model_path = output_path / f"{model_name}-{run_tag2}.pkl"
    learn.export(model_path)
    logger.info(f"Model saved to: {model_path}")
    
    # Save training history as JSON
    try:
        import json
        training_history = {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate
        }
        
        # Safely add recorder data if available using FastAI v2 attributes
        if hasattr(learn, 'recorder') and learn.recorder:
            if hasattr(learn.recorder, 'log') and learn.recorder.log:
                training_history['log'] = [float(x) if isinstance(x, (int, float)) else str(x) for x in learn.recorder.log]
            
            if hasattr(learn.recorder, 'values') and learn.recorder.values:
                training_history['values'] = [[float(v) if isinstance(v, (int, float)) else str(v) for v in row] for row in learn.recorder.values]
            
            if hasattr(learn.recorder, 'losses') and learn.recorder.losses:
                training_history['losses'] = [float(x) for x in learn.recorder.losses]
            
            if hasattr(learn.recorder, 'metric_names') and learn.recorder.metric_names:
                training_history['metric_names'] = list(learn.recorder.metric_names)
        
        with open(output_path / "training_history.json", 'w') as f:
            json.dump(training_history, f, indent=2)
        logger.info(f"Training history saved to: {output_path / 'training_history.json'}")
    except Exception as e:
        logger.warning(f"Could not save training history: {e}")
    
    # Return the learner for now (can be wrapped later if needed)
    return learn, model_path


def main():
    """CLI interface for mitochondria training."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train mitochondria segmentation model')
    parser.add_argument('--config', help='Optional YAML config file to override defaults')
    parser.add_argument('--data-dir', required=True, help='Directory containing image_patches/ and mask_patches/')
    parser.add_argument('--model-dir', default=DEFAULT_MITOCHONDRIA_MODEL_DIR, help='Directory to save trained model')
    parser.add_argument('--model-name', default='mitochondria_model', help='Base name for saved model files')
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
            # training hyperparams
            model_cfg = cfg_yaml.get('model', {}) if isinstance(cfg_yaml.get('model'), dict) else {}
            training_cfg = model_cfg.get('training', {}) if isinstance(model_cfg.get('training'), dict) else {}
            if 'epochs' in training_cfg and not parser.get_default('epochs') == args.epochs:
                args.epochs = int(training_cfg['epochs'])
            if 'batch_size' in training_cfg and not parser.get_default('batch_size') == args.batch_size:
                args.batch_size = int(training_cfg['batch_size'])
            if 'learning_rate' in training_cfg and not parser.get_default('learning_rate') == args.learning_rate:
                args.learning_rate = float(training_cfg['learning_rate'])
            # output model dir from checkpoint_path
            if 'checkpoint_path' in cfg_yaml.get('model', {}):
                from pathlib import Path as _P
                ckpt = _P(cfg_yaml['model']['checkpoint_path'])
                args.model_dir = str(ckpt.parent)
                # Only take name from YAML if CLI didn't override
                if parser.get_default('model_name') == args.model_name:
                    args.model_name = ckpt.stem
        except Exception as _e:  # pragma: no cover
            print(f"⚠️  Failed to load config {args.config}: {_e}")
    
    try:
        logger = get_logger("eq.mitochondria_training")
        logger.info("🚀 Starting mitochondria model training...")
        logger.info(f"📁 Data directory: {args.data_dir}")
        logger.info(f"📁 Model directory: {args.model_dir}")
        logger.info(f"⚙️  Epochs: {args.epochs}, Batch size: {args.batch_size}")
        
        # Train the model using DataBlock approach
        model, model_path = train_mitochondria_with_datablock(
            data_dir=args.data_dir,
            output_dir=args.model_dir,
            model_name=args.model_name,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            image_size=args.image_size
        )
        
        logger.info("🎉 Mitochondria training completed successfully!")
        print(f"✅ Model saved to: {model_path}")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        print(f"❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
