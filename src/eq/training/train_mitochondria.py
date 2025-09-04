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

from eq.models.fastai_segmenter import FastaiSegmenter
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
from fastai.losses import BCEWithLogitsLossFlat


# Use standardized getter function for compatibility
get_y = get_y_mitochondria


def train_mitochondria_model(
    train_images: np.ndarray,
    train_masks: np.ndarray,
    val_images: np.ndarray,
    val_masks: np.ndarray,
    output_dir: str,
    model_name: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    image_size: int = DEFAULT_IMAGE_SIZE
) -> FastaiSegmenter:
    """
    Train a mitochondria segmentation model.
    
    Args:
        train_images: Training images (N, H, W, C)
        train_masks: Training masks (N, H, W, C) 
        val_images: Validation images (N, H, W, C)
        val_masks: Validation masks (N, H, W, C)
        output_dir: Directory to save model and results
        model_name: Name for the model
        batch_size: Training batch size
        epochs: Number of training epochs
        learning_rate: Learning rate
        image_size: Input image size
        
    Returns:
        Trained FastaiSegmenter instance
    """
    logger = get_logger("eq.mitochondria_training")
    logger.info("Starting mitochondria model training...")
    
    # Create output directory
    output_path = Path(output_dir) / model_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create temporary directory for fastai data
    temp_dir = Path("/tmp/mitochondria_training")
    temp_dir.mkdir(exist_ok=True)
    
    logger.info("Preparing data for fastai...")
    
    # Save images and masks as temporary files
    train_img_files = []
    train_mask_files = []
    
    for i in range(len(train_images)):
        img_path = temp_dir / f"train_img_{i:04d}.jpg"
        mask_path = temp_dir / f"train_mask_{i:04d}.png"
        
        # Save image (convert to 3-channel if needed)
        img = train_images[i]
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        img_uint8 = (img * 255).astype(np.uint8)
        PILImage.create(img_uint8).save(img_path)
        
        # Save mask
        mask = train_masks[i]
        mask_uint8 = (mask * 255).astype(np.uint8)
        # Fix PIL format issue - ensure 2D array
        if mask_uint8.ndim == 3 and mask_uint8.shape[2] == 1:
            mask_uint8 = mask_uint8.squeeze()
        PILMask.create(mask_uint8).save(mask_path)
        
        train_img_files.append(img_path)
        train_mask_files.append(mask_path)
    
    # Save validation data
    val_img_files = []
    val_mask_files = []
    
    for i in range(len(val_images)):
        img_path = temp_dir / f"val_img_{i:04d}.jpg"
        mask_path = temp_dir / f"val_mask_{i:04d}.png"
        
        # Save image
        img = val_images[i]
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        img_uint8 = (img * 255).astype(np.uint8)
        PILImage.create(img_uint8).save(img_path)
        
        # Save mask
        mask = val_masks[i]
        mask_uint8 = (mask * 255).astype(np.uint8)
        # Fix PIL format issue - ensure 2D array
        if mask_uint8.ndim == 3 and mask_uint8.shape[2] == 1:
            mask_uint8 = mask_uint8.squeeze()
        PILMask.create(mask_uint8).save(mask_path)
        
        val_img_files.append(img_path)
        val_mask_files.append(mask_path)
    
    logger.info(f"Saved {len(train_img_files)} training and {len(val_img_files)} validation samples")
    
    # Create DataBlock using the new v2 API
    logger.info("Creating DataLoaders using FastAI v2 DataBlock...")
    print("📊 Creating DataLoaders using FastAI v2 DataBlock...")
    
    # Use the new canonical API
    dls = build_segmentation_dls(temp_dir, bs=batch_size, num_workers=0)
    
    logger.info(f"Created dataloaders: {len(dls.train_ds)} train, {len(dls.valid_ds)} val")
    print(f"✅ DataLoaders created: {len(dls.train_ds)} train, {len(dls.valid_ds)} val samples")
    
    # Create U-Net model with v2 n_out parameter
    logger.info("Creating U-Net model...")
    print("🏗️  Creating U-Net model...")
    
    # No custom logging callback - let fastai handle all the progress display
    
    learn = unet_learner(
        dls, 
        resnet34, 
        n_out=1,  # FastAI v2 requires n_out parameter for binary segmentation
        metrics=Dice,
        opt_func=Adam,
        pretrained=False,  # Start from scratch for mitochondria
    )
    
    # Set the correct loss function for binary segmentation
    learn.loss_func = BCEWithLogitsLossFlat()  # type: ignore
    
    print("✅ U-Net model created")
    logger.info("U-Net model created successfully")
    
    # Check if we have a cached optimal learning rate
    lr_cache_file = output_path / "optimal_lr.txt"
    final_lr = None
    
    import os
    QUICK_TEST = os.getenv('QUICK_TEST', 'false').lower() == 'true'
    
    if QUICK_TEST:
        # For QUICK_TEST mode, just use the default learning rate
        # No learning rate finding during testing - just get on with training!
        final_lr = float(learning_rate)
        logger.info(f"QUICK_TEST mode: Using default learning rate {final_lr}")
        print(f"🎯 QUICK_TEST mode: Using default learning rate {final_lr}")
    else:
        if lr_cache_file.exists():
            try:
                with open(lr_cache_file, 'r') as f:
                    cached_lr = float(f.read().strip())
                logger.info(f"Using cached optimal learning rate: {cached_lr}")
                print(f"🎯 Using cached optimal learning rate: {cached_lr}")
                final_lr = cached_lr
            except Exception as e:
                logger.warning(f"Failed to read cached LR: {e}")
        
        # If no cached LR, find optimal learning rate
        if final_lr is None:
            logger.info("Finding optimal learning rate...")
            print("🔍 Finding optimal learning rate...")
            learn.lr_find()
            logger.info("Learning rate finder completed")
            
            # Use the correct method to get suggested learning rate
            try:
                suggested_lr = learn.recorder.suggestion()
            except AttributeError:
                # Fallback: use the default learning rate if suggestion not available
                suggested_lr = None
                logger.warning("Could not get suggested learning rate, using default")
            
            logger.info(f"Suggested learning rate: {suggested_lr}")
            
            if suggested_lr:
                print(f"🎯 Suggested learning rate: {suggested_lr}")
                logger.info(f"Suggested learning rate: {suggested_lr}")
                final_lr = suggested_lr
            else:
                print(f"🎯 Using default learning rate: {learning_rate}")
                logger.info(f"Using default learning rate: {learning_rate}")
                final_lr = learning_rate
            
            # Cache the optimal learning rate for future use
            try:
                with open(lr_cache_file, 'w') as f:
                    f.write(str(final_lr))
                logger.info(f"Cached optimal learning rate: {final_lr}")
            except Exception as e:
                logger.warning(f"Failed to cache learning rate: {e}")
        
        # Ensure learning rate is a float
        if final_lr is None:
            final_lr = float(learning_rate)
        else:
            final_lr = float(final_lr)
    
    # Train the model with progress bars visible
    logger.info(f"Training for {epochs} epochs with learning rate {final_lr}")
    print(f"\n🚀 Starting training: {epochs} epochs, batch size {batch_size}")
    print("=" * 50)
    
    # Log training start
    logger.info("Training started - fit_one_cycle")
    
    # Force console output to be visible during training
    import sys
    sys.stdout.flush()
    
    # Explicitly set the n_epoch property to ensure fastai uses our epoch count
    learn.n_epoch = epochs
    
    # Double-check that we're using the correct epoch count
    print(f"🔍 Confirmed: Training for exactly {epochs} epochs")
    logger.info(f"Confirmed training configuration: {epochs} epochs, batch size {batch_size}")
    
    # Double-check that fastai will use our epoch count
    print(f"🔍 FastAI n_epoch set to: {learn.n_epoch}")
    logger.info(f"FastAI n_epoch set to: {learn.n_epoch}")
    
    # Final verification - ensure we're using the correct epochs
    if learn.n_epoch != epochs:
        print(f"⚠️  WARNING: FastAI n_epoch ({learn.n_epoch}) != requested epochs ({epochs})")
        logger.warning(f"FastAI n_epoch ({learn.n_epoch}) != requested epochs ({epochs})")
        # Force the correct epoch count
        learn.n_epoch = epochs
        print(f"🔧 Forced FastAI n_epoch to: {learn.n_epoch}")
        logger.info(f"Forced FastAI n_epoch to: {learn.n_epoch}")
    
    # Train using fit_one_cycle - this is NOT for finding learning rate!
    # fit_one_cycle is a training strategy that uses cyclical learning rate scheduling
    # and momentum scheduling for better convergence. The learning rate was already
    # found and cached above.
    learn.fit_one_cycle(epochs, final_lr)
    
    # Force output after training
    sys.stdout.flush()
    print()  # Add blank line after training
    
    # Log training completion
    logger.info("Training completed successfully")
    print("=" * 50)
    print("✅ Training completed!")
    
    # Generate diagnostic plots
    logger.info("Generating diagnostic plots...")
    
    # 1. Learning rate finder plot
    lr_finder_plot = output_path / "lr_finder.png"
    try:
        learn.recorder.plot_lr_find()
        if QUICK_TEST:
            plt.title('TESTING RUN - Learning Rate Finder')
        plt.savefig(lr_finder_plot, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Learning rate finder plot saved to {lr_finder_plot}")
    except Exception as e:
        logger.warning(f"Could not save learning rate finder plot: {e}")
    
    # 2. Training curves (loss and metrics)
    training_curves_plot = output_path / "training_curves.png"
    try:
        learn.recorder.plot_loss()
        if QUICK_TEST:
            plt.title('TESTING RUN - Training Curves')
        plt.savefig(training_curves_plot, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Training curves plot saved to {training_curves_plot}")
    except Exception as e:
        logger.warning(f"Could not save training curves plot: {e}")
    
    # 3. Sample predictions on validation set
    predictions_plot = output_path / "sample_predictions.png"
    try:
        # Get a few validation samples
        val_batch = next(iter(learn.dls.valid))
        images, masks = val_batch
        
        # Make predictions
        with learn.no_bar():
            preds = learn.predict(images[:4])  # Predict on first 4 images
        # FastAI v2 predict returns (prediction, target, loss), we want just the prediction
        if isinstance(preds, tuple):
            preds = preds[0]
        
        # Ensure preds is a tensor and handle None case
        if preds is None:
            preds = torch.zeros((4, 1, images[0].shape[-2], images[0].shape[-1]))
        elif not isinstance(preds, torch.Tensor):
            preds = torch.tensor(preds)
        
        # Create a grid of images, masks, and predictions
        fig, axes = plt.subplots(4, 3, figsize=(12, 16))
        
        # Add testing indicator to plot title
        if QUICK_TEST:
            fig.suptitle('TESTING RUN - Sample Predictions: Image | Ground Truth | Prediction', fontsize=16)
        else:
            fig.suptitle('Sample Predictions: Image | Ground Truth | Prediction', fontsize=16)
        
        for i in range(4):
            # Original image - handle different tensor shapes safely
            img_tensor = images[i]
            if img_tensor.dim() == 3:  # (C, H, W)
                img_display = img_tensor.permute(1, 2, 0)
            elif img_tensor.dim() == 2:  # (H, W)
                img_display = img_tensor
            else:
                img_display = img_tensor.squeeze()
            
            axes[i, 0].imshow(img_display, cmap='gray')
            axes[i, 0].set_title(f'Image {i+1}')
            axes[i, 0].axis('off')
            
            # Ground truth mask
            axes[i, 1].imshow(masks[i].squeeze(), cmap='gray')
            axes[i, 1].set_title(f'Ground Truth {i+1}')
            axes[i, 1].axis('off')
            
            # Prediction - for binary segmentation, use sigmoid and threshold
            if i < len(preds) and preds[i] is not None:
                pred_mask = (preds[i].sigmoid() > DEFAULT_PREDICTION_THRESHOLD).float()
            else:
                pred_mask = torch.zeros_like(masks[i])
            axes[i, 2].imshow(pred_mask, cmap='gray')
            axes[i, 2].set_title(f'Prediction {i+1}')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(predictions_plot, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Sample predictions plot saved to {predictions_plot}")
        
    except Exception as e:
        logger.warning(f"Could not save sample predictions plot: {e}")
    
    # Save the model first (define model_path before using it) with parameterized name
    def _fmt_suffix(e, b, lr, sz, tag=""):
        lr_str = (f"{lr:.0e}" if lr < 1e-2 else f"{lr:.3f}").replace("-0", "-")
        parts = [f"e{e}", f"b{b}", f"lr{lr_str}", f"sz{sz}"]
        if tag:
            parts.insert(0, tag)
        return "_".join(parts)

    # Always use production-safe tag names in models/segmentation
    run_tag = _fmt_suffix(epochs, batch_size, learning_rate, image_size, tag="pretrain")
    model_filename = f"{model_name}-{run_tag}.pkl"
    model_path = output_path / model_filename
    learn.export(model_path)
    logger.info(f"Model saved to {model_path}")
    
    # 4. Training metrics summary
    metrics_summary = output_path / "training_summary.txt"
    try:
        with open(metrics_summary, 'w') as f:
            f.write("Mitochondria Segmentation Training Summary\n")
            f.write("==========================================\n\n")
            
            # Add testing indicator if in QUICK_TEST mode
            if QUICK_TEST:
                f.write("TESTING RUN - QUICK_TEST MODE\n")
                f.write("This is a TESTING run with reduced epochs and parameters.\n")
                f.write("DO NOT use this model for production inference!\n\n")
            
            f.write(f"Model: {model_name}\n")
            f.write(f"Training samples: {len(train_images)}\n")
            f.write(f"Validation samples: {len(val_images)}\n")
            f.write(f"Epochs: {epochs}\n")
            f.write(f"Batch size: {batch_size}\n")
            f.write(f"Learning rate: {final_lr}\n")
            f.write(f"Image size: {image_size}\n\n")
            
            # Final metrics
            if hasattr(learn.recorder, 'values'):
                final_metrics = learn.recorder.values[-1] if learn.recorder.values else []
                f.write(f"Final training loss: {final_metrics[0] if len(final_metrics) > 0 else 'N/A'}\n")
                f.write(f"Final validation loss: {final_metrics[1] if len(final_metrics) > 1 else 'N/A'}\n")
                f.write(f"Final Dice score: {final_metrics[2] if len(final_metrics) > 2 else 'N/A'}\n")
            
            f.write(f"\nModel saved to: {model_path}\n")
            f.write(f"Training history saved to: {output_path / 'training_history.pkl'}\n")
            f.write(f"Diagnostic plots saved to: {output_path}\n")
        
        logger.info(f"Training summary saved to {metrics_summary}")
        
    except Exception as e:
        logger.warning(f"Could not save training summary: {e}")
    
    # Save training history
    history = {
        'loss': [float(x) for x in learn.recorder.losses],
        'metrics': learn.recorder.metrics if hasattr(learn.recorder, 'metrics') else [],
        'is_testing_run': QUICK_TEST,
        'testing_info': {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': final_lr,
            'timestamp': datetime.now().isoformat()
        } if QUICK_TEST else None
    }
    
    history_path = output_path / "training_history.pkl"
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    
    logger.info("Mitochondria training completed successfully!")
    
    # Clean up temp files
    import shutil
    shutil.rmtree(temp_dir)
    
    # Create a FastaiSegmenter wrapper for the trained learner
    from eq.models.fastai_segmenter import FastaiSegmenter, SegmentationConfig
    
    # Create a config object with the training parameters
    config = SegmentationConfig(
        image_size=image_size,
        batch_size=batch_size,
        learning_rate=final_lr,
        epochs=epochs
    )
    
    # Create segmenter and set the trained learner
    segmenter = FastaiSegmenter(config)
    segmenter.learn = learn  # Set the trained learner directly
    
    return segmenter


def train_mitochondria_from_cache(
    cache_dir: str,
    output_dir: str,
    model_name: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    image_size: int = DEFAULT_IMAGE_SIZE
) -> FastaiSegmenter:
    """
    Train mitochondria model from cached data.
    
    Args:
        cache_dir: Directory containing cached pickle files
        output_dir: Directory to save model and results
        model_name: Name for the model
        batch_size: Training batch size
        epochs: Number of training epochs
        learning_rate: Learning rate
        image_size: Input image size
        
    Returns:
        Trained model
    """
    logger = get_logger("eq.mitochondria_training")
    
    # Load cached data
    cache_path = Path(cache_dir)
    
    with open(cache_path / "train_images.pickle", 'rb') as f:
        train_images = pickle.load(f)
    with open(cache_path / "train_masks.pickle", 'rb') as f:
        train_masks = pickle.load(f)
    with open(cache_path / "val_images.pickle", 'rb') as f:
        val_images = pickle.load(f)
    with open(cache_path / "val_masks.pickle", 'rb') as f:
        val_masks = pickle.load(f)
    
    logger.info(f"Loaded data: {train_images.shape} train, {val_images.shape} val")
    
    # Train the model
    return train_mitochondria_model(
        train_images=train_images,
        train_masks=train_masks,
        val_images=val_images,
        val_masks=val_masks,
        output_dir=output_dir,
        model_name=model_name,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        image_size=image_size
    )


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
        Trained FastaiSegmenter instance
    """
    logger = get_logger("eq.mitochondria_training")
    logger.info("Starting mitochondria model training with DataBlock...")
    
    # Create output directory
    output_path = Path(output_dir) / model_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading data from: {data_dir}")
    
    # Build DataLoaders using DataBlock approach
    dls = build_segmentation_dls(data_dir, bs=batch_size, num_workers=0)
    
    logger.info(f"Data loaded: {len(dls.train_ds)} train, {len(dls.valid_ds)} val samples")
    
    # Create learner
    learn = unet_learner(dls, resnet34, n_out=2, metrics=DiceMulti())
    
    # Train the model with callbacks for visualization
    logger.info(f"Training for {epochs} epochs...")
    
    # Add callbacks for training visualization
    from fastai.callback.tracker import SaveModelCallback
    
    # Save best model during training
    save_callback = SaveModelCallback(monitor='valid_loss', fname='best_model')
    
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
    
    # Plot learning rate schedule
    try:
        learn.recorder.plot_lr()
        plt.savefig(output_path / "learning_rate_schedule.png", dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Learning rate plot saved to: {output_path / 'learning_rate_schedule.png'}")
    except Exception as e:
        logger.warning(f"Could not save LR plot: {e}")
    
    # Plot metrics
    try:
        learn.recorder.plot_metrics()
        plt.savefig(output_path / "training_metrics.png", dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Training metrics plot saved to: {output_path / 'training_metrics.png'}")
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
        for i in range(min(3, len(images))):
            # Original image
            axes[i, 0].imshow(images[i].permute(1, 2, 0).cpu().numpy())
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
            
            # Overlay
            overlay = images[i].permute(1, 2, 0).cpu().numpy().copy()
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
    return learn


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
        model = train_mitochondria_with_datablock(
            data_dir=args.data_dir,
            output_dir=args.model_dir,
            model_name=args.model_name,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            image_size=args.image_size
        )
        
        logger.info("🎉 Mitochondria training completed successfully!")
        print(f"✅ Model saved to: {args.model_dir}/{args.model_name}")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        print(f"❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
