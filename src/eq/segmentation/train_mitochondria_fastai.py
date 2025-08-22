#!/usr/bin/env python3
"""
Simple Mitochondria Training with FastAI

This is a clean, straightforward implementation for training mitochondria models.
"""

import pickle
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from fastai.callback.all import *
from fastai.vision.all import *

from eq.segmentation.fastai_segmenter import FastaiSegmenter
from eq.utils.logger import get_logger


def get_y(x):
    """Get mask path for image path."""
    return str(x).replace('.jpg', '.png').replace('img_', 'mask_')


def train_mitochondria_model(
    train_images: np.ndarray,
    train_masks: np.ndarray,
    val_images: np.ndarray,
    val_masks: np.ndarray,
    output_dir: str,
    model_name: str,
    batch_size: int = 16,
    epochs: int = 50,
    learning_rate: float = 1e-3,
    image_size: int = 224
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
    
    # Create DataBlock
    data_block = DataBlock(
        blocks=(ImageBlock, MaskBlock(codes=['background', 'mitochondria'])),
        get_items=lambda x: x,
        get_y=get_y,
        splitter=RandomSplitter(valid_pct=0.0, seed=42),  # We already have split
        item_tfms=[Resize(image_size)],
        batch_tfms=[Normalize.from_stats(*imagenet_stats)]
    )
    
    # Create DataLoaders with our predefined split
    all_img_files = train_img_files + val_img_files
    all_mask_files = train_mask_files + val_mask_files
    
    # Custom splitter that uses our predefined train/val split
    def custom_splitter(items):
        n_train = len(train_img_files)
        return (list(range(n_train)), list(range(n_train, len(items))))
    
    # Override the splitter
    data_block.splitter = custom_splitter
    
    # Create dataloaders
    print("📊 Creating DataLoaders...")
    logger.info("Creating DataLoaders...")
    dls = data_block.dataloaders(all_img_files, bs=batch_size)
    
    logger.info(f"Created dataloaders: {len(dls.train_ds)} train, {len(dls.valid_ds)} val")
    print(f"✅ DataLoaders created: {len(dls.train_ds)} train, {len(dls.valid_ds)} val samples")
    
    # Create U-Net model
    logger.info("Creating U-Net model...")
    print("🏗️  Creating U-Net model...")
    
    # No custom logging callback - let fastai handle all the progress display
    
    learn = unet_learner(
        dls, 
        resnet34, 
        metrics=Dice,
        opt_func=Adam,
        pretrained=False,  # Start from scratch for mitochondria
    )
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
            
            # Prediction
            pred_mask = preds[i].argmax(dim=0) if preds[i].dim() > 2 else preds[i]
            axes[i, 2].imshow(pred_mask, cmap='gray')
            axes[i, 2].set_title(f'Prediction {i+1}')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(predictions_plot, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Sample predictions plot saved to {predictions_plot}")
        
    except Exception as e:
        logger.warning(f"Could not save sample predictions plot: {e}")
    
    # Save the model first (define model_path before using it)
    if QUICK_TEST:
        model_filename = f"{model_name}_TESTING_RUN.pkl"
        logger.info("TESTING RUN: Model will be saved with TESTING_RUN suffix")
    else:
        model_filename = f"{model_name}.pkl"
    
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
            f.write(f"Training history saved to: {history_path}\n")
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
    
    return learn


def train_mitochondria_from_cache(
    cache_dir: str,
    output_dir: str,
    model_name: str,
    batch_size: int = 16,
    epochs: int = 50,
    learning_rate: float = 1e-3,
    image_size: int = 224
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
