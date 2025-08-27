#!/usr/bin/env python3
"""
Retrain Glomeruli Model Using Original Approach - FastAI v2 Compatible

This script reproduces the exact training approach used in the original git commits,
adapted for FastAI v2 syntax and API.
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

logger = get_logger("eq.retrain_glomeruli_original")

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

def get_glom_mask_file(image_file, p2c, thresh=127):
    """Get glomeruli mask file with color mapping."""
    # For derived data, mask is in the mask_patches directory with '_mask' suffix
    mask_path = image_file.parent.parent / "mask_patches" / f"{image_file.stem}_mask{image_file.suffix}"
    
    # Convert to an array (mask)
    msk = np.array(PILMask.create(mask_path))
    # Derived data should already be binary, but ensure it's 0/1
    msk = (msk > thresh).astype(np.uint8)
    return PILMask.create(msk)

def get_glom_y(o):
    """Get glomeruli mask for a given image file."""
    return get_glom_mask_file(o, p2c)

def retrain_glomeruli_original():
    """Retrain glomeruli model using the original approach."""
    logger.info("🔄 Starting glomeruli retraining using original approach")
    
    # Set up paths (using derived data that already has train/test splits)
    project_directory = Path.cwd()
    train_mask_path = project_directory / 'derived_data/glomeruli_data/training/mask_patches'
    train_image_path = project_directory / 'derived_data/glomeruli_data/training/image_patches'
    
    # Load data paths
    logger.info("📁 Loading data paths...")
    glom_mask_files = get_all_paths(train_mask_path)
    glom_image_files = get_all_paths(train_image_path)
    
    logger.info(f"Found {len(glom_image_files)} images and {len(glom_mask_files)} masks")
    assert len(glom_image_files) <= len(glom_mask_files)
    
    # Get color codes
    logger.info("🎨 Determining color codes...")
    p2c = n_glom_codes(glom_mask_files)
    logger.info(f"Color codes: {p2c}")
    
    # Set up namespace for model loading (required for FastAI v1 models)
    logger.info("🔧 Setting up namespace for model loading...")
    import __main__
    __main__.__dict__['get_y'] = get_glom_y
    __main__.__dict__['get_glom_mask_file'] = get_glom_mask_file
    __main__.__dict__['n_glom_codes'] = n_glom_codes
    __main__.__dict__['p2c'] = p2c
    
    # Load pretrained mitochondria model
    logger.info("🧠 Loading pretrained mitochondria model...")
    mito_model_path = project_directory / "backups/mito_dynamic_unet_seg_model-e50_b16.pkl"
    if not mito_model_path.exists():
        raise FileNotFoundError(f"Mitochondria model not found: {mito_model_path}")
    
    # Use safer loading method to avoid pickle security warnings
    try:
        # Try to use the safer Learner.load method first
        from fastai.learner import Learner
        segmentation_model = Learner.load(mito_model_path, with_opt=False)
        logger.info("✅ Loaded pretrained mitochondria model using safe method")
    except Exception as e:
        logger.warning(f"Safe loading failed ({e}), falling back to load_learner")
        # Fallback to load_learner if safe method fails
        segmentation_model = load_learner(mito_model_path)
        logger.info("✅ Loaded pretrained mitochondria model using fallback method")
    logger.info("✅ Loaded pretrained mitochondria model")
    
    # Set up data augmentation (matching original)
    logger.info("🔄 Setting up data augmentation...")
    gpt_rec_batch_aug = [*aug_transforms(size=256,  # 256x256 output
                                       flip_vert=True,
                                       max_rotate=45,
                                       min_zoom=0.8,
                                       max_zoom=1.3,
                                       max_warp=0.4,
                                       max_lighting=0.2),
                       RandomErasing(p=0.5, sl=0.01, sh=0.3, min_aspect=0.3, max_count=3)]
    
    # Create DataBlock (matching original)
    logger.info("📊 Creating DataBlock...")
    
    # Debug: Check what we have
    logger.info(f"glom_image_files type: {type(glom_image_files)}")
    logger.info(f"glom_image_files length: {len(glom_image_files)}")
    logger.info(f"First few image files: {glom_image_files[:3]}")
    logger.info(f"p2c: {p2c}")
    
    # Create a closure that captures the p2c value
    def get_glom_y_with_p2c(o):
        logger.info(f"get_glom_y_with_p2c called with: {o}")
        result = get_glom_mask_file(o, p2c)
        logger.info(f"get_glom_y_with_p2c result: {type(result)}")
        return result
    
    # Test the function on a sample
    try:
        test_result = get_glom_y_with_p2c(glom_image_files[0])
        logger.info(f"Test call successful: {type(test_result)}")
    except Exception as e:
        logger.error(f"Test call failed: {e}")
        raise
    
    gloms = DataBlock(blocks=(ImageBlock, MaskBlock(codes=np.array(['not_glom', 'glom']))),
                     splitter=RandomSplitter(valid_pct=0.2, seed=42),
                     get_items=lambda x: glom_image_files,
                     get_y=get_glom_y_with_p2c,
                     item_tfms=[RandomResizedCrop(512, min_scale=0.45)],  # 512x512 crop, then resize
                     batch_tfms=gpt_rec_batch_aug,
                     n_inp=1)
    
    # Create dataloaders
    batch_size = 16
    glom_dls = gloms.dataloaders(glom_image_files, bs=batch_size)
    logger.info(f"Created dataloaders: {len(glom_dls.train_ds)} train, {len(glom_dls.valid_ds)} val")
    
    # Show batch to verify
    logger.info("👀 Showing sample batch...")
    glom_dls.show_batch(max_n=4, vmin=0, vmax=1, figsize=(8, 8))
    plt.savefig('debug_original_batch.png')
    plt.close()
    
    # Set up model for transfer learning
    logger.info("🔄 Setting up transfer learning...")
    segmentation_model.dls = glom_dls
    
    # Freeze the model except the last layer(s)
    logger.info("🔒 Freezing pretrained layers...")
    segmentation_model.freeze()
    
    # Find optimal learning rate for head training
    logger.info("🔍 Finding optimal learning rate for head...")
    
    # Suppress NumPy deprecation warnings during lr_find
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="numpy")
        lr_min, lr_steep, lr_valley, lr_slide = segmentation_model.lr_find(suggest_funcs=(minimum, steep, valley, slide))
    
    logger.info(f"Learning rates - min: {lr_min:.2e}, steep: {lr_steep:.2e}, valley: {lr_valley:.2e}")
    
    # Train the last layer(s) using fit_one_cycle
    n_epochs_head = 15
    lr_max_head = lr_min
    logger.info(f"🎯 Training head for {n_epochs_head} epochs with LR {lr_max_head:.2e}")
    
    segmentation_model.fit_one_cycle(n_epochs_head, lr_max_head,
                                    cbs=[EarlyStoppingCallback(monitor='valid_loss', min_delta=0.001, patience=5),
                                         SaveModelCallback(monitor='valid_loss', fname='best_model')])
    
    # Plot training history
    segmentation_model.recorder.plot_loss()
    plt.savefig('debug_head_training_loss.png')
    plt.close()
    
    # Unfreeze the entire model
    logger.info("🔓 Unfreezing all layers...")
    segmentation_model.unfreeze()
    
    # Find optimal learning rate for full fine-tuning
    logger.info("🔍 Finding optimal learning rate for full fine-tuning...")
    lr_min, lr_steep, lr_valley, lr_slide = segmentation_model.lr_find(suggest_funcs=(minimum, steep, valley, slide))
    logger.info(f"Learning rates - min: {lr_min:.2e}, steep: {lr_steep:.2e}, valley: {lr_valley:.2e}")
    
    # Full fine-tuning
    n_epochs = 50
    my_lr_max = 5e-4  # From original notebook
    logger.info(f"🎯 Full fine-tuning for {n_epochs} epochs with LR {my_lr_max:.2e}")
    
    segmentation_model.fit_one_cycle(n_epochs, my_lr_max,
                                    cbs=[EarlyStoppingCallback(monitor='valid_loss', min_delta=0.0001, patience=5),
                                         SaveModelCallback(monitor='valid_loss', fname='best_model')])
    
    # Plot final training history
    segmentation_model.recorder.plot_loss()
    plt.savefig('debug_full_training_loss.png')
    plt.close()
    
    # Show results
    logger.info("📊 Showing final results...")
    segmentation_model.show_results(max_n=6, figsize=(2, 6))
    plt.savefig('debug_final_results.png')
    plt.close()
    
    # Print final metrics safely
    logger.info("Final metrics:")
    try:
        names = segmentation_model.recorder.metric_names
        vals = segmentation_model.recorder.log
        last = vals[-1] if len(vals) > 0 else None
        logger.info(f"  Names: {names}")
        logger.info(f"  Last row: {last}")
        if last is not None:
            # Typically: train_loss, valid_loss, dice, time
            if len(last) >= 1:
                logger.info(f"  Training loss: {last[0]:.4f}")
            if len(last) >= 2:
                logger.info(f"  Validation loss: {last[1]:.4f}")
            if len(last) >= 3 and isinstance(last[2], (float, int)):
                logger.info(f"  Dice Coef: {last[2]:.4f}")
    except Exception as e:
        logger.warning(f"Could not parse metrics from recorder: {e}")
    
    # Save the model
    logger.info("💾 Saving model...")
    output_dir = project_directory / "models/segmentation/glomeruli_retrained"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fname = f"glomerulus_segmentation_model-dynamic_unet-e{n_epochs}_b{batch_size}_s{len(glom_image_files)}_retrained.pkl"
    output_file = output_dir / fname
    logger.info(f"Saving to: {output_file}")
    
    # Save the whole model
    segmentation_model.export(output_file)
    logger.info("✅ Model saved successfully!")
    
    return str(output_file)

if __name__ == "__main__":
    try:
        model_path = retrain_glomeruli_original()
        print("\n🎉 GLOMERULI RETRAINING COMPLETED SUCCESSFULLY!")
        print(f"📁 Model saved to: {model_path}")
        print("🔍 Check the debug_*.png files for training visualizations")
    except Exception as e:
        logger.error(f"❌ Retraining failed: {e}")
        raise
