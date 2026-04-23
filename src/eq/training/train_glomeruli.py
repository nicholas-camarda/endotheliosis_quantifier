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

import re
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from fastai.vision.all import *

from eq.utils.logger import get_logger
from eq.utils.run_io import (
    save_splits, attach_best_model_callback, save_plots, 
    save_training_history, save_run_metadata, export_final_model,
    load_supported_segmentation_artifact_metadata,
)
from eq.data_management.datablock_loader import (
    TRAINING_MODE_DYNAMIC_FULL_IMAGE,
    build_segmentation_dls_dynamic_patching,
    validate_supported_segmentation_training_root,
)
from eq.training.transfer_learning import transfer_learn_glomeruli
from eq.utils.hardware_detection import get_segmentation_training_batch_size


# BCEWithLogitsLossFlat import removed - using FastAI v2 automatic loss selection
from eq.core.constants import (
    DEFAULT_IMAGE_SIZE,
    DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, DEFAULT_LEARNING_RATE,
    DEFAULT_GLOMERULI_MODEL_DIR, DEFAULT_MITOCHONDRIA_MODEL_DIR,
    DEFAULT_POSITIVE_FOCUS_P, DEFAULT_MIN_POS_PIXELS, DEFAULT_POS_CROP_ATTEMPTS
)

logger = get_logger("eq.retrain_glomeruli_original")

def _format_run_suffix(epochs: int, batch_size: int, learning_rate: float, image_size: int, tag: str = "") -> str:
    """Create a concise, filesystem-safe suffix describing training params."""
    lr_str = (f"{learning_rate:.0e}" if learning_rate < 1e-2 else f"{learning_rate:.3f}").replace("-0", "-")
    parts = [f"e{epochs}", f"b{batch_size}", f"lr{lr_str}", f"sz{image_size}"]
    if tag:
        parts.insert(0, tag)
    return "_".join(parts)

def train_glomeruli_with_transfer_learning(
    data_dir: str,
    output_dir: str,
    model_name: str,
    base_model_path: Optional[str] = None,
    batch_size: Optional[int] = None,
    epochs: int = 30,  # Reduced from DEFAULT_EPOCHS for transfer learning efficiency
    learning_rate: float = 1e-3,  # Reduced from 5e-4 for proper transfer learning
    image_size: int = DEFAULT_IMAGE_SIZE,
    positive_focus_p: float = DEFAULT_POSITIVE_FOCUS_P,
    min_pos_pixels: int = DEFAULT_MIN_POS_PIXELS,
    pos_crop_attempts: int = DEFAULT_POS_CROP_ATTEMPTS,
    loss_name: Optional[str] = None,
    crop_size: Optional[int] = None,
    use_lr_find: bool = True,
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
    batch_size = get_segmentation_training_batch_size(
        "glomeruli",
        image_size=image_size,
        crop_size=crop_size or image_size,
        requested_batch_size=batch_size,
    )
    
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
    
    # Use transfer learning with positive-aware cropping
    learn = transfer_learn_glomeruli(
        base_model_path=base_model_path,
        glomeruli_data_dir=data_dir,
        output_dir=output_dir,
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        image_size=image_size,
        positive_focus_p=positive_focus_p,
        min_pos_pixels=min_pos_pixels,
        pos_crop_attempts=pos_crop_attempts,
        loss_name=loss_name,
        crop_size=crop_size,
        use_lr_find=use_lr_find,
    )
    
    return learn


def train_glomeruli_with_datablock(
    data_dir: str,
    output_dir: str,
    model_name: str,
    base_model_path: Optional[str] = None,
    batch_size: Optional[int] = None,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    image_size: int = DEFAULT_IMAGE_SIZE,
    config_path: Optional[str] = None,
    positive_focus_p: float = DEFAULT_POSITIVE_FOCUS_P,
    min_pos_pixels: int = DEFAULT_MIN_POS_PIXELS,
    pos_crop_attempts: int = DEFAULT_POS_CROP_ATTEMPTS,
):
    """
    Train glomeruli segmentation model using FastAI v2 DataBlock approach with transfer learning.
    
    Args:
        data_dir: Full-image training root containing images/ and masks/
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
    data_root = validate_supported_segmentation_training_root(data_dir, stage="glomeruli")
    batch_size = get_segmentation_training_batch_size(
        "glomeruli",
        image_size=image_size,
        crop_size=image_size,
        requested_batch_size=batch_size,
    )
    
    # Create output directory with descriptive model name
    approach = "transfer" if base_model_path and Path(base_model_path).exists() else "scratch"
    # Tag should reflect LR policy. For scratch we use a single LR = learning_rate.
    # For transfer, the called function composes a richer tag; here keep simple and truthful for scratch.
    model_tag = _format_run_suffix(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, image_size=image_size, tag=approach)
    model_folder_name = f"{model_name}-{model_tag}"
    
    # Create the appropriate subdirectory (transfer/ or scratch/)
    approach_dir = Path(output_dir) / approach
    approach_dir.mkdir(parents=True, exist_ok=True)
    output_path = approach_dir / model_folder_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading data from: {data_root}")
    logger.info(f"Training mode: {TRAINING_MODE_DYNAMIC_FULL_IMAGE}")
    
    # Build DataLoaders using DataBlock approach with positive-aware cropping
    dls = build_segmentation_dls_dynamic_patching(
        data_root,
        bs=batch_size,
        num_workers=0,
        crop_size=image_size,
        positive_focus_p=positive_focus_p,
        min_pos_pixels=min_pos_pixels,
        pos_crop_attempts=pos_crop_attempts,
    )

    # Save data splits manifest
    save_splits(output_path, model_folder_name, {
        "stage": "glomeruli",
        "training_mode": TRAINING_MODE_DYNAMIC_FULL_IMAGE,
        "data_root": str(data_root),
        "train_items": getattr(dls.train_ds, 'items', []),
        "valid_items": getattr(dls.valid_ds, 'items', [])
    })
    
    logger.info(f"Data loaded: {len(dls.train_ds)} train, {len(dls.valid_ds)} val samples")
    
    # Create learner: binary glomeruli segmentation (FastAI v2 approach)
    learn = unet_learner(
        dls,
        resnet34,
        n_out=2,  # 2 classes: background (0) + glomeruli (1)
        metrics=[Dice, JaccardCoeff()],  # Track both Dice and IoU for segmentation quality
        path=output_path,  # Save artifacts directly under the model output directory
        model_dir='.'  # Ensure callbacks/save go inside output_path
    )
    
    logger.info(f"Using default loss function with positive-aware cropping: {learn.loss_func}")
    
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
    
    # Train the model with callbacks
    logger.info(f"Training for {epochs} epochs...")
    save_callback = attach_best_model_callback(model_folder_name)
    learn.fit_one_cycle(epochs, lr_max=learning_rate, cbs=[save_callback])
    
    # Save training history BEFORE any plotting/predictions that may alter recorder state
    save_training_history(learn, output_path, model_folder_name, {
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'image_size': image_size,
        'base_model_path': base_model_path or 'None (from scratch)',
        'training_approach': 'transfer_learning' if base_model_path else 'from_scratch',
        'training_mode': TRAINING_MODE_DYNAMIC_FULL_IMAGE,
        'data_root': str(data_root),
    })

    # Generate training visualizations
    logger.info("Generating training visualizations...")
    save_plots(learn, output_path, model_folder_name)

    # Save the model
    model_path = export_final_model(learn, output_path, model_folder_name)
    
    # Save run metadata
    save_run_metadata(
        output_path,
        model_folder_name,
        config_path,
        extra_metadata={
            "stage": "glomeruli",
            "artifact_status": "supported_runtime",
            "scientific_promotion_status": "not_evaluated",
            "training_mode": TRAINING_MODE_DYNAMIC_FULL_IMAGE,
            "data_root": str(data_root),
            "model_path": str(model_path),
            "base_model_path": base_model_path or None,
            "invocation": {
                "data_dir": str(data_root),
                "output_dir": str(output_dir),
                "model_name": model_name,
                "base_model_path": base_model_path,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "image_size": image_size,
            },
        },
    )
    
    # Return the learner for now (can be wrapped later if needed)
    return learn


# Legacy transfer learning functions removed - using modern FastAI v2 approach

def find_best_mitochondria_model() -> Optional[str]:
    """
    Auto-detect the best available mitochondria model for transfer learning.
    
    Returns:
        str: Path to best mitochondria model, or None if not found
    """
    from eq.utils.logger import setup_logging
    logger = setup_logging(verbose=True)
    
    # Common mitochondria model locations
    search_paths = [
        "models/segmentation/mitochondria",
        DEFAULT_MITOCHONDRIA_MODEL_DIR,
        "models/mitochondria"
    ]
    
    best_model = None
    best_score = -1
    
    for search_path in search_paths:
        model_dir = Path(search_path)
        if not model_dir.exists():
            continue
            
        # Find all .pkl files in the directory tree
        pkl_files = list(model_dir.glob("**/*.pkl"))
        
        for pkl_file in pkl_files:
            try:
                load_supported_segmentation_artifact_metadata(pkl_file)
            except ValueError:
                continue
            # Prefer models with "pretrain" in the name (from mitochondria training)
            score = 0
            if "pretrain" in pkl_file.name:
                score += 10
            if "mitochondria" in pkl_file.name.lower():
                score += 5
            
            # Prefer newer files (higher epochs, recent timestamp)
            if "_e" in pkl_file.name:
                try:
                    # Extract epoch count from filename (e.g., "e10" -> 10)
                    epoch_match = re.search(r'_e(\d+)', pkl_file.name)
                    if epoch_match:
                        epochs = int(epoch_match.group(1))
                        score += epochs  # More epochs = better model
                except:
                    pass
            
            if score > best_score:
                best_score = score
                best_model = str(pkl_file)
    
    if best_model:
        logger.info(f"🔍 Auto-detected mitochondria model: {best_model}")
        return best_model
    else:
        logger.info("🔍 No mitochondria model found for transfer learning")
        return None

def main():
    """CLI interface for glomeruli training."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train glomeruli segmentation model')
    parser.add_argument('--config', help='Optional YAML config file to override defaults')
    parser.add_argument('--data-dir', required=True, help='Full-image training root containing images/ and masks/')
    parser.add_argument('--model-dir', default=DEFAULT_GLOMERULI_MODEL_DIR, help='Directory to save trained model')
    parser.add_argument('--model-name', default='glomeruli_model', help='Base name for saved model files')
    parser.add_argument('--base-model', help='Path to base model for transfer learning (auto-detected by default)')
    parser.add_argument('--from-scratch', action='store_true', help='Force training from scratch (bypass transfer learning)')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None, help='Training batch size (default: machine-aware recommendation)')
    parser.add_argument('--learning-rate', type=float, default=DEFAULT_LEARNING_RATE, help='Learning rate')
    parser.add_argument('--image-size', type=int, default=DEFAULT_IMAGE_SIZE, help='Final network input size (output_size)')
    parser.add_argument('--crop-size', type=int, default=DEFAULT_IMAGE_SIZE, help='Dynamic patching crop size before resizing')
    parser.add_argument('--loss', type=str, default='', help='Loss to use: dice | bcedice | tversky (default: fastai/weighted)')
    parser.add_argument('--skip-lr-find', action='store_true', help='Skip lr_find and use provided learning rate for fine-tune')
    
    args = parser.parse_args()

    # Optional: load YAML config and overlay onto args
    config_path = None
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
            if 'epochs' in training_cfg and parser.get_default('epochs') == args.epochs:
                args.epochs = int(training_cfg['epochs'])
            if 'batch_size' in training_cfg and args.batch_size is None:
                args.batch_size = int(training_cfg['batch_size'])
            if 'learning_rate' in training_cfg and parser.get_default('learning_rate') == args.learning_rate:
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
            config_path = args.config
        except Exception as _e:  # pragma: no cover
            print(f"⚠️  Failed to load config {args.config}: {_e}")
    
    try:
        args.batch_size = get_segmentation_training_batch_size(
            "glomeruli",
            image_size=args.image_size,
            crop_size=args.crop_size,
            requested_batch_size=args.batch_size,
        )
        # Set up logging first
        from eq.utils.logger import setup_logging
        logger = setup_logging(verbose=True)
        logger.info("🚀 Starting glomeruli model training...")
        logger.info(f"📁 Data directory: {args.data_dir}")
        logger.info(f"📁 Model directory: {args.model_dir}")
        logger.info(f"🧾 Model name: {args.model_name}")
        logger.info(f"⚙️  Epochs: {args.epochs}, Batch size: {args.batch_size}")
        
        # AUTO-DETECT MITOCHONDRIA MODEL FOR TRANSFER LEARNING (PRIMARY APPROACH)
        if args.from_scratch:
            logger.info("🔧 FROM SCRATCH (Forced): User requested training from scratch")
            args.base_model = None
        elif not args.base_model:
            # Try to auto-detect the best mitochondria model
            auto_detected_model = find_best_mitochondria_model()
            if auto_detected_model:
                args.base_model = auto_detected_model
                logger.info("🎯 TRANSFER LEARNING (Primary): Auto-detected mitochondria model")
                logger.info(f"🔄 Using base model: {args.base_model}")
            else:
                logger.info("🔧 FROM SCRATCH (Fallback): No mitochondria model found")
        else:
            logger.info("🔄 TRANSFER LEARNING (Manual): Using provided base model")
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
                image_size=args.image_size,
                loss_name=args.loss or None,
                crop_size=args.crop_size,
                # Skip lr_find if user asked to, or use provided LR directly
                use_lr_find=(not args.skip_lr_find)
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
                image_size=args.image_size,
                config_path=config_path,
                # TODO: implement loss_name
                # loss_name=args.loss or None
            )
        
        logger.info("🎉 Glomeruli training completed successfully!")
        print(f"✅ Model saved to: {args.model_dir}/glomeruli_model")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        print(f"❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
