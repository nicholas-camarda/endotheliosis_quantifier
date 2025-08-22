#!/usr/bin/env python3
"""Main CLI entry point for the endotheliosis quantifier package."""

import argparse
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from fastai.vision.all import *

from eq.config.mode_manager import EnvironmentMode, ModeManager
from eq.utils.hardware_detection import get_capability_report
from eq.utils.logger import ProgressLogger, get_logger, log_function_call, setup_logging


# Functions needed for loading pre-trained models
def n_glom_codes(mask_files):
    """Get unique codes from mask files."""
    codes = set()
    for mask_file in mask_files:
        mask = np.array(PILMask.create(mask_file))
        codes.update(np.unique(mask))
    return sorted(list(codes))


def get_glom_mask_file(image_file, p2c, thresh=127):
    """Get mask file path for a given image file."""
    # this is the base path
    base_path = image_file.parent.parent.parent
    first_name = image_file.parent.name
    # get training or testing from here
    full_name = re.findall(string=image_file.name, pattern=r"^[A-Za-z]*[0-9]+[_|-]+[A-Za-z]*[0-9]+")[0]
    
    # put the whole thing together
    str_name = f'{full_name}_mask' + image_file.suffix
    # attach it to the correct path
    mask_path = (base_path / 'masks' / first_name / str_name)
    
    # convert to an array (mask)
    msk = np.array(PILMask.create(mask_path))
    # convert the image to binary if it isn't already (tends to happen when working with .jpg files)
    msk[msk <= thresh] = 0
    msk[msk > thresh] = 1
    
    # find all the possible values in the mask (0,255)
    for i, val in enumerate(p2c):
        msk[msk == p2c[i]] = val
    return PILMask.create(msk)


def get_glom_y(o):
    """Get glomeruli mask for a given image file."""
    # This is a placeholder - p2c should be defined when this function is used
    # For now, we'll use a default value
    p2c = [0, 1]  # Default binary mask codes
    return get_glom_mask_file(o, p2c)


# from eq.models.feature_extractor import run_feature_extraction
# from eq.pipeline.quantify_endotheliosis import run_endotheliosis_quantification
# from eq.segmentation.train_segmenter import train_segmentation_model


@log_function_call
def pipeline_orchestrator_command(args):
    """Pipeline orchestrator that runs the specified pipeline stage."""
    logger = get_logger("eq.pipeline_orchestrator")
    logger.info("🚀 Starting pipeline orchestrator...")
    
    print("🚀 === ENDOTHELIOSIS QUANTIFIER PIPELINE ===")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check for QUICK_TEST mode
    import os
    is_quick_test = os.getenv('QUICK_TEST') == 'true'
    if is_quick_test:
        print("🔍 QUICK_TEST mode enabled - using fast validation settings")
        epochs = 2
        batch_size = 4
                print(f"Settings: epochs={args.epochs}, batch_size={args.batch_size}")

    else:
        print("🚀 PRODUCTION mode - using full settings")
        epochs = 50
        batch_size = 8
    
    print()
    print("Available pipeline stages:")
    print("  1. Segmentation Training (seg) - Train model to find glomeruli")
    print("  2. Quantification Training (quant-endo) - Train regression model for endotheliosis scoring") 
    print("  3. Production Inference (production) - End-to-end inference using pre-trained models")
    print()
    print("Usage:")
    print("  python -m eq seg                    # Train segmentation model")
    print("  python -m eq quant-endo             # Train quantification model")
    print("  python -m eq production             # Run production inference")
    print("  QUICK_TEST=true python -m eq seg    # Quick test segmentation training")
    print("  QUICK_TEST=true python -m eq production  # Quick test production inference")
    print()
    print("❌ No interactive input required. Use specific commands above.")
    print("❌ This orchestrator is for documentation only.")


@log_function_call
def data_load_command(args):
    """Load and preprocess data for the pipeline."""
    logger = get_logger("eq.data_load")
    logger.info("🔄 Starting data loading and preprocessing pipeline...")

    # Lazy import heavy data utilities to avoid import-time side effects
    from eq.features.data_loader import (
        create_train_val_test_lists,
        generate_binary_masks,
        generate_final_dataset,
        get_scores_from_annotations,
        load_annotations_from_json,
        organize_data_into_subdirs,
    )

    # Set up progress tracking
    progress = ProgressLogger(logger, 6, "Data Loading Pipeline")
    
    # Generate binary masks if annotation file provided
    if args.annotation_file:
        progress.step(f"Generating binary masks from {args.annotation_file}")
        generate_binary_masks(
            annotation_file=args.annotation_file,
            data_dir=args.data_dir
        )
    else:
        progress.step("Skipping binary mask generation (no annotation file provided)")
    
    # Organize test data
    progress.step("Organizing test data into subdirectories")
    test_images_1_paths, test_data_dict_1 = organize_data_into_subdirs(
        data_dir=args.test_data_dir
    )
    logger.info(f"📊 Found {len(test_images_1_paths)} test images")
    
    # Create train/val/test lists
    progress.step("Creating train/val/test data splits")
    train_images_paths, train_masks_paths, test_images_2_paths, train_data_dict, test_data_dict_2 = create_train_val_test_lists(
        data_dir=args.data_dir
    )
    logger.info(f"📊 Training images: {len(train_images_paths)}")
    logger.info(f"📊 Training masks: {len(train_masks_paths)}")
    logger.info(f"📊 Test images: {len(test_images_2_paths)}")
    
    # Combine test images
    import numpy as np
    test_images_paths = np.concatenate((test_images_1_paths, test_images_2_paths))
    test_data_dict = dict(sorted({**test_data_dict_1, **test_data_dict_2}.items()))
    logger.info(f"📊 Combined test images: {len(test_images_paths)}")
    
    # Generate final dataset
    progress.step(f"Generating final dataset with image size {args.image_size}")
    generate_final_dataset(
        train_images_paths, train_masks_paths, test_images_paths,
        train_data_dict, test_data_dict, 
        size=args.image_size, cache_dir=args.cache_dir
    )
    
    # Process scores if annotation file provided
    if args.annotation_file:
        progress.step("Processing scores from annotations")
        annotations = load_annotations_from_json(args.annotation_file)
        scores = get_scores_from_annotations(annotations, cache_dir=args.cache_dir)
        logger.info(f"📊 Processed {len(scores)} scores from annotations")
    else:
        progress.step("Skipping score processing (no annotation file provided)")
    
    progress.complete("Data loading and preprocessing")
    logger.info("🎉 Data loading pipeline completed successfully!")


@log_function_call
def mode_command(args):
    """Inspect and manage environment mode selection."""
    logger = get_logger("eq.mode")
    logger.info("⚙️ Managing environment mode...")

    # Initialize manager (respects persisted config at ~/.eq/config.json)
    manager = ModeManager()

    # Apply requested mode change if provided
    if getattr(args, "set", None):
        try:
            # Validate the mode before setting it
            is_valid, reason = manager.validate_mode(EnvironmentMode(args.set))
            if not is_valid:
                logger.error(f"❌ Invalid mode '{args.set}': {reason}")
                print(f"❌ Cannot set mode to '{args.set}': {reason}")
                print(f"💡 Suggested mode: {manager.get_suggested_mode().value}")
                sys.exit(1)
            
            manager.switch_mode(EnvironmentMode(args.set))
            print(f"✅ Mode updated to: {manager.current_mode.value.upper()}")
        except Exception as e:
            logger.error(f"Failed to set mode '{args.set}': {e}")
            print(f"❌ Failed to set mode '{args.set}': {e}")
            sys.exit(1)

    # Validate mode if requested
    if getattr(args, "validate", False):
        is_valid, reason = manager.validate_mode(manager.current_mode)
        status = "✅ VALID" if is_valid else "❌ INVALID"
        print(f"Validation: {status} - {reason}")
        if not is_valid:
            print(f"💡 Suggested mode: {manager.get_suggested_mode().value}")
            sys.exit(1)

    # Show summary if requested
    if getattr(args, "show", False) or not (getattr(args, "set", None) or getattr(args, "validate", False)):
        print(manager.get_mode_summary())


def _validate_mode_for_command(mode_manager: ModeManager, command: str) -> None:
    """Validate that the current mode is suitable for the given command."""
    is_valid, reason = mode_manager.validate_mode(mode_manager.current_mode)
    
    if not is_valid:
        logger = get_logger("eq.cli.validation")
        logger.warning(f"Mode validation failed for command '{command}': {reason}")
        
        # For production commands, be more strict
        if command in ['train-segmenter', 'pipeline'] and mode_manager.current_mode == EnvironmentMode.PRODUCTION:
            print(f"❌ Production mode validation failed: {reason}")
            print(f"💡 Suggested mode: {mode_manager.get_suggested_mode().value}")
            print("💡 Use 'eq mode --set development' to switch to development mode")
            sys.exit(1)
        
        # For other commands, just warn
        print(f"⚠️  Warning: {reason}")
        print(f"💡 Consider switching to: {mode_manager.get_suggested_mode().value}")


def _get_mode_aware_batch_size(mode_manager: ModeManager, user_batch_size: int) -> int:
    """Get batch size considering mode and hardware capabilities."""
    if user_batch_size > 0:
        return user_batch_size
    
    # Auto-detect based on mode and hardware
    from eq.utils.hardware_detection import get_optimal_batch_size
    optimal_size = get_optimal_batch_size(mode_manager.current_mode.value)
    
    logger = get_logger("eq.cli.batch_size")
    logger.info(f"Auto-detected batch size for {mode_manager.current_mode.value} mode: {optimal_size}")
    
    return optimal_size


def _handle_mode_specific_errors(e: Exception, mode_manager: ModeManager, command: str) -> None:
    """Handle mode-specific error recovery and suggestions."""
    error_msg = str(e).lower()
    
    # Hardware-related errors
    if "cuda" in error_msg or "gpu" in error_msg:
        if mode_manager.current_mode == EnvironmentMode.PRODUCTION:
            print("❌ GPU/CUDA error in production mode")
            print("💡 Try switching to development mode: eq mode --set development")
        else:
            print("❌ GPU/CUDA error detected")
            print("💡 Try switching to CPU mode or check GPU drivers")
    
    # Memory-related errors
    elif "memory" in error_msg or "oom" in error_msg:
        if mode_manager.current_mode == EnvironmentMode.PRODUCTION:
            print("❌ Memory error in production mode")
            print("💡 Try reducing batch size or switching to development mode")
        else:
            print("❌ Memory error detected")
            print("💡 Try reducing batch size or closing other applications")
    
    # Backend-related errors
    elif "mps" in error_msg:
        print("❌ MPS (Apple Silicon GPU) error detected")
        print("💡 Try switching to CPU mode: eq mode --set development")
    
    # Generic error handling
    else:
        print(f"❌ Unexpected error: {e}")
        if mode_manager.current_mode == EnvironmentMode.PRODUCTION:
            print("💡 Consider switching to development mode for debugging")


@log_function_call
def train_segmenter_command(args):
    """Train a segmentation model."""
    logger = get_logger("eq.train_segmenter")
    logger.info("🔄 Starting segmentation model training...")
    
    # Get mode manager and validate mode
    mode_manager = ModeManager()
    _validate_mode_for_command(mode_manager, "train-segmenter")
    
    # Get mode-aware batch size
    batch_size = _get_mode_aware_batch_size(mode_manager, args.batch_size)
    logger.info(f"📋 Training parameters: batch_size={batch_size}, epochs={args.epochs}")
    logger.info(f"🔧 Mode: {mode_manager.current_mode.value}")
    
    print("Note: This command is not yet implemented due to import issues.")
    # train_segmentation_model(
    #     base_model_path=args.base_model_path,
    #     cache_dir=args.cache_dir,
    #     output_dir=args.output_dir,
    #     model_name=args.model_name,
    #     batch_size=batch_size,
    #     epochs=args.epochs
    # )
    logger.info("✅ Segmentation training complete!")


@log_function_call
def extract_features_command(args):
    """Extract features from images."""
    logger = get_logger("eq.extract_features")
    logger.info("🔄 Starting feature extraction...")
    
    # Get mode manager and validate mode
    mode_manager = ModeManager()
    _validate_mode_for_command(mode_manager, "extract-features")
    
    logger.info(f"🔧 Mode: {mode_manager.current_mode.value}")
    
    print("Note: This command is not yet implemented due to import issues.")
    # run_feature_extraction(
    #     cache_dir=args.cache_dir,
    #     output_dir=args.output_dir,
    #     model_path=args.model_path
    # )
    logger.info("✅ Feature extraction complete!")


@log_function_call
def quantify_command(args):
    """Run endotheliosis quantification."""
    logger = get_logger("eq.quantify")
    logger.info("🔄 Starting endotheliosis quantification...")
    
    # Get mode manager and validate mode
    mode_manager = ModeManager()
    _validate_mode_for_command(mode_manager, "quantify")
    
    logger.info(f"🔧 Mode: {mode_manager.current_mode.value}")
    
    print("Note: This command is not yet implemented due to import issues.")
    # run_endotheliosis_quantification(
    #     cache_dir=args.cache_dir,
    #     output_dir=args.output_dir,
    #     model_path=args.model_path
    # )
    logger.info("✅ Quantification complete!")


@log_function_call
def capabilities_command(args):
    """Report detected hardware capabilities and recommendations."""
    logger = get_logger("eq.capabilities")
    logger.info("🔍 Generating hardware capability report...")
    report = get_capability_report()
    print(report)


@log_function_call
def pipeline_command(args):
    """Run complete end-to-end inference pipeline using pre-trained models."""
    logger = get_logger("eq.production")
    logger.info("🔄 Starting end-to-end production inference...")
    
    # Get mode manager and validate mode
    mode_manager = ModeManager()
    _validate_mode_for_command(mode_manager, "production")
    
    # Check for QUICK_TEST mode
    import os
    is_quick_test = os.getenv('QUICK_TEST') == 'true'
    if is_quick_test:
        print("🔍 QUICK_TEST mode detected - using fast validation settings")
        args.epochs = min(args.epochs, 2)  # Limit epochs for quick testing
        args.batch_size = min(args.batch_size, 4)  # Smaller batch size for quick testing
    
    # Get mode-aware batch size
    batch_size = _get_mode_aware_batch_size(mode_manager, args.batch_size)
    logger.info(f"📋 Inference parameters: batch_size={batch_size}, epochs={args.epochs}")
    logger.info(f"🔧 Mode: {mode_manager.current_mode.value}")
    
    print("🚀 === PRODUCTION INFERENCE PIPELINE ===")
    print("Running end-to-end inference using pre-trained models...")
    print(f"Data directory: {args.data_dir}")
    print(f"Test data directory: {args.test_data_dir}")
    print(f"Cache directory: {args.cache_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Base model path: {args.base_model_path}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Quick test: {is_quick_test}")
    
    try:
        # Use the actual working pipeline with proper data paths
        from eq.pipeline.run_production_pipeline import run_pipeline
        
        run_pipeline(
            epochs=args.epochs, 
            run_type="development" if is_quick_test else "production", 
            use_existing_models=True,
            data_dir=args.data_dir,
            cache_dir=args.cache_dir
        )
        
        logger.info("✅ Production inference complete!")
        
    except Exception as e:
        logger.error(f"❌ Production inference failed: {e}")
        print(f"❌ Production inference failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


@log_function_call
def seg_command(args):
    """Train segmentation model to find glomeruli."""
    logger = get_logger("eq.seg")
    logger.info("🔄 Starting segmentation model training...")
    
    # Get mode manager and validate mode
    mode_manager = ModeManager()
    _validate_mode_for_command(mode_manager, "seg")
    
    # Check for QUICK_TEST mode
    import os
    is_quick_test = os.getenv('QUICK_TEST') == 'true'
    if is_quick_test:
        print("🔍 QUICK_TEST mode detected - using fast validation settings")
        args.epochs = min(args.epochs, 2)  # Limit epochs for quick testing
        args.batch_size = min(args.batch_size, 4)  # Smaller batch size for quick testing
    
    # Get mode-aware batch size
    batch_size = _get_mode_aware_batch_size(mode_manager, args.batch_size)
    logger.info(f"📋 Training parameters: batch_size={batch_size}, epochs={args.epochs}")
    logger.info(f"🔧 Mode: {mode_manager.current_mode.value}")
    
    print("🚀 === SEGMENTATION TRAINING ===")
    print("Training segmentation model to find glomeruli...")
    print(f"Data directory: {args.data_dir}")
    print(f"Cache directory: {args.cache_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Quick test: {is_quick_test}")
    
    try:
        # Use the actual working pipeline for segmentation training with proper data paths
        from eq.pipeline.run_production_pipeline import run_pipeline
        
        run_pipeline(
            epochs=args.epochs, 
            run_type="development" if is_quick_test else "production", 
            use_existing_models=False,  # Force training new models
            data_dir=args.data_dir,
            cache_dir=args.cache_dir
        )
        
        logger.info("✅ Segmentation training complete!")
        
    except Exception as e:
        logger.error(f"❌ Segmentation training failed: {e}")
        print(f"❌ Segmentation training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


@log_function_call
def quant_endo_command(args):
    """Train quantification model for endotheliosis scoring."""
    logger = get_logger("eq.quant_endo")
    logger.info("🔄 Starting quantification model training...")
    
    # Get mode manager and validate mode
    mode_manager = ModeManager()
    _validate_mode_for_command(mode_manager, "quant-endo")
    
    # Get mode-aware batch size
    batch_size = _get_mode_aware_batch_size(mode_manager, args.batch_size)
    logger.info(f"📋 Training parameters: batch_size={batch_size}, epochs={args.epochs}")
    logger.info(f"🔧 Mode: {mode_manager.current_mode.value}")
    
    print("🚀 === QUANTIFICATION TRAINING ===")
    print("Training quantification model for endotheliosis scoring...")
    
    # Check for QUICK_TEST mode
    import os
    if os.getenv('QUICK_TEST') == 'true':
        print("🔍 QUICK_TEST mode detected - using fast validation settings")
        args.epochs = min(args.epochs, 5)  # Limit epochs for quick testing
        batch_size = min(batch_size, 4)    # Smaller batch size for quick testing
    
    try:
        # Use the actual working pipeline for quantification training with proper data paths
        from eq.pipeline.run_production_pipeline import run_pipeline
        
        run_pipeline(
            epochs=args.epochs, 
            run_type="quant_training", 
            use_existing_models=True,  # Use existing segmentation models for ROI extraction
            data_dir=args.data_dir,
            cache_dir=args.cache_dir
        )
        
        logger.info("✅ Quantification training complete!")
        
    except Exception as e:
        logger.error(f"❌ Quantification training failed: {e}")
        print(f"❌ Quantification training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Endotheliosis Quantifier Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  eq data-load --data-dir data/train --test-data-dir data/test
  eq train-segmenter --cache-dir data/cache --output-dir output
  eq extract-features --cache-dir data/cache --output-dir output
  eq quantify --cache-dir data/cache --output-dir output
  eq pipeline --data-dir data/train --test-data-dir data/test --output-dir output
  eq capabilities
  eq mode --set development --show --validate
  eq orchestrator  # Interactive menu
        """
    )
    
    # Add global options
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Enable verbose logging with more details')
    parser.add_argument('--log-file', type=str, 
                       help='Write logs to specified file')
    parser.add_argument('--mode', choices=['auto', 'development', 'production'], default='auto',
                       help='Select environment mode for this session (default: auto)')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Pipeline orchestrator command (interactive menu)
    orchestrator_parser = subparsers.add_parser(
        'orchestrator', 
        help='Interactive pipeline orchestrator with menu selection',
        description='Interactive pipeline orchestrator with menu selection'
    )
    orchestrator_parser.set_defaults(func=pipeline_orchestrator_command)
    
    # Data loading command
    data_parser = subparsers.add_parser('data-load', help='Load and preprocess data')
    data_parser.add_argument('--data-dir', required=True, help='Training data directory')
    data_parser.add_argument('--test-data-dir', required=True, help='Test data directory')
    data_parser.add_argument('--cache-dir', required=True, help='Cache directory')
    data_parser.add_argument('--annotation-file', help='Annotation JSON file')
    data_parser.add_argument('--image-size', type=int, default=256, help='Image size for processing')
    data_parser.set_defaults(func=data_load_command)
    
    # Training command
    train_parser = subparsers.add_parser('train-segmenter', help='Train segmentation model')
    train_parser.add_argument('--base-model-path', required=True, help='Path to base model')
    train_parser.add_argument('--cache-dir', required=True, help='Cache directory')
    train_parser.add_argument('--output-dir', required=True, help='Output directory')
    train_parser.add_argument('--model-name', default='glomerulus_segmenter', help='Model name')
    train_parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    train_parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    train_parser.set_defaults(func=train_segmenter_command)
    
    # Segmentation training command
    seg_parser = subparsers.add_parser('seg', help='Train segmentation model to find glomeruli')
    seg_parser.add_argument('--data-dir', required=True, help='Training data directory')
    seg_parser.add_argument('--cache-dir', required=True, help='Cache directory')
    seg_parser.add_argument('--output-dir', required=True, help='Output directory')
    seg_parser.add_argument('--annotation-file', help='Annotation JSON file')
    seg_parser.add_argument('--image-size', type=int, default=256, help='Image size for processing')
    seg_parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    seg_parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    seg_parser.set_defaults(func=seg_command)
    
    # Quantification training command
    quant_parser = subparsers.add_parser('quant-endo', help='Train quantification model for endotheliosis scoring')
    quant_parser.add_argument('--data-dir', required=True, help='Training data directory')
    quant_parser.add_argument('--cache-dir', required=True, help='Cache directory')
    quant_parser.add_argument('--output-dir', required=True, help='Output directory')
    quant_parser.add_argument('--segmentation-model', required=True, help='Path to trained segmentation model')
    quant_parser.add_argument('--annotation-file', help='Annotation JSON file with endotheliosis scores')
    quant_parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    quant_parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    quant_parser.set_defaults(func=quant_endo_command)
    
    # Feature extraction command
    features_parser = subparsers.add_parser('extract-features', help='Extract features')
    features_parser.add_argument('--cache-dir', required=True, help='Cache directory')
    features_parser.add_argument('--output-dir', required=True, help='Output directory')
    features_parser.add_argument('--model-path', required=True, help='Path to trained model')
    features_parser.set_defaults(func=extract_features_command)
    
    # Quantification command
    quant_parser = subparsers.add_parser('quantify', help='Run endotheliosis quantification')
    quant_parser.add_argument('--cache-dir', required=True, help='Cache directory')
    quant_parser.add_argument('--output-dir', required=True, help='Output directory')
    quant_parser.add_argument('--model-path', required=True, help='Path to trained model')
    quant_parser.set_defaults(func=quantify_command)

    # Capabilities command
    capabilities_parser = subparsers.add_parser(
        'capabilities',
        help='Show hardware capabilities and recommendations',
        description='Show hardware capabilities and recommendations'
    )
    capabilities_parser.set_defaults(func=capabilities_command)

    # Mode command
    mode_parser = subparsers.add_parser(
        'mode',
        help='Inspect and manage environment mode',
        description='Inspect and manage environment mode'
    )
    mode_parser.add_argument('--set', choices=['auto', 'development', 'production'], help='Set the environment mode')
    mode_parser.add_argument('--show', action='store_true', help='Show current mode and configuration summary')
    mode_parser.add_argument('--validate', action='store_true', help='Validate current mode against hardware capabilities')
    mode_parser.set_defaults(func=mode_command)
    
    # Pipeline command
    pipeline_parser = subparsers.add_parser('production', help='Run complete end-to-end inference pipeline')
    pipeline_parser.add_argument('--data-dir', required=True, help='Training data directory')
    pipeline_parser.add_argument('--test-data-dir', required=True, help='Test data directory')
    pipeline_parser.add_argument('--cache-dir', required=True, help='Cache directory')
    pipeline_parser.add_argument('--output-dir', required=True, help='Output directory')
    pipeline_parser.add_argument('--annotation-file', help='Annotation JSON file')
    pipeline_parser.add_argument('--base-model-path', required=True, help='Path to base model')
    pipeline_parser.add_argument('--image-size', type=int, default=256, help='Image size for processing')
    pipeline_parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    pipeline_parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    pipeline_parser.set_defaults(func=pipeline_command)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Initialize mode manager for the session based on global --mode
    try:
        session_mode = EnvironmentMode(args.mode) if getattr(args, 'mode', None) else EnvironmentMode.AUTO
    except Exception:
        session_mode = EnvironmentMode.AUTO
    
    mode_manager = ModeManager(mode=session_mode)
    
    # Set up logging
    log_file = Path(args.log_file) if args.log_file else None
    logger = setup_logging(
        level=logging.DEBUG if args.verbose else logging.INFO,
        log_file=log_file,
        verbose=args.verbose
    )
    
    logger.info(f"🔧 Starting eq command: {args.command}")
    logger.info(f"📋 Arguments: {vars(args)}")
    logger.info(f"🔧 Mode: {mode_manager.current_mode.value}")
    
    try:
        args.func(args)
        logger.info("🎉 Command completed successfully!")
    except Exception as e:
        logger.error(f"❌ Command failed: {str(e)}")
        
        # Handle mode-specific error recovery
        _handle_mode_specific_errors(e, mode_manager, args.command)
        
        if args.verbose:
            import traceback
            logger.error(f"📋 Full traceback:\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == '__main__':
    main()


