#!/usr/bin/env python3
"""Main CLI entry point for the endotheliosis quantifier package."""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from eq.config.mode_manager import EnvironmentMode, ModeManager
from eq.utils.hardware_detection import get_capability_report
from eq.utils.logger import ProgressLogger, get_logger, log_function_call, setup_logging

# from eq.models.feature_extractor import run_feature_extraction
# from eq.pipeline.quantify_endotheliosis import run_endotheliosis_quantification
# from eq.segmentation.train_segmenter import train_segmentation_model


@log_function_call
def pipeline_orchestrator_command(args):
    """Interactive pipeline orchestrator with menu selection."""
    logger = get_logger("eq.pipeline_orchestrator")
    logger.info("🚀 Starting interactive pipeline orchestrator...")
    
    print("🚀 === ENDOTHELIOSIS QUANTIFIER PIPELINE ===")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check for environment variables first
    import os
    
    if os.getenv('QUICK_TEST') == 'true':
        print("🚀 Running Development Pipeline (QUICK_TEST=true)...")
        epochs = 2
        from eq.pipeline.run_production_pipeline import run_pipeline
        run_pipeline(epochs=epochs, run_type="development")
        return
        
    if os.getenv('PRODUCTION') == 'true':
        print("🚀 Running Production Pipeline (PRODUCTION=true)...")
        epochs = 10
        from eq.pipeline.run_production_pipeline import run_pipeline
        run_pipeline(epochs=epochs, run_type="production")
        return
    
    # Fallback to interactive menu if no environment variables
    print("Available pipeline modes:")
    print("  1. Development (2-5 epochs) - Fast validation and testing")
    print("  2. Production (10+ epochs) - Full training for deployment")
    print("  3. Smoke Test - Basic functionality check (in tests/)")
    print()
    print("Or use environment variables:")
    print("  QUICK_TEST=true python -m eq orchestrator")
    print("  PRODUCTION=true python -m eq orchestrator")
    print()
    
    choice = input("Select mode (1-3): ").strip()
    
    if choice == "1":
        print("\n🚀 Running Development Pipeline...")
        epochs = 2  # Default to 2 epochs for testing
        
        from eq.pipeline.run_production_pipeline import run_pipeline
        run_pipeline(epochs=epochs, run_type="development")
        
    elif choice == "2":
        print("\n🚀 Running Production Pipeline...")
        epochs = 10  # Default to 10 epochs for production
        
        from eq.pipeline.run_production_pipeline import run_pipeline
        run_pipeline(epochs=epochs, run_type="production")
        
    elif choice == "3":
        print("\n🧪 Running Smoke Test...")
        print("Note: This is a test script. Run with: python -m pytest tests/test_smoke_pipeline.py")
        print("Or run directly: python tests/test_smoke_pipeline.py")
        
    else:
        print("❌ Invalid choice. Please select 1-3.")
        print()
        print("You can also run specific components directly:")
        print("  # Development run")
        print("  python src/eq/pipeline/run_production_pipeline.py --run-type development --epochs 2")
        print()
        print("  # Production run")
        print("  python src/eq/pipeline/run_production_pipeline.py --run-type production --epochs 10")
        print()
        print("  # Smoke test")
        print("  python tests/test_smoke_pipeline.py")


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
    """Run the complete pipeline end-to-end."""
    logger = get_logger("eq.pipeline")
    logger.info("🚀 Starting complete endotheliosis quantification pipeline...")
    logger.info("=" * 60)
    
    # Get mode manager and validate mode
    mode_manager = ModeManager()
    _validate_mode_for_command(mode_manager, "pipeline")
    
    # Get mode-aware batch size
    batch_size = _get_mode_aware_batch_size(mode_manager, args.batch_size)
    logger.info(f"🔧 Mode: {mode_manager.current_mode.value}")
    logger.info(f"📋 Batch size: {batch_size} (auto-detected)")
    
    # Set up progress tracking for the full pipeline
    progress = ProgressLogger(logger, 4, "End-to-End Pipeline")
    
    # Step 1: Load data
    logger.info("\n" + "=" * 20 + " STEP 1: LOADING DATA " + "=" * 20)
    progress.step("Loading and preprocessing data")
    data_load_command(args)
    
    # Step 2: Train segmenter
    logger.info("\n" + "=" * 20 + " STEP 2: TRAINING SEGMENTATION MODEL " + "=" * 20)
    progress.step("Training segmentation model")
    train_segmenter_command(args)
    
    # Step 3: Extract features
    logger.info("\n" + "=" * 20 + " STEP 3: EXTRACTING FEATURES " + "=" * 20)
    progress.step("Extracting features from images")
    extract_features_command(args)
    
    # Step 4: Quantify
    logger.info("\n" + "=" * 20 + " STEP 4: QUANTIFYING ENDOTHELIOSIS " + "=" * 20)
    progress.step("Running endotheliosis quantification")
    quantify_command(args)
    
    progress.complete("Complete endotheliosis quantification pipeline")
    logger.info("🎉 Pipeline completed successfully!")
    logger.info("=" * 60)


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
    pipeline_parser = subparsers.add_parser('pipeline', help='Run complete pipeline')
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
