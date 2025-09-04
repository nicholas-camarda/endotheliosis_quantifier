#!/usr/bin/env python3
"""Main CLI entry point for the endotheliosis quantifier package."""

import argparse
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from eq.utils.logger import ProgressLogger, get_logger, log_function_call, setup_logging
from eq.utils.mode_manager import EnvironmentMode, ModeManager
from eq.core.constants import DEFAULT_MASK_THRESHOLD, DEFAULT_IMAGE_SIZE

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from eq.pipeline.run_production_pipeline import run_pipeline
from eq.data_management.metadata_processor import process_metadata_file
from eq.utils.image_mask_vis import visualize_mask, visualize_image_mask_pair, visualize_batch_masks

# Optional conda environment activation (opt-in via EQ_AUTO_CONDA=1)
try:
    from eq import ensure_conda_environment
    if os.environ.get('EQ_AUTO_CONDA', '0') == '1':
        ensure_conda_environment()
except Exception:
    pass

# AUTOMATIC ENVIRONMENT SETUP - runs at CLI import time
def auto_setup_environment():
    """Automatically set up environment and hardware detection on CLI startup."""
    try:
        # Lazy import to avoid hard dependency when just showing help
        from eq.utils.hardware_detection import get_hardware_capabilities

        # Auto-detect hardware and set optimal mode (completely silent)
        hardware_capabilities = get_hardware_capabilities()
        
        # Temporarily suppress logging during auto-setup
        import logging
        original_level = logging.getLogger().level
        logging.getLogger().setLevel(logging.ERROR)
        
        mode_manager = ModeManager()
        suggested_mode = mode_manager.get_suggested_mode()
        mode_manager.set_mode(suggested_mode)
        
        # Restore logging level
        logging.getLogger().setLevel(original_level)
        
        # Configure platform-specific settings
        if hardware_capabilities.platform == "Darwin":
            import os
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        
        return mode_manager
        
    except Exception:
        # Completely silent fallback
        return None


# Run automatic setup when module is imported
_auto_mode_manager = auto_setup_environment()


# Functions needed for loading pre-trained models
def n_glom_codes(mask_files):
    """Get unique codes from mask files."""
    codes = set()
    # Lazy import to avoid fastai dependency unless needed
    try:
        from fastai.vision.all import PILMask
    except Exception:
        raise ImportError("fastai not available; install fastai to use n_glom_codes")
    for mask_file in mask_files:
        mask = np.array(PILMask.create(mask_file))
        codes.update(np.unique(mask))
    return sorted(list(codes))


def get_glom_mask_file(image_file, p2c, thresh=DEFAULT_MASK_THRESHOLD):
    """Get mask path for image file."""
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
    try:
        from fastai.vision.all import PILMask
    except Exception:
        raise ImportError("fastai not available; install fastai to use get_glom_mask_file")
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
# from eq.models.train_segmenter_fastai import train_segmentation_model


@log_function_call
def pipeline_orchestrator_command(args):
    """Pipeline orchestrator that runs the specified pipeline stage."""
    logger = get_logger("eq.pipeline_orchestrator")
    logger.info("🚀 Starting pipeline orchestrator...")
    
    print("🚀 === ENDOTHELIOSIS QUANTIFIER PIPELINE ===")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Use the auto-detected mode manager
    if _auto_mode_manager:
        current_mode = _auto_mode_manager.current_mode
        current_config = _auto_mode_manager.current_config
        print(f"🎯 Current Mode: {current_mode.value.upper()}")
        print(f"   Batch Size: {current_config.batch_size or 'Auto'}")
        print(f"   Device: {_auto_mode_manager.get_device_recommendation()}")
    
    # Check for QUICK_TEST mode
    import os
    is_quick_test = os.getenv('QUICK_TEST') == 'true'
    if is_quick_test:
        print("🔍 QUICK_TEST mode enabled - using fast validation settings")
        epochs = 5
        batch_size = 4
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
    # Note: These functions are not yet implemented in the consolidated architecture
    # TODO: Implement these functions in the appropriate modules
    logger.warning("⚠️  Data loading functions not yet implemented in consolidated architecture")
    logger.warning("⚠️  Skipping data loading step")
    return

    # Note: create_train_val_test_lists, organize_data_into_subdirs, and
    # generate_binary_masks were part of legacy features modules. The
    # consolidated loader returns train/val/test splits directly.
    # Set up progress tracking
    progress = ProgressLogger(logger, 6, "Data Loading Pipeline")
    
    # Load and split data using the unified loader (into cache)
    progress.step("Loading and caching glomeruli dataset")
    data_splits = generate_final_dataset(
        processed_images_dir=args.data_dir,
        cache_dir=args.cache_dir
    )
    logger.info(f"📊 Train samples: {data_splits['metadata']['train_samples']}")
    logger.info(f"📊 Val samples: {data_splits['metadata']['val_samples']}")
    logger.info(f"📊 Test samples: {data_splits['metadata']['test_samples']}")
    
    # Process scores if annotation file provided
    if args.annotation_file:
        progress.step("Processing scores from annotations")
        annotations = load_annotations_from_json(args.annotation_file)
        scores = get_scores_from_annotations(annotations)
        logger.info(f"📊 Processed {len(scores)} scores from annotations")
    else:
        progress.step("Skipping score processing (no annotation file provided)")
    
    progress.complete("Data loading and preprocessing")
    logger.info("🎉 Data loading pipeline completed successfully!")


def process_data_command(args):
    """Process raw data into derived_data."""
    logger = get_logger("eq.process_data")
    logger.info("🔄 Starting data processing pipeline...")

    from pathlib import Path
    import os
    from datetime import datetime
    from eq.processing.image_mask_preprocessing import patchify_dataset
    from eq.core.constants import EXPECTED_INPUT_WIDTH, EXPECTED_INPUT_HEIGHT, EXPECTED_PATCHES_PER_IMAGE
    
    # Set up progress tracking
    progress = ProgressLogger(logger, 4, "Data Processing Pipeline")
    
    # Validate input directory
    input_path = Path(args.input_dir)
    if not input_path.exists():
        logger.error(f"❌ Input directory does not exist: {input_path}")
        sys.exit(1)
    
    # Show expected processing info
    logger.info(f"📏 Expected input image dimensions: {EXPECTED_INPUT_WIDTH}x{EXPECTED_INPUT_HEIGHT}")
    logger.info(f"✂️  Creating {args.patch_size}x{args.patch_size} patches")
    logger.info(f"📊 Expected patches per image: ~{EXPECTED_PATCHES_PER_IMAGE}")
    
    # Create output directory structure
    output_path = Path(args.output_dir)
    progress.step("Creating output directory structure")
    
    # Create the main derived_data structure
    image_patches_dir = output_path / "image_patches"
    mask_patches_dir = output_path / "mask_patches"
    cache_dir = output_path / "cache"
    
    for dir_path in [image_patches_dir, mask_patches_dir, cache_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Created directory: {dir_path}")
    
    # Single unified processing step
    progress.step("Processing data (auto-detect masks and structure)")
    counts = patchify_dataset(
        input_root=str(input_path),
        output_root=str(output_path),
        patch_size=args.patch_size,
    )
    
    # Count processed files (search recursively through subdirectories)
    image_count = counts.get("images", 0)
    mask_count = counts.get("masks", 0)
    subjects_count = counts.get("subjects", 0)
    
    progress.step("Finalizing output")
    
    # Create metadata file
    metadata = {
        'input_directory': str(input_path),
        'output_directory': str(output_path),
        'patch_size': args.patch_size,
        'overlap': args.overlap,
        'has_masks': mask_count > 0,
        'processed_at': datetime.now().isoformat(),
        'statistics': {
            'image_patches': image_count,
            'mask_patches': mask_count,
            'subjects_processed': subjects_count,
            'total_files': image_count + mask_count
        }
    }
    
    import json
    metadata_file = output_path / "processing_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    progress.complete("Data processing")
    logger.info(f"🎉 Data processing completed successfully!")
    logger.info(f"📊 Generated {image_count} image patches from {subjects_count} subjects")
    if mask_count > 0:
        logger.info(f"📊 Generated {mask_count} mask patches")
    logger.info(f"📁 Output saved to: {output_path}")
    logger.info(f"📄 Metadata saved to: {metadata_file}")


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
def metadata_process_command(args):
    """Process metadata files (e.g., glomeruli scoring matrix) via CLI."""
    logger = get_logger("eq.metadata_process")
    logger.info("🔄 Processing metadata file...")

    try:
        exported = process_metadata_file(
            input_file=args.input_file,
            output_dir=args.output_dir,
            file_type=args.file_type,
        )
        print("✅ Metadata processed. Exported files:")
        for k, v in exported.items():
            print(f"  - {k}: {v}")
    except Exception as e:
        logger.error(f"❌ Metadata processing failed: {e}")
        print(f"❌ Metadata processing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# === Derived data audit (images/masks) ===
def audit_derived_command(args):
    """Audit derived_data directory for 1:1 pairs, size match, and binary masks.

    Writes a JSON report under <data_dir>/cache/audit_masks.json
    """
    logger = get_logger("eq.audit_derived")
    data_dir = Path(args.data_dir)
    img_dir = data_dir / 'image_patches'
    msk_dir = data_dir / 'mask_patches'
    cache_dir = data_dir / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not img_dir.exists() or not msk_dir.exists():
        print(f"❌ Expected subdirs not found: {img_dir} and {msk_dir}")
        sys.exit(1)

    from PIL import Image
    import numpy as _np
    import json as _json

    images = sorted([p for p in img_dir.glob('*.png')])
    total_images = len(images)

    missing_masks = []
    size_mismatch = []
    non_binary_masks = []

    def _mask_path(p: Path) -> Path:
        return msk_dir / f"{p.stem}_mask{p.suffix}"

    for p in images:
        mp = _mask_path(p)
        if not mp.exists():
            # Fallback common renames
            alt = Path(str(p).replace('.jpeg', '.png').replace('.jpg', '.png').replace('img_', 'mask_'))
            if alt.exists():
                mp = alt
            else:
                missing_masks.append({"image": str(p), "expected_mask": str(mp)})
                continue
        try:
            im = Image.open(p)
            mm = Image.open(mp)
            if im.size != mm.size:
                size_mismatch.append({"image": str(p), "mask": str(mp), "image_size": im.size, "mask_size": mm.size})
            arr = _np.array(mm)
            uniq = set(_np.unique(arr).tolist())
            if not (uniq.issubset({0,1}) or uniq.issubset({0,255})):
                non_binary_masks.append({"mask": str(mp), "unique_values": sorted(list(uniq))[:20]})
        except Exception as e:
            non_binary_masks.append({"mask": str(mp), "error": str(e)})

    report = {
        "data_dir": str(data_dir),
        "summary": {
            "total_images": total_images,
            "missing_masks": len(missing_masks),
            "size_mismatch": len(size_mismatch),
            "non_binary_masks": len(non_binary_masks)
        },
        "examples": {
            "missing_masks": missing_masks[:20],
            "size_mismatch": size_mismatch[:20],
            "non_binary_masks": non_binary_masks[:20]
        }
    }

    out_path = cache_dir / 'audit_masks.json'
    with open(out_path, 'w') as f:
        _json.dump(report, f, indent=2)

    print(f"✅ Audit complete. Report: {out_path}")
    print(_json.dumps(report["summary"], indent=2))


@log_function_call
def capabilities_command(args):
    """Report detected hardware capabilities and recommendations."""
    logger = get_logger("eq.capabilities")
    logger.info("🔍 Generating hardware capability report...")
    try:
        from eq.utils.hardware_detection import get_capability_report
    except Exception as e:
        print(f"❌ Unable to load hardware detection: {e}")
        print("Install PyTorch to enable capability reporting.")
        return
    report = get_capability_report()
    print(report)


@log_function_call
def visualize_command(args):
    """Visualize masks and images for debugging."""
    logger = get_logger("eq.visualize")
    
    try:
        if args.batch:
            # Batch visualization
            output_path = visualize_batch_masks(
                args.batch,
                output_path=args.output,
                max_masks=args.max_masks,
                title=args.title
            )
        elif args.image and args.mask:
            # Image-mask pair visualization
            output_path = visualize_image_mask_pair(
                args.image,
                args.mask,
                output_path=args.output,
                title=args.title
            )
        elif args.mask:
            # Single mask visualization
            output_path = visualize_mask(
                args.mask,
                output_path=args.output,
                title=args.title
            )
        else:
            print("❌ Please specify --mask, or both --image and --mask, or --batch")
            return
        
        print(f"✅ Visualization saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        print(f"❌ Error: {e}")
        raise


@log_function_call
def pipeline_command(args):
    """Run the production inference pipeline."""
    logger = get_logger("eq.pipeline")
    logger.info("🔄 Starting end-to-end production inference...")
    
    # Auto-determine cache and output directories
    data_dir = args.data_dir
    cache_dir = f"{data_dir}/cache"
    output_dir = "output"
    
    # Check for QUICK_TEST mode
    is_quick_test = os.getenv('QUICK_TEST') == 'true'
    if is_quick_test:
        args.epochs = 5
        args.batch_size = 4
        run_type = "development"
    else:
        run_type = "production"
    
    print("🚀 === PRODUCTION INFERENCE PIPELINE ===")
    print("Running end-to-end inference using pre-trained models...")
    print(f"Data directory: {data_dir}")
    print(f"Test data directory: {args.test_data_dir}")
    print(f"Cache directory: {cache_dir} (auto-detected)")
    print(f"Output directory: {output_dir} (auto-detected)")
    print(f"Base model path: {args.base_model_path}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Quick test: {is_quick_test}")
    if args.segmentation_model:
        print(f"Segmentation model: {args.segmentation_model}")
    
    # Run the pipeline
    run_pipeline(
        epochs=args.epochs,
        run_type=run_type,
        use_existing_models=True,
        data_dir=data_dir,
        cache_dir=cache_dir,
        segmentation_model=args.segmentation_model
    )


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
        args.epochs = 5  # Force 5 epochs for quick testing
        args.batch_size = min(args.batch_size, 4)  # Smaller batch size for quick testing
    
    # Get mode-aware batch size
    batch_size = _get_mode_aware_batch_size(mode_manager, args.batch_size)
    logger.info(f"📋 Training parameters: batch_size={batch_size}, epochs={args.epochs}")
    logger.info(f"🔧 Mode: {mode_manager.current_mode.value}")
    
    # Auto-determine cache and output directories
    data_dir = args.data_dir
    cache_dir = f"{data_dir}/cache"
    output_dir = "output"  # Always use the output directory in project root
    
    print("🚀 === SEGMENTATION TRAINING ===")
    print("Training segmentation model to find glomeruli...")
    print(f"Data directory: {data_dir}")
    print(f"Cache directory: {cache_dir} (auto-detected)")
    print(f"Output directory: {output_dir} (auto-detected)")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Quick test: {is_quick_test}")
    
    try:
        # Use the actual working pipeline for segmentation training with proper data paths
        from eq.pipeline.run_production_pipeline import run_pipeline

        success = run_pipeline(
            epochs=args.epochs,
            run_type="development" if is_quick_test else "production",
            use_existing_models=False,  # Force training new models
            data_dir=data_dir,
            cache_dir=cache_dir
        )

        if success:
            logger.info("✅ Segmentation training complete!")
            print("🎉 Segmentation training completed successfully!")
        else:
            logger.error("❌ Segmentation training failed!")
            print("❌ Segmentation training failed!")
            sys.exit(1)

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
    
    # Auto-determine cache and output directories
    data_dir = args.data_dir
    cache_dir = f"{data_dir}/cache"
    output_dir = "output"  # Always use the output directory in project root
    
    print("🚀 === QUANTIFICATION TRAINING ===")
    print("Training quantification model for endotheliosis scoring...")
    print(f"Data directory: {data_dir}")
    print(f"Cache directory: {cache_dir} (auto-detected)")
    print(f"Output directory: {output_dir} (auto-detected)")
    print(f"Segmentation model: {args.segmentation_model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Quick test: {is_quick_test}")
    
    try:
        # Use the actual working pipeline for quantification training with proper data paths
        from eq.pipeline.run_production_pipeline import run_pipeline
        
        run_pipeline(
            epochs=args.epochs, 
            run_type="development" if is_quick_test else "production", 
            use_existing_models=True,  # Use existing segmentation models for ROI extraction
            data_dir=data_dir,
            cache_dir=cache_dir
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
    parser.add_argument('--quiet', '-q', action='store_true', 
                       help='Run in quiet mode (no logs except errors)')
    parser.add_argument('--info', action='store_true',
                       help='Show hardware info and environment setup')
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
    data_parser.add_argument('--image-size', type=int, default=DEFAULT_IMAGE_SIZE, help='Image size for processing')
    data_parser.set_defaults(func=data_load_command)

    # Data processing command (NEW - direct processing to data/derived_data)
    process_parser = subparsers.add_parser('process-data', help='Process raw data into data/derived_data with auto-detection')
    process_parser.add_argument('--input-dir', required=True, help='Input directory with raw images (supports nested images/ and masks/ subdirs)')
    process_parser.add_argument('--output-dir', default='data/derived_data', help='Output directory (default: data/derived_data)')
    from eq.core.constants import DEFAULT_PATCH_SIZE, EXPECTED_PATCHES_PER_IMAGE
    process_parser.add_argument('--patch-size', type=int, default=DEFAULT_PATCH_SIZE, help=f'Patch size for processing (default: {DEFAULT_PATCH_SIZE}, expected ~{EXPECTED_PATCHES_PER_IMAGE} patches per image)')
    from eq.core.constants import DEFAULT_PATCH_OVERLAP
    process_parser.add_argument('--overlap', type=float, default=DEFAULT_PATCH_OVERLAP, help=f'Overlap between patches (default: {DEFAULT_PATCH_OVERLAP})')
    # auto-detect masks; no explicit flag needed
    process_parser.set_defaults(func=process_data_command)
    
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
    seg_parser.add_argument('--annotation-file', help='Annotation JSON file')
    seg_parser.add_argument('--image-size', type=int, default=DEFAULT_IMAGE_SIZE, help='Image size for processing')
    seg_parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    seg_parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    seg_parser.set_defaults(func=seg_command)
    
    # Quantification training command
    quant_parser = subparsers.add_parser('quant-endo', help='Train quantification model for endotheliosis scoring')
    quant_parser.add_argument('--data-dir', required=True, help='Training data directory')
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

    # Metadata processing command
    metadata_parser = subparsers.add_parser('metadata-process', help='Process metadata (e.g., glomeruli scoring matrix)')
    metadata_parser.add_argument('--input-file', required=True, help='Path to input metadata file (e.g., .xlsx)')
    metadata_parser.add_argument('--output-dir', required=True, help='Directory to write standardized outputs')
    metadata_parser.add_argument('--file-type', default='auto', choices=['auto','glomeruli_matrix','csv','json'], help='Type of metadata file (default: auto)')
    metadata_parser.set_defaults(func=metadata_process_command)

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
    
    # Production pipeline command
    production_parser = subparsers.add_parser('production', help='Run production inference pipeline')
    production_parser.add_argument('--data-dir', required=True, help='Path to data directory')
    production_parser.add_argument('--test-data-dir', required=True, help='Path to test data directory')
    production_parser.add_argument('--annotation-file', help='Path to annotation file')
    production_parser.add_argument('--base-model-path', default='segmentation_model_dir', help='Path to base model directory')
    production_parser.add_argument('--image-size', type=int, default=DEFAULT_IMAGE_SIZE, help='Image size for processing')
    production_parser.add_argument('--batch-size', type=int, default=8, help='Batch size for processing')
    production_parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    production_parser.add_argument('--segmentation-model', help='Segmentation model to use (glomeruli, mitochondria, etc.)')
    production_parser.set_defaults(func=pipeline_command)

    # Derived data audit command
    audit_parser = subparsers.add_parser('audit-derived', help='Audit derived_data image/mask pairs for binary masks and mapping')
    audit_parser.add_argument('--data-dir', required=True, help='Path to a derived_data project folder (with image_patches/ and mask_patches/)')
    audit_parser.set_defaults(func=audit_derived_command)
    
    # Visualization command
    viz_parser = subparsers.add_parser('visualize', help='Visualize masks and images for debugging')
    viz_parser.add_argument('--mask', help='Path to mask file to visualize')
    viz_parser.add_argument('--image', help='Path to image file (for image-mask pair visualization)')
    viz_parser.add_argument('--output', help='Output path for visualization')
    viz_parser.add_argument('--title', help='Title for the visualization')
    viz_parser.add_argument('--batch', nargs='+', help='Multiple mask paths for batch visualization')
    viz_parser.add_argument('--max-masks', type=int, default=16, help='Maximum masks for batch visualization')
    viz_parser.set_defaults(func=visualize_command)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Ensure important folders exist up front
    try:
        Path('data/raw_data').mkdir(parents=True, exist_ok=True)
        Path('data/derived_data').mkdir(parents=True, exist_ok=True)
        Path('models/segmentation/mitochondria').mkdir(parents=True, exist_ok=True)
        Path('models/segmentation/glomeruli').mkdir(parents=True, exist_ok=True)
        Path('test_output').mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    # Initialize mode manager for the session based on global --mode
    try:
        session_mode = EnvironmentMode(args.mode) if getattr(args, 'mode', None) else EnvironmentMode.AUTO
    except Exception:
        session_mode = EnvironmentMode.AUTO
    
    # Use the global mode manager from auto-setup to avoid duplicates
    if _auto_mode_manager:
        mode_manager = _auto_mode_manager
        # Only change mode if explicitly requested and different
        if session_mode != _auto_mode_manager.current_mode:
            mode_manager.set_mode(session_mode)
    else:
        # Fallback: only create new one if auto-setup failed
        mode_manager = ModeManager(mode=session_mode)
    
    # Show hardware info if requested
    if args.info:
        from eq.utils.hardware_detection import get_hardware_capabilities
        hardware_capabilities = get_hardware_capabilities()
        if hardware_capabilities and hardware_capabilities.backend_type:
            print("🔧 Hardware Info:")
            print(f"   Platform: {hardware_capabilities.platform}")
            print(f"   Backend: {hardware_capabilities.backend_type.value.upper()}")
            print(f"   Memory: {hardware_capabilities.total_memory_gb:.1f}GB")
            print(f"   Hardware Tier: {hardware_capabilities.hardware_tier.value.upper()}")
            print(f"   Mode: {mode_manager.current_mode.value.upper()}")
            print()
        else:
            print("⚠️ Hardware detection unavailable")
            print()
    
    # Set up logging
    log_file = Path(args.log_file) if args.log_file else None
    log_level = logging.DEBUG if args.verbose else (logging.WARNING if args.quiet else logging.INFO)
    
    logger = setup_logging(
        level=log_level,
        log_file=log_file,
        verbose=args.verbose
    )
    
    # Only log essential info unless verbose mode
    if not args.quiet:
        logger.info(f"Starting eq command: {args.command}")
        if args.verbose:
            logger.debug(f"Arguments: {vars(args)}")
            logger.debug(f"Mode: {mode_manager.current_mode.value}")
    
    try:
        args.func(args)
        if not args.quiet:
            logger.info("Command completed successfully!")
    except Exception as e:
        logger.error(f"Command failed: {str(e)}")
        
        # Handle mode-specific error recovery
        _handle_mode_specific_errors(e, mode_manager, args.command)
        
        if args.verbose:
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == '__main__':
    main()
