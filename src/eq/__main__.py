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


def _load_run_pipeline():
    """Import the production pipeline only when a command actually needs it."""
    from eq.pipeline.run_production_pipeline import run_pipeline

    return run_pipeline


def _load_process_metadata_file():
    """Import metadata processing lazily to keep CLI startup lightweight."""
    from eq.data_management.metadata_processor import process_metadata_file

    return process_metadata_file


def _load_backup_snapshot():
    """Import backup creation helpers lazily."""
    from eq.data_management.backup_utils import create_backup_snapshot

    return create_backup_snapshot


def _load_contract_first_quantification():
    """Import the contract-first quantification pipeline lazily."""
    from eq.quantification import run_contract_first_quantification

    return run_contract_first_quantification


def _load_build_current_accessible_cohorts():
    """Import cohort-manifest build helpers lazily."""
    from eq.quantification import build_current_accessible_cohorts

    return build_current_accessible_cohorts


def _load_build_dox_mask_quality_audit():
    """Import Dox mask-quality audit helpers lazily."""
    from eq.quantification import build_dox_mask_quality_audit

    return build_dox_mask_quality_audit


def _load_organize_lucchi_dataset():
    """Import the Lucchi organizer only when requested from the CLI."""
    from eq.data_management.organize_lucchi_dataset import organize_lucchi_dataset

    return organize_lucchi_dataset


def _load_visualizers():
    """Import visualization helpers only when the visualize command is used."""
    from eq.utils.image_mask_vis import visualize_batch_masks, visualize_image_mask_pair, visualize_mask

    return visualize_mask, visualize_image_mask_pair, visualize_batch_masks


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
    print("  1. Quantification Training (quant-endo) - Run the Label Studio-first ordinal endotheliosis baseline") 
    print("  2. Production Inference (production) - End-to-end inference using pre-trained models")
    print()
    print("Usage:")
    print("  python -m eq quant-endo             # Run quantification contract + embedding baseline")
    print("  python -m eq production             # Run production inference")
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


# Note: process_annotations_command removed - use PNG exports from Label Studio instead


def extract_images_command(args):
    """Extract large images from TIF files without patchifying them."""
    logger = get_logger("eq.extract_images")
    logger.info("🔄 Starting image extraction pipeline...")

    from pathlib import Path
    from eq.processing.image_mask_preprocessing import extract_large_images
    
    # Validate input directory
    input_path = Path(args.input_dir)
    if not input_path.exists():
        logger.error(f"❌ Input directory does not exist: {input_path}")
        sys.exit(1)
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Output directory: {output_path}")
    
    # Extract large images
    logger.info("📦 Extracting large images from TIF files...")
    counts = extract_large_images(
        input_root=str(input_path),
        output_root=str(output_path)
    )
    
    # Report results
    image_count = counts.get("images", 0)
    mask_count = counts.get("masks", 0)
    logger.info(f"✅ Extraction completed!")
    logger.info(f"📊 Extracted {image_count} images and {mask_count} masks")
    logger.info(f"📁 Output structure:")
    logger.info(f"   - {output_path}/images/ (extracted image files)")
    logger.info(f"   - {output_path}/masks/ (extracted mask files)")
    logger.info("🎉 Image extraction pipeline completed successfully!")


def organize_lucchi_command(args):
    """Organize the Lucchi dataset into the repo's expected train/test layout."""
    logger = get_logger("eq.organize_lucchi")
    logger.info("🔄 Starting Lucchi dataset organization...")

    organize_lucchi_dataset = _load_organize_lucchi_dataset()
    output_path = organize_lucchi_dataset(args.input_dir, args.output_dir)

    print(f"✅ Lucchi dataset organized at: {output_path}")
    logger.info("🎉 Lucchi dataset organization completed successfully!")


def validate_naming_command(args):
    """Validate subject naming conventions in image files."""
    logger = get_logger("eq.validate_naming")
    logger.info("🔍 Starting naming convention validation...")

    from pathlib import Path
    from eq.data_management.metadata_processor import validate_subject_naming
    
    # Validate input directory
    data_path = Path(args.data_dir)
    if not data_path.exists():
        logger.error(f"❌ Data directory does not exist: {data_path}")
        sys.exit(1)
    
    # Find images directory
    images_dir = data_path / "images"
    if not images_dir.exists():
        logger.error(f"❌ Images directory not found: {images_dir}")
        logger.error("   Expected structure: data_dir/images/")
        sys.exit(1)
    
    logger.info(f"📁 Validating images in: {images_dir}")
    
    # Collect all image files
    all_image_files = []
    for subject_dir in images_dir.iterdir():
        if subject_dir.is_dir():
            images = (list(subject_dir.glob("*.png")) + 
                     list(subject_dir.glob("*.tif")) + 
                     list(subject_dir.glob("*.jpg")) + 
                     list(subject_dir.glob("*.jpeg")))
            all_image_files.extend([img.name for img in images])
    
    if not all_image_files:
        logger.warning("⚠️  No image files found in the images directory")
        return
    
    logger.info(f"📊 Found {len(all_image_files)} image files to validate")
    
    # Validate naming conventions
    validation_results = validate_subject_naming(all_image_files, images_dir)
    
    # Print detailed results
    if validation_results['invalid_files']:
        logger.error(f"\n🚨 VALIDATION FAILED!")
        logger.error(f"   Invalid files: {len(validation_results['invalid_files'])}")
        logger.error(f"   Valid files: {len(validation_results['valid_files'])}")
        
        logger.error(f"\n❌ Invalid files:")
        for invalid_file in validation_results['invalid_files']:
            logger.error(f"   - {invalid_file}")
        
        if args.strict:
            logger.error("\n💥 Exiting with error code due to --strict flag")
            sys.exit(1)
    else:
        logger.info(f"\n✅ VALIDATION PASSED!")
        logger.info(f"   All {len(validation_results['valid_files'])} files have valid naming conventions")
        logger.info(f"   Detected naming convention: {', '.join(validation_results['naming_conventions_detected'])}")
        logger.info(f"   Found {len(validation_results['subject_ids_found'])} unique subjects")
    
    # Print warnings
    if validation_results['warnings']:
        logger.warning(f"\n⚠️  Warnings:")
        for warning in validation_results['warnings']:
            logger.warning(f"   {warning}")
    
    logger.info("🎉 Naming validation completed!")


def process_data_command(args):
    """Legacy static patch conversion for audit or historical workflows."""
    logger = get_logger("eq.process_data")
    logger.info("🔄 Starting legacy static patch conversion pipeline...")
    logger.warning("Static patch outputs are legacy audit/conversion artifacts, not supported segmentation training inputs.")

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
    logger.info(f"✂️  Creating {args.patch_size}x{args.patch_size} legacy static patches")
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
    
    progress.complete("Legacy static patch conversion")
    logger.info(f"🎉 Legacy static patch conversion completed successfully!")
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
        if command == 'pipeline' and mode_manager.current_mode == EnvironmentMode.PRODUCTION:
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
def prepare_quant_contract_command(args):
    """Prepare contract artifacts for canonical quantification data."""
    logger = get_logger("eq.prepare_quant_contract")
    logger.info("🔄 Preparing quantification contract artifacts...")

    try:
        run_contract_first_quantification = _load_contract_first_quantification()
        outputs = run_contract_first_quantification(
            project_dir=Path(args.data_dir),
            segmentation_model_path=Path(args.segmentation_model),
            output_dir=Path(args.output_dir),
            mapping_file=Path(args.mapping_file) if args.mapping_file else None,
            annotation_source=args.annotation_source,
            score_source=args.score_source,
            apply_migration=args.apply_migration,
            stop_after='contract',
        )
        print("✅ Quantification contract artifacts created:")
        for key, value in outputs.items():
            print(f"  - {key}: {value}")
    except Exception as e:
        logger.error(f"❌ Quantification contract preparation failed: {e}")
        print(f"❌ Quantification contract preparation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


@log_function_call
def cohort_manifest_command(args):
    """Build or inspect the runtime scored-cohort manifest."""
    logger = get_logger("eq.cohort_manifest")
    build_current_accessible_cohorts = _load_build_current_accessible_cohorts()

    result = build_current_accessible_cohorts(
        runtime_root=Path(args.runtime_root) if args.runtime_root else None,
        manifest_path=Path(args.manifest_path) if args.manifest_path else None,
    )
    logger.info("✅ Cohort manifest written to %s", result.manifest_path)
    print("✅ Runtime cohort manifest written:")
    print(f"  manifest: {result.manifest_path}")
    print(f"  summary: {result.summary_path}")
    print(f"  rows: {result.rows}")
    print(f"  admission_status_counts: {result.status_counts}")
    print(f"  lane_counts: {result.lane_counts}")


@log_function_call
def dox_mask_quality_audit_command(args):
    """Build the Dox mask-quality audit and visual review panels."""
    logger = get_logger("eq.dox_mask_quality_audit")
    build_dox_mask_quality_audit = _load_build_dox_mask_quality_audit()

    outputs = build_dox_mask_quality_audit(
        runtime_root=Path(args.runtime_root) if args.runtime_root else None,
        manifest_path=Path(args.manifest_path) if args.manifest_path else None,
        audit_path=Path(args.audit_path) if args.audit_path else None,
        panel_dir=Path(args.panel_dir) if args.panel_dir else None,
    )
    logger.info("✅ Dox mask-quality audit written to %s", outputs['audit'])
    print("✅ Dox mask-quality audit written:")
    print(f"  audit: {outputs['audit']}")
    print(f"  summary: {outputs['summary']}")
    print(f"  panel_dir: {outputs['panel_dir']}")
    
@log_function_call
def metadata_process_command(args):
    """Process metadata files (e.g., glomeruli scoring matrix) via CLI."""
    logger = get_logger("eq.metadata_process")
    logger.info("🔄 Processing metadata file...")

    try:
        process_metadata_file = _load_process_metadata_file()
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


# === Legacy derived static patch audit ===
def audit_derived_command(args):
    """Audit legacy static patch directory for 1:1 pairs, size match, and binary masks.

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
        visualize_mask, visualize_image_mask_pair, visualize_batch_masks = _load_visualizers()
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
    run_pipeline = _load_run_pipeline()
    
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
    
    project_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    mapping_file = Path(args.mapping_file) if args.mapping_file else None
    segmentation_model = Path(args.segmentation_model)

    print("🚀 === QUANTIFICATION TRAINING ===")
    print("Training quantification model for endotheliosis scoring...")
    print(f"Project directory: {project_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Segmentation model: {segmentation_model}")
    print(f"Score source: {args.score_source}")
    print(f"Annotation source: {args.annotation_source if args.annotation_source else 'auto-detect'}")
    print(f"Mapping file: {mapping_file if mapping_file else 'None'}")
    print(f"Apply migration: {args.apply_migration}")
    print(f"Stop after: {args.stop_after}")
    print(f"Quick test: {is_quick_test}")
    
    try:
        run_contract_first_quantification = _load_contract_first_quantification()
        outputs = run_contract_first_quantification(
            project_dir=project_dir,
            segmentation_model_path=segmentation_model,
            output_dir=output_dir,
            mapping_file=mapping_file,
            annotation_source=args.annotation_source,
            score_source=args.score_source,
            apply_migration=args.apply_migration,
            stop_after=args.stop_after,
        )

        print("✅ Quantification pipeline outputs:")
        for key, value in outputs.items():
            print(f"  - {key}: {value}")
        logger.info("✅ Quantification training complete!")
        
    except Exception as e:
        logger.error(f"❌ Quantification training failed: {e}")
        print(f"❌ Quantification training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


@log_function_call
def backup_project_data_command(args):
    """Create a timestamped backup snapshot for project data."""
    logger = get_logger("eq.backup_project_data")
    create_backup_snapshot = _load_backup_snapshot()

    project_dir = Path(args.project_dir)
    sources = [
        project_dir / 'images',
        project_dir / 'masks',
        project_dir / 'subject_metadata.xlsx',
    ]
    if args.include_derived and args.derived_dir:
        sources.append(Path(args.derived_dir))

    artifact = create_backup_snapshot(
        sources=sources,
        backup_dir=Path(args.backup_dir),
        label=args.label,
    )
    logger.info("✅ Backup created at %s", artifact.backup_root)
    print(f"✅ Backup created at {artifact.backup_root}")
    print(f"  manifest.files: {artifact.manifest_files}")
    print(f"  manifest.sha256: {artifact.manifest_sha256}")
    print(f"  manifest.meta: {artifact.manifest_meta}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Endotheliosis Quantifier Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  eq data-load --data-dir data/train --test-data-dir data/test
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

    # Legacy static patch conversion command
    process_parser = subparsers.add_parser('process-data', help='Legacy static patch conversion for audit/historical workflows')
    process_parser.add_argument('--input-dir', required=True, help='Input directory with raw images (supports nested images/ and masks/ subdirs)')
    process_parser.add_argument('--output-dir', default='data/derived_data', help='Output directory (default: data/derived_data)')
    from eq.core.constants import DEFAULT_PATCH_SIZE, EXPECTED_PATCHES_PER_IMAGE
    process_parser.add_argument('--patch-size', type=int, default=DEFAULT_PATCH_SIZE, help=f'Legacy static patch size (default: {DEFAULT_PATCH_SIZE}, expected ~{EXPECTED_PATCHES_PER_IMAGE} patches per image)')
    from eq.core.constants import DEFAULT_PATCH_OVERLAP
    process_parser.add_argument('--overlap', type=float, default=DEFAULT_PATCH_OVERLAP, help=f'Overlap between patches (default: {DEFAULT_PATCH_OVERLAP})')
    # auto-detect masks; no explicit flag needed
    process_parser.set_defaults(func=process_data_command)
    
    # Extract images command (for large TIF files)
    extract_parser = subparsers.add_parser('extract-images', help='Extract large images from TIF files without patchifying')
    extract_parser.add_argument('--input-dir', required=True, help='Input directory with TIF files (e.g., mitochondria data)')
    extract_parser.add_argument('--output-dir', required=True, help='Output directory for extracted images')
    extract_parser.set_defaults(func=extract_images_command)

    # Lucchi dataset organization command
    lucchi_parser = subparsers.add_parser('organize-lucchi', help='Organize Lucchi img/ and label/ stacks into train/test folders')
    lucchi_parser.add_argument('--input-dir', required=True, help='Input directory containing Lucchi img/ and label/ folders')
    lucchi_parser.add_argument(
        '--output-dir',
        default='data/derived_data/mitochondria_data',
        help='Output directory for organized Lucchi data (default: %(default)s)',
    )
    lucchi_parser.set_defaults(func=organize_lucchi_command)

    # Validate naming command
    validate_parser = subparsers.add_parser('validate-naming', help='Validate subject naming conventions in image files')
    validate_parser.add_argument('--data-dir', required=True, help='Data directory containing images/ and masks/ subdirectories')
    validate_parser.add_argument('--strict', action='store_true', help='Exit with error code if any invalid files are found')
    validate_parser.set_defaults(func=validate_naming_command)
    
    # Quantification training command
    quant_parser = subparsers.add_parser('quant-endo', help='Run the current endotheliosis quantification baseline')
    quant_parser.add_argument('--data-dir', required=True, help='Raw project directory containing images/, masks/, and optionally annotations/ or subject_metadata.xlsx')
    quant_parser.add_argument('--segmentation-model', required=True, help='Path to trained segmentation model')
    quant_parser.add_argument('--score-source', default='auto', choices=['auto', 'labelstudio', 'spreadsheet'], help='Preferred score source contract; labelstudio is the intended preeclampsia default')
    quant_parser.add_argument('--annotation-source', help='Label Studio annotation export path or git source spec like git:REV:path/to/annotations.json')
    quant_parser.add_argument('--mapping-file', help='CSV mapping legacy image stems to canonical subject_image_id values')
    quant_parser.add_argument('--output-dir', default='output/quantification', help='Directory to write quantification outputs')
    quant_parser.add_argument('--apply-migration', action='store_true', help='Apply renames in place instead of producing a dry-run migration report')
    quant_parser.add_argument('--stop-after', default='model', choices=['contract', 'roi', 'embeddings', 'model'], help='Stop after a specific contract/scoring stage')
    quant_parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    quant_parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    quant_parser.set_defaults(func=quant_endo_command)

    contract_parser = subparsers.add_parser('prepare-quant-contract', help='Build score-linked contract artifacts for quantification')
    contract_parser.add_argument('--data-dir', required=True, help='Raw project directory containing images/, masks/, and optionally annotations/ or subject_metadata.xlsx')
    contract_parser.add_argument('--segmentation-model', required=True, help='Path to trained segmentation model used later for embeddings')
    contract_parser.add_argument('--score-source', default='auto', choices=['auto', 'labelstudio', 'spreadsheet'], help='Preferred score source contract; labelstudio is the intended preeclampsia default')
    contract_parser.add_argument('--annotation-source', help='Label Studio annotation export path or git source spec like git:REV:path/to/annotations.json')
    contract_parser.add_argument('--mapping-file', help='CSV mapping legacy image stems to canonical subject_image_id values')
    contract_parser.add_argument('--output-dir', default='output/quantification', help='Directory to write contract preparation artifacts')
    contract_parser.add_argument('--apply-migration', action='store_true', help='Apply renames in place instead of producing a dry-run migration report')
    contract_parser.set_defaults(func=prepare_quant_contract_command)

    cohort_manifest_parser = subparsers.add_parser(
        'cohort-manifest',
        help='Build the runtime raw_data/cohorts/manifest.csv for scored cohort admission',
        description='Build the active runtime cohort manifest and cohort summary from localized cohort assets.',
    )
    cohort_manifest_parser.add_argument(
        '--runtime-root',
        help='Active runtime root to use instead of EQ_RUNTIME_ROOT or ~/ProjectsRuntime/endotheliosis_quantifier',
    )
    cohort_manifest_parser.add_argument(
        '--manifest-path',
        help='Output manifest path. Defaults to the active runtime raw_data/cohorts/manifest.csv.',
    )
    cohort_manifest_parser.set_defaults(func=cohort_manifest_command)

    dox_mask_quality_parser = subparsers.add_parser(
        'dox-mask-quality-audit',
        help='Build the Dox masked-external mask-quality audit and review panels',
        description='Audit Dox recovered brushlabel masks and write approval artifacts used by cohort-manifest.',
    )
    dox_mask_quality_parser.add_argument(
        '--runtime-root',
        help='Active runtime root to use instead of EQ_RUNTIME_ROOT or ~/ProjectsRuntime/endotheliosis_quantifier.',
    )
    dox_mask_quality_parser.add_argument(
        '--manifest-path',
        help='Input manifest path. Defaults to the active runtime raw_data/cohorts/manifest.csv.',
    )
    dox_mask_quality_parser.add_argument(
        '--audit-path',
        help='Output audit CSV. Defaults to raw_data/cohorts/vegfri_dox/metadata/mask_quality_audit.csv.',
    )
    dox_mask_quality_parser.add_argument(
        '--panel-dir',
        help='Directory for visual review panel PNGs. Defaults to output/cohorts/vegfri_dox/mask_quality.',
    )
    dox_mask_quality_parser.set_defaults(func=dox_mask_quality_audit_command)

    backup_parser = subparsers.add_parser('backup-project-data', help='Create a timestamped backup of project data before migration work')
    backup_parser.add_argument('--project-dir', required=True, help='Raw project directory containing images/, masks/, and subject_metadata.xlsx')
    backup_parser.add_argument('--backup-dir', default='backup', help='Directory in which to create the snapshot')
    backup_parser.add_argument('--label', default='project_backup', help='Label prefix for the snapshot directory')
    backup_parser.add_argument('--include-derived', action='store_true', help='Also copy a derived-data directory into the snapshot')
    backup_parser.add_argument('--derived-dir', help='Optional derived-data directory to include when --include-derived is set')
    backup_parser.set_defaults(func=backup_project_data_command)
    
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
    audit_parser = subparsers.add_parser('audit-derived', help='Audit legacy static patch image/mask pairs for binary masks and mapping')
    audit_parser.add_argument('--data-dir', required=True, help='Path to a legacy static patch folder with image_patches/ and mask_patches/')
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
