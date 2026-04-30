#!/usr/bin/env python3
"""Main CLI entry point for the endotheliosis quantifier package."""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from eq.utils.logger import get_logger, log_function_call, setup_logging
from eq.utils.mode_manager import EnvironmentMode, ModeManager
from eq.utils.paths import (
    get_output_path,
    get_runtime_mitochondria_data_path,
    get_runtime_models_path,
    get_runtime_output_path,
    get_runtime_raw_data_path,
)

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Optional conda environment activation (opt-in via EQ_AUTO_CONDA=1)
from eq import ensure_conda_environment

if os.environ.get('EQ_AUTO_CONDA', '0') == '1':
    ensure_conda_environment()


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

        return mode_manager

    except Exception:
        raise RuntimeError('CLI environment auto-setup failed') from None


# Run automatic setup when module is imported
_auto_mode_manager = auto_setup_environment()


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
    from eq.quantification.run_endotheliosis_quantification_workflow import (
        run_endotheliosis_quantification_inputs,
    )

    return run_endotheliosis_quantification_inputs


def _load_labelstudio_bootstrap():
    """Import Label Studio bootstrap lazily to keep CLI startup lightweight."""
    from eq.labelstudio.bootstrap import run_bootstrap

    return run_bootstrap


def _load_build_current_accessible_cohorts():
    """Import cohort-manifest build helpers lazily."""
    from eq.quantification import build_current_accessible_cohorts

    return build_current_accessible_cohorts


def _load_build_dox_mask_quality_audit():
    """Import optional Dox mask-provenance helpers lazily."""
    from eq.quantification import build_dox_mask_quality_audit

    return build_dox_mask_quality_audit


def _load_build_dox_scored_only_resolution_audit():
    """Import Dox scored-only image resolution helpers lazily."""
    from eq.quantification import build_dox_scored_only_resolution_audit

    return build_dox_scored_only_resolution_audit


def _load_organize_lucchi_dataset():
    """Import the Lucchi organizer only when requested from the CLI."""
    from eq.data_management.organize_lucchi_dataset import organize_lucchi_dataset

    return organize_lucchi_dataset


def _load_glomeruli_overcoverage_audit():
    """Import the glomeruli overcoverage audit runner lazily."""
    from eq.training.glomeruli_overcoverage_audit import run_overcoverage_audit

    return run_overcoverage_audit


def _load_visualizers():
    """Import visualization helpers only when the visualize command is used."""
    from eq.utils.image_mask_vis import (
        visualize_batch_masks,
        visualize_image_mask_pair,
        visualize_mask,
    )

    return visualize_mask, visualize_image_mask_pair, visualize_batch_masks


# Functions needed for loading pre-trained models
# from eq.models.feature_extractor import run_feature_extraction


@log_function_call
def pipeline_orchestrator_command(args):
    """Pipeline orchestrator that runs the specified pipeline stage."""
    logger = get_logger('eq.pipeline_orchestrator')
    logger.info('🚀 Starting pipeline orchestrator...')

    print('🚀 === ENDOTHELIOSIS QUANTIFIER PIPELINE ===')
    print(f'Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

    # Use the auto-detected mode manager
    if _auto_mode_manager:
        current_mode = _auto_mode_manager.current_mode
        current_config = _auto_mode_manager.current_config
        print(f'🎯 Current Mode: {current_mode.value.upper()}')
        print(f'   Batch Size: {current_config.batch_size or "Auto"}')
        print(f'   Device: {_auto_mode_manager.get_device_recommendation()}')

    # Check for QUICK_TEST mode
    import os

    is_quick_test = os.getenv('QUICK_TEST') == 'true'
    if is_quick_test:
        print('🔍 QUICK_TEST mode enabled - using fast validation settings')
        epochs = 5
        batch_size = 4
    else:
        print('🚀 PRODUCTION mode - using full settings')
        epochs = 50
        batch_size = 8

    print()
    print('Available pipeline stages:')
    print(
        '  1. Quantification (quant-endo) - Run burden-index quantification plus comparator artifacts'
    )
    print()
    print('Usage:')
    print(
        '  python -m eq quant-endo             # Run quantification contract + burden/comparator models'
    )
    print()
    print('❌ No interactive input required. Use specific commands above.')
    print('❌ This orchestrator is for documentation only.')


# Note: process_annotations_command removed - use PNG exports from Label Studio instead


def extract_images_command(args):
    """Extract large images from TIF files without patchifying them."""
    logger = get_logger('eq.extract_images')
    logger.info('🔄 Starting image extraction pipeline...')

    from pathlib import Path

    from eq.processing.image_mask_preprocessing import extract_large_images

    # Validate input directory
    input_path = Path(args.input_dir)
    if not input_path.exists():
        logger.error(f'❌ Input directory does not exist: {input_path}')
        sys.exit(1)

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f'📁 Output directory: {output_path}')

    # Extract large images
    logger.info('📦 Extracting large images from TIF files...')
    counts = extract_large_images(
        input_root=str(input_path), output_root=str(output_path)
    )

    # Report results
    image_count = counts.get('images', 0)
    mask_count = counts.get('masks', 0)
    logger.info('✅ Extraction completed!')
    logger.info(f'📊 Extracted {image_count} images and {mask_count} masks')
    logger.info('📁 Output structure:')
    logger.info(f'   - {output_path}/images/ (extracted image files)')
    logger.info(f'   - {output_path}/masks/ (extracted mask files)')
    logger.info('🎉 Image extraction pipeline completed successfully!')


def organize_lucchi_command(args):
    """Organize the Lucchi dataset into the repo's expected train/test layout."""
    logger = get_logger('eq.organize_lucchi')
    logger.info('🔄 Starting Lucchi dataset organization...')

    organize_lucchi_dataset = _load_organize_lucchi_dataset()
    output_path = organize_lucchi_dataset(args.input_dir, args.output_dir)

    print(f'✅ Lucchi dataset organized at: {output_path}')
    logger.info('🎉 Lucchi dataset organization completed successfully!')


def validate_naming_command(args):
    """Validate subject naming conventions in image files."""
    logger = get_logger('eq.validate_naming')
    logger.info('🔍 Starting naming convention validation...')

    from pathlib import Path

    from eq.data_management.metadata_processor import validate_subject_naming

    # Validate input directory
    data_path = Path(args.data_dir)
    if not data_path.exists():
        logger.error(f'❌ Data directory does not exist: {data_path}')
        sys.exit(1)

    # Find images directory
    images_dir = data_path / 'images'
    if not images_dir.exists():
        logger.error(f'❌ Images directory not found: {images_dir}')
        logger.error('   Expected structure: data_dir/images/')
        sys.exit(1)

    logger.info(f'📁 Validating images in: {images_dir}')

    # Collect all image files
    all_image_files = []
    for subject_dir in images_dir.iterdir():
        if subject_dir.is_dir():
            images = (
                list(subject_dir.glob('*.png'))
                + list(subject_dir.glob('*.tif'))
                + list(subject_dir.glob('*.jpg'))
                + list(subject_dir.glob('*.jpeg'))
            )
            all_image_files.extend([img.name for img in images])

    if not all_image_files:
        logger.warning('⚠️  No image files found in the images directory')
        return

    logger.info(f'📊 Found {len(all_image_files)} image files to validate')

    # Validate naming conventions
    validation_results = validate_subject_naming(all_image_files, images_dir)

    # Print detailed results
    if validation_results['invalid_files']:
        logger.error('\n🚨 VALIDATION FAILED!')
        logger.error(f'   Invalid files: {len(validation_results["invalid_files"])}')
        logger.error(f'   Valid files: {len(validation_results["valid_files"])}')

        logger.error('\n❌ Invalid files:')
        for invalid_file in validation_results['invalid_files']:
            logger.error(f'   - {invalid_file}')

        if args.strict:
            logger.error('\n💥 Exiting with error code due to --strict flag')
            sys.exit(1)
    else:
        logger.info('\n✅ VALIDATION PASSED!')
        logger.info(
            f'   All {len(validation_results["valid_files"])} files have valid naming conventions'
        )
        logger.info(
            f'   Detected naming convention: {", ".join(validation_results["naming_conventions_detected"])}'
        )
        logger.info(
            f'   Found {len(validation_results["subject_ids_found"])} unique subjects'
        )

    # Print warnings
    if validation_results['warnings']:
        logger.warning('\n⚠️  Warnings:')
        for warning in validation_results['warnings']:
            logger.warning(f'   {warning}')

    logger.info('🎉 Naming validation completed!')


def mode_command(args):
    """Inspect and manage environment mode selection."""
    logger = get_logger('eq.mode')
    logger.info('⚙️ Managing environment mode...')

    # Initialize manager (respects persisted config at ~/.eq/config.json)
    manager = ModeManager()

    # Apply requested mode change if provided
    if getattr(args, 'set', None):
        try:
            # Validate the mode before setting it
            is_valid, reason = manager.validate_mode(EnvironmentMode(args.set))
            if not is_valid:
                logger.error(f"❌ Invalid mode '{args.set}': {reason}")
                print(f"❌ Cannot set mode to '{args.set}': {reason}")
                print(f'💡 Suggested mode: {manager.get_suggested_mode().value}')
                sys.exit(1)

            manager.switch_mode(EnvironmentMode(args.set))
            print(f'✅ Mode updated to: {manager.current_mode.value.upper()}')
        except Exception as e:
            logger.error(f"Failed to set mode '{args.set}': {e}")
            print(f"❌ Failed to set mode '{args.set}': {e}")
            sys.exit(1)

    # Validate mode if requested
    if getattr(args, 'validate', False):
        is_valid, reason = manager.validate_mode(manager.current_mode)
        status = '✅ VALID' if is_valid else '❌ INVALID'
        print(f'Validation: {status} - {reason}')
        if not is_valid:
            print(f'💡 Suggested mode: {manager.get_suggested_mode().value}')
            sys.exit(1)

    # Show summary if requested
    if getattr(args, 'show', False) or not (
        getattr(args, 'set', None) or getattr(args, 'validate', False)
    ):
        print(manager.get_mode_summary())


def _validate_mode_for_command(mode_manager: ModeManager, command: str) -> None:
    """Validate that the current mode is suitable for the given command."""
    is_valid, reason = mode_manager.validate_mode(mode_manager.current_mode)

    if not is_valid:
        logger = get_logger('eq.cli.validation')
        logger.warning(f"Mode validation failed for command '{command}': {reason}")

        # For production commands, be more strict
        if (
            command == 'pipeline'
            and mode_manager.current_mode == EnvironmentMode.PRODUCTION
        ):
            print(f'❌ Production mode validation failed: {reason}')
            print(f'💡 Suggested mode: {mode_manager.get_suggested_mode().value}')
            print("💡 Use 'eq mode --set development' to switch to development mode")
            sys.exit(1)

        # For other commands, just warn
        print(f'⚠️  Warning: {reason}')
        print(f'💡 Consider switching to: {mode_manager.get_suggested_mode().value}')


def _get_mode_aware_batch_size(mode_manager: ModeManager, user_batch_size: int) -> int:
    """Get batch size considering mode and hardware capabilities."""
    if user_batch_size > 0:
        return user_batch_size

    # Auto-detect based on mode and hardware
    from eq.utils.hardware_detection import get_optimal_batch_size

    optimal_size = get_optimal_batch_size(mode_manager.current_mode.value)

    logger = get_logger('eq.cli.batch_size')
    logger.info(
        f'Auto-detected batch size for {mode_manager.current_mode.value} mode: {optimal_size}'
    )

    return optimal_size


def _handle_mode_specific_errors(
    e: Exception, mode_manager: ModeManager, command: str
) -> None:
    """Handle mode-specific error recovery and suggestions."""
    error_msg = str(e).lower()

    # Hardware-related errors
    if 'cuda' in error_msg or 'gpu' in error_msg:
        if mode_manager.current_mode == EnvironmentMode.PRODUCTION:
            print('❌ GPU/CUDA error in production mode')
            print('💡 Try switching to development mode: eq mode --set development')
        else:
            print('❌ GPU/CUDA error detected')
            print('💡 Try switching to CPU mode or check GPU drivers')

    # Memory-related errors
    elif 'memory' in error_msg or 'oom' in error_msg:
        if mode_manager.current_mode == EnvironmentMode.PRODUCTION:
            print('❌ Memory error in production mode')
            print('💡 Try reducing batch size or switching to development mode')
        else:
            print('❌ Memory error detected')
            print('💡 Try reducing batch size or closing other applications')

    # Backend-related errors
    elif 'mps' in error_msg:
        print('❌ MPS (Apple Silicon GPU) error detected')
        print('💡 Try switching to CPU mode: eq mode --set development')

    # Generic error handling
    else:
        print(f'❌ Unexpected error: {e}')
        if mode_manager.current_mode == EnvironmentMode.PRODUCTION:
            print('💡 Consider switching to development mode for debugging')


@log_function_call
def prepare_quant_contract_command(args):
    """Prepare contract artifacts for canonical quantification data."""
    logger = get_logger('eq.prepare_quant_contract')
    logger.info('🔄 Preparing quantification contract artifacts...')

    try:
        if not args.label_overrides:
            raise ValueError(
                'Direct prepare-quant-contract requires --label-overrides for the '
                'current reviewed-label contract. Preferred workflow: eq run-config '
                '--config configs/endotheliosis_quantification.yaml'
            )
        run_endotheliosis_quantification = _load_contract_first_quantification()
        outputs = run_endotheliosis_quantification(
            data_dir=Path(args.data_dir),
            segmentation_model=Path(args.segmentation_model),
            output_dir=Path(args.output_dir),
            mapping_file=Path(args.mapping_file) if args.mapping_file else None,
            annotation_source=args.annotation_source,
            score_source=args.score_source,
            label_overrides_path=Path(args.label_overrides)
            if args.label_overrides
            else None,
            apply_migration=args.apply_migration,
            stop_after='contract',
        )
        print('✅ Quantification contract artifacts created:')
        for key, value in outputs.items():
            print(f'  - {key}: {value}')
    except Exception as e:
        logger.error(f'❌ Quantification contract preparation failed: {e}')
        print(f'❌ Quantification contract preparation failed: {e}')
        import traceback

        traceback.print_exc()
        sys.exit(1)


@log_function_call
def cohort_manifest_command(args):
    """Build or inspect the runtime scored-cohort manifest."""
    logger = get_logger('eq.cohort_manifest')
    build_current_accessible_cohorts = _load_build_current_accessible_cohorts()

    result = build_current_accessible_cohorts(
        runtime_root=Path(args.runtime_root) if args.runtime_root else None,
        manifest_path=Path(args.manifest_path) if args.manifest_path else None,
    )
    logger.info('✅ Cohort manifest written to %s', result.manifest_path)
    print('✅ Runtime cohort manifest written:')
    print(f'  manifest: {result.manifest_path}')
    print(f'  summary: {result.summary_path}')
    print(f'  rows: {result.rows}')
    print(f'  admission_status_counts: {result.status_counts}')
    print(f'  lane_counts: {result.lane_counts}')


@log_function_call
def run_config_command(args):
    """Run a repository workflow YAML config."""
    from eq.run_config import run_config

    run_config(Path(args.config), dry_run=args.dry_run)


@log_function_call
def labelstudio_start_command(args):
    """Start local Label Studio and import image tasks for glomerulus grading."""
    run_bootstrap = _load_labelstudio_bootstrap()
    result = run_bootstrap(
        images_dir=Path(args.images),
        runtime_root=Path(args.runtime_root) if args.runtime_root else None,
        project_name=args.project_name,
        port=args.port,
        container_name=args.container_name,
        docker_image=args.docker_image,
        username=args.username,
        password=args.password,
        api_token=args.api_token,
        timeout_seconds=args.timeout_seconds,
        dry_run=args.dry_run,
    )
    print(result.message)
    print(f'Task manifest: {result.task_manifest_path}')
    print(f'Label Studio URL: {result.plan.url}')
    print(f'Login email: {args.username}')
    print(f'Login password: {args.password}')
    if result.project_url:
        print(f'Project URL: {result.project_url}')


@log_function_call
def dox_mask_quality_audit_command(args):
    """Build optional Dox mask import-provenance panels."""
    logger = get_logger('eq.dox_mask_quality_audit')
    build_dox_mask_quality_audit = _load_build_dox_mask_quality_audit()

    outputs = build_dox_mask_quality_audit(
        runtime_root=Path(args.runtime_root) if args.runtime_root else None,
        manifest_path=Path(args.manifest_path) if args.manifest_path else None,
        audit_path=Path(args.audit_path) if args.audit_path else None,
        panel_dir=Path(args.panel_dir) if args.panel_dir else None,
    )
    logger.info('✅ Dox mask provenance audit written to %s', outputs['audit'])
    print('✅ Dox mask provenance audit written:')
    print(f'  audit: {outputs["audit"]}')
    print(f'  summary: {outputs["summary"]}')
    print(f'  panel_dir: {outputs["panel_dir"]}')


@log_function_call
def dox_scored_only_resolution_audit_command(args):
    """Resolve Dox scored-only rows to Label Studio upload images."""
    logger = get_logger('eq.dox_scored_only_resolution_audit')
    build_dox_scored_only_resolution_audit = (
        _load_build_dox_scored_only_resolution_audit()
    )

    outputs = build_dox_scored_only_resolution_audit(
        runtime_root=Path(args.runtime_root) if args.runtime_root else None,
        manifest_path=Path(args.manifest_path) if args.manifest_path else None,
        upload_root=Path(args.upload_root) if args.upload_root else None,
        audit_path=Path(args.audit_path) if args.audit_path else None,
        smoke_manifest_path=Path(args.smoke_manifest_path)
        if args.smoke_manifest_path
        else None,
        localized_image_root=Path(args.localized_image_root)
        if args.localized_image_root
        else None,
        update_manifest=not args.no_update_manifest,
    )
    logger.info('✅ Dox scored-only resolution audit written to %s', outputs['audit'])
    print('✅ Dox scored-only resolution audit written:')
    print(f'  audit: {outputs["audit"]}')
    print(f'  smoke_manifest: {outputs["smoke_manifest"]}')
    print(f'  localized_image_root: {outputs["localized_image_root"]}')
    print(f'  summary: {outputs["summary"]}')
    print(f'  counts: {outputs["counts"]}')


@log_function_call
def metadata_process_command(args):
    """Process metadata files (e.g., glomeruli scoring matrix) via CLI."""
    logger = get_logger('eq.metadata_process')
    logger.info('🔄 Processing metadata file...')

    try:
        process_metadata_file = _load_process_metadata_file()
        exported = process_metadata_file(
            input_file=args.input_file,
            output_dir=args.output_dir,
            file_type=args.file_type,
        )
        print('✅ Metadata processed. Exported files:')
        for k, v in exported.items():
            print(f'  - {k}: {v}')
    except Exception as e:
        logger.error(f'❌ Metadata processing failed: {e}')
        print(f'❌ Metadata processing failed: {e}')
        import traceback

        traceback.print_exc()
        sys.exit(1)


@log_function_call
def capabilities_command(args):
    """Report detected hardware capabilities and recommendations."""
    logger = get_logger('eq.capabilities')
    logger.info('🔍 Generating hardware capability report...')
    try:
        from eq.utils.hardware_detection import get_capability_report
    except Exception as e:
        print(f'❌ Unable to load hardware detection: {e}')
        print('Install PyTorch to enable capability reporting.')
        return
    report = get_capability_report()
    print(report)


@log_function_call
def visualize_command(args):
    """Visualize masks and images for debugging."""
    logger = get_logger('eq.visualize')

    try:
        visualize_mask, visualize_image_mask_pair, visualize_batch_masks = (
            _load_visualizers()
        )
        if args.batch:
            # Batch visualization
            output_path = visualize_batch_masks(
                args.batch,
                output_path=args.output,
                max_masks=args.max_masks,
                title=args.title,
            )
        elif args.image and args.mask:
            # Image-mask pair visualization
            output_path = visualize_image_mask_pair(
                args.image, args.mask, output_path=args.output, title=args.title
            )
        elif args.mask:
            # Single mask visualization
            output_path = visualize_mask(
                args.mask, output_path=args.output, title=args.title
            )
        else:
            print('❌ Please specify --mask, or both --image and --mask, or --batch')
            return

        print(f'✅ Visualization saved to: {output_path}')

    except Exception as e:
        logger.error(f'Visualization failed: {e}')
        print(f'❌ Error: {e}')
        raise


@log_function_call
def quant_endo_command(args):
    """Run the canonical endotheliosis quantification workflow."""
    logger = get_logger('eq.quant_endo')
    logger.info('🔄 Starting endotheliosis quantification workflow...')

    # Get mode manager and validate mode
    mode_manager = ModeManager()
    _validate_mode_for_command(mode_manager, 'quant-endo')

    logger.info(f'🔧 Mode: {mode_manager.current_mode.value}')

    project_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    mapping_file = Path(args.mapping_file) if args.mapping_file else None
    segmentation_model = Path(args.segmentation_model)

    print('🚀 === QUANTIFICATION TRAINING ===')
    print('Running endotheliosis burden-index quantification with comparator models...')
    print(f'Project directory: {project_dir}')
    print(f'Output directory: {output_dir}')
    print(f'Segmentation model: {segmentation_model}')
    print(f'Score source: {args.score_source}')
    print(
        f'Annotation source: {args.annotation_source if args.annotation_source else "auto-detect"}'
    )
    print(f'Mapping file: {mapping_file if mapping_file else "None"}')
    print(f'Apply migration: {args.apply_migration}')
    print(f'Stop after: {args.stop_after}')

    try:
        if not args.label_overrides:
            raise ValueError(
                'Direct quant-endo requires --label-overrides for the current '
                'reviewed-label contract. Preferred workflow: eq run-config '
                '--config configs/endotheliosis_quantification.yaml'
            )
        run_endotheliosis_quantification = _load_contract_first_quantification()
        outputs = run_endotheliosis_quantification(
            data_dir=project_dir,
            segmentation_model=segmentation_model,
            output_dir=output_dir,
            mapping_file=mapping_file,
            annotation_source=args.annotation_source,
            score_source=args.score_source,
            label_overrides_path=Path(args.label_overrides)
            if args.label_overrides
            else None,
            apply_migration=args.apply_migration,
            stop_after=args.stop_after,
        )

        print('✅ Quantification pipeline outputs:')
        for key, value in outputs.items():
            print(f'  - {key}: {value}')
        logger.info('✅ Quantification training complete!')

    except Exception as e:
        logger.error(f'❌ Quantification training failed: {e}')
        print(f'❌ Quantification training failed: {e}')
        import traceback

        traceback.print_exc()
        sys.exit(1)


@log_function_call
def backup_project_data_command(args):
    """Create a timestamped backup snapshot for project data."""
    logger = get_logger('eq.backup_project_data')
    create_backup_snapshot = _load_backup_snapshot()

    project_dir = Path(args.project_dir)
    sources = [
        project_dir / 'images',
        project_dir / 'masks',
        project_dir / 'metadata' / 'subject_metadata.xlsx',
    ]
    if args.include_derived and args.derived_dir:
        sources.append(Path(args.derived_dir))

    artifact = create_backup_snapshot(
        sources=sources, backup_dir=Path(args.backup_dir), label=args.label
    )
    logger.info('✅ Backup created at %s', artifact.backup_root)
    print(f'✅ Backup created at {artifact.backup_root}')
    print(f'  manifest.files: {artifact.manifest_files}')
    print(f'  manifest.sha256: {artifact.manifest_sha256}')
    print(f'  manifest.meta: {artifact.manifest_meta}')


@log_function_call
def glomeruli_overcoverage_audit_command(args):
    """Run deterministic probability and threshold audit for glomeruli candidates."""
    run_overcoverage_audit = _load_glomeruli_overcoverage_audit()
    summary = run_overcoverage_audit(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Endotheliosis Quantifier Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  eq quant-endo --data-dir <raw-project> --segmentation-model <model.pkl>
  eq prepare-quant-contract --data-dir <raw-project> --segmentation-model <model.pkl>
  eq run-config --config configs/glomeruli_candidate_comparison.yaml --dry-run
  eq capabilities
  eq mode --set development --show --validate
  eq orchestrator  # Interactive menu
        """,
    )

    # Add global options
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose logging with more details',
    )
    parser.add_argument(
        '--quiet',
        '-q',
        action='store_true',
        help='Run in quiet mode (no logs except errors)',
    )
    parser.add_argument(
        '--info', action='store_true', help='Show hardware info and environment setup'
    )
    parser.add_argument('--log-file', type=str, help='Write logs to specified file')
    parser.add_argument(
        '--mode',
        choices=['auto', 'development', 'production'],
        default='auto',
        help='Select environment mode for this session (default: auto)',
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Pipeline orchestrator command (interactive menu)
    orchestrator_parser = subparsers.add_parser(
        'orchestrator',
        help='Interactive pipeline orchestrator with menu selection',
        description='Interactive pipeline orchestrator with menu selection',
    )
    orchestrator_parser.set_defaults(func=pipeline_orchestrator_command)

    # Extract images command (for large TIF files)
    extract_parser = subparsers.add_parser(
        'extract-images', help='Extract large images from TIF files without patchifying'
    )
    extract_parser.add_argument(
        '--input-dir',
        required=True,
        help='Input directory with TIF files (e.g., mitochondria data)',
    )
    extract_parser.add_argument(
        '--output-dir', required=True, help='Output directory for extracted images'
    )
    extract_parser.set_defaults(func=extract_images_command)

    # Lucchi dataset organization command
    lucchi_parser = subparsers.add_parser(
        'organize-lucchi',
        help='Organize Lucchi img/ and label/ stacks into train/test folders',
    )
    lucchi_parser.add_argument(
        '--input-dir',
        required=True,
        help='Input directory containing Lucchi img/ and label/ folders',
    )
    lucchi_parser.add_argument(
        '--output-dir',
        default=str(get_runtime_mitochondria_data_path()),
        help='Output directory for organized Lucchi data (default: active runtime raw_data/mitochondria_data)',
    )
    lucchi_parser.set_defaults(func=organize_lucchi_command)

    # Validate naming command
    validate_parser = subparsers.add_parser(
        'validate-naming', help='Validate subject naming conventions in image files'
    )
    validate_parser.add_argument(
        '--data-dir',
        required=True,
        help='Data directory containing images/ and masks/ subdirectories',
    )
    validate_parser.add_argument(
        '--strict',
        action='store_true',
        help='Exit with error code if any invalid files are found',
    )
    validate_parser.set_defaults(func=validate_naming_command)

    # Quantification training command
    quant_parser = subparsers.add_parser(
        'quant-endo',
        help='Run endotheliosis burden-index quantification with comparator artifacts',
    )
    quant_parser.add_argument(
        '--data-dir',
        required=True,
        help='Raw cohort/project directory containing images/, masks/, and optionally scores/ or metadata/subject_metadata.xlsx',
    )
    quant_parser.add_argument(
        '--segmentation-model', required=True, help='Path to trained segmentation model'
    )
    quant_parser.add_argument(
        '--score-source',
        default='auto',
        choices=['auto', 'labelstudio', 'spreadsheet'],
        help='Preferred score source contract; labelstudio is the intended default for scored-cohort workflows',
    )
    quant_parser.add_argument(
        '--annotation-source',
        help='Label Studio annotation export path or git source spec like git:REV:path/to/annotations.json',
    )
    quant_parser.add_argument(
        '--mapping-file',
        help='CSV mapping legacy image stems to canonical subject_image_id values',
    )
    quant_parser.add_argument(
        '--label-overrides',
        help=(
            'Reviewed rubric label override CSV. Must be a stable derived input, '
            'not a prior output/quantification_results artifact.'
        ),
    )
    quant_parser.add_argument(
        '--output-dir',
        default=str(
            get_runtime_output_path()
            / 'quantification_results'
            / 'endotheliosis_quantification'
        ),
        help='Directory to write quantification outputs',
    )
    quant_parser.add_argument(
        '--apply-migration',
        action='store_true',
        help='Apply renames in place instead of producing a dry-run migration report',
    )
    quant_parser.add_argument(
        '--stop-after',
        default='model',
        choices=['contract', 'roi', 'embeddings', 'model'],
        help='Stop after a specific contract/scoring stage',
    )
    quant_parser.set_defaults(func=quant_endo_command)

    contract_parser = subparsers.add_parser(
        'prepare-quant-contract',
        help='Build score-linked image/mask contract artifacts before quantification modeling',
    )
    contract_parser.add_argument(
        '--data-dir',
        required=True,
        help='Raw cohort/project directory containing images/, masks/, and optionally scores/ or metadata/subject_metadata.xlsx',
    )
    contract_parser.add_argument(
        '--segmentation-model',
        required=True,
        help='Path to trained segmentation model used later for embeddings',
    )
    contract_parser.add_argument(
        '--score-source',
        default='auto',
        choices=['auto', 'labelstudio', 'spreadsheet'],
        help='Preferred score source contract; labelstudio is the intended default for scored-cohort workflows',
    )
    contract_parser.add_argument(
        '--annotation-source',
        help='Label Studio annotation export path or git source spec like git:REV:path/to/annotations.json',
    )
    contract_parser.add_argument(
        '--mapping-file',
        help='CSV mapping legacy image stems to canonical subject_image_id values',
    )
    contract_parser.add_argument(
        '--label-overrides',
        help=(
            'Reviewed rubric label override CSV. Must be a stable derived input, '
            'not a prior output/quantification_results artifact.'
        ),
    )
    contract_parser.add_argument(
        '--output-dir',
        default=str(
            get_runtime_output_path()
            / 'quantification_results'
            / 'endotheliosis_quantification'
        ),
        help='Directory to write contract preparation artifacts',
    )
    contract_parser.add_argument(
        '--apply-migration',
        action='store_true',
        help='Apply renames in place instead of producing a dry-run migration report',
    )
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

    run_config_parser = subparsers.add_parser(
        'run-config',
        help='Run a repository workflow YAML config',
        description='Run a supported repository workflow directly from its YAML config.',
    )
    run_config_parser.add_argument(
        '--config', required=True, help='Workflow YAML config to run.'
    )
    run_config_parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print commands without launching training or analysis.',
    )
    run_config_parser.set_defaults(func=run_config_command)

    labelstudio_parser = subparsers.add_parser(
        'labelstudio',
        help='Manage local Label Studio grading workflows',
        description='Bootstrap local Label Studio projects for glomerulus grading.',
    )
    labelstudio_subparsers = labelstudio_parser.add_subparsers(
        dest='labelstudio_command', help='Label Studio commands'
    )
    labelstudio_start_parser = labelstudio_subparsers.add_parser(
        'start',
        help='Start local Label Studio and import an image directory',
        description='Point at an image directory and open a configured local Label Studio glomerulus-grading project.',
    )
    labelstudio_start_parser.add_argument(
        '--images',
        required=True,
        help='Directory of images to import recursively into Label Studio.',
    )
    labelstudio_start_parser.add_argument(
        '--project-name',
        default='EQ Glomerulus Grading',
        help='Label Studio project title.',
    )
    labelstudio_start_parser.add_argument(
        '--runtime-root',
        help='Runtime directory for Label Studio files. Defaults to active runtime root/labelstudio.',
    )
    labelstudio_start_parser.add_argument(
        '--port', type=int, default=8080, help='Local host port for Label Studio.'
    )
    labelstudio_start_parser.add_argument(
        '--container-name',
        default='eq-labelstudio',
        help='Docker container name for local Label Studio.',
    )
    labelstudio_start_parser.add_argument(
        '--docker-image',
        default='heartexlabs/label-studio:latest',
        help='Docker image to use for Label Studio.',
    )
    labelstudio_start_parser.add_argument(
        '--username',
        default='eq-admin@example.local',
        help='Local Label Studio admin username.',
    )
    labelstudio_start_parser.add_argument(
        '--password',
        default='eq-labelstudio',
        help='Local Label Studio admin password.',
    )
    labelstudio_start_parser.add_argument(
        '--api-token',
        default='eq-local-token',
        help='Local Label Studio API token for project bootstrap.',
    )
    labelstudio_start_parser.add_argument(
        '--timeout-seconds',
        type=int,
        default=60,
        help='Seconds to wait for Label Studio API readiness.',
    )
    labelstudio_start_parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Write the task manifest and print the plan without Docker or API calls.',
    )
    labelstudio_start_parser.set_defaults(func=labelstudio_start_command)

    overcoverage_parser = subparsers.add_parser(
        'glomeruli-overcoverage-audit',
        help='Audit glomeruli overcoverage across probability thresholds before retraining',
        description='Run deterministic no-training probability and threshold audit for glomeruli transfer and scratch candidates.',
    )
    overcoverage_parser.add_argument(
        '--run-id', required=True, help='Run directory name for this audit'
    )
    overcoverage_parser.add_argument(
        '--transfer-model-path',
        required=True,
        help='Current-namespace transfer candidate artifact',
    )
    overcoverage_parser.add_argument(
        '--scratch-model-path',
        required=True,
        help='Current-namespace scratch/no-mito-base candidate artifact',
    )
    overcoverage_parser.add_argument(
        '--data-dir',
        required=True,
        help='Supported glomeruli raw-data root or manifest-backed cohorts root',
    )
    overcoverage_parser.add_argument(
        '--output-dir', help='Optional output root; run id is appended when supplied'
    )
    overcoverage_parser.add_argument(
        '--thresholds',
        default='0.01,0.05,0.10,0.25,0.50',
        help='Comma-separated foreground probability threshold grid',
    )
    overcoverage_parser.add_argument(
        '--image-size', type=int, default=256, help='Model input size'
    )
    overcoverage_parser.add_argument(
        '--crop-size', type=int, default=512, help='Deterministic validation crop size'
    )
    overcoverage_parser.add_argument(
        '--examples-per-category',
        type=int,
        default=2,
        help='Examples per background/boundary/positive category',
    )
    overcoverage_parser.add_argument(
        '--device',
        choices=['mps', 'cuda', 'cpu'],
        default='cpu',
        help='Device label recorded in audit provenance',
    )
    overcoverage_parser.add_argument(
        '--negative-crop-manifest',
        help='Optional validated negative/background crop manifest path',
    )
    overcoverage_parser.set_defaults(func=glomeruli_overcoverage_audit_command)

    dox_mask_quality_parser = subparsers.add_parser(
        'dox-mask-quality-audit',
        help='Build optional Dox mask import-provenance panels',
        description='Document Dox recovered brushlabel masks that are already accepted as manual-mask training labels.',
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
        help='Directory for visual provenance panel PNGs. Defaults to raw_data/cohorts/vegfri_dox/metadata/mask_quality_panels.',
    )
    dox_mask_quality_parser.set_defaults(func=dox_mask_quality_audit_command)

    dox_scored_only_parser = subparsers.add_parser(
        'dox-scored-only-resolution-audit',
        help='Resolve Dox scored-only rows to exact Label Studio upload images',
        description='Build the Dox scored-only resolution audit and clean scored-no-mask smoke manifest.',
    )
    dox_scored_only_parser.add_argument(
        '--runtime-root',
        help='Active runtime root to use instead of EQ_RUNTIME_ROOT or ~/ProjectsRuntime/endotheliosis_quantifier.',
    )
    dox_scored_only_parser.add_argument(
        '--manifest-path',
        help='Input manifest path. Defaults to the active runtime raw_data/cohorts/manifest.csv.',
    )
    dox_scored_only_parser.add_argument(
        '--upload-root',
        help='Label Studio upload media root. Defaults to the Dox project label-studio/media/upload directory.',
    )
    dox_scored_only_parser.add_argument(
        '--audit-path',
        help='Output audit CSV. Defaults to raw_data/cohorts/vegfri_dox/metadata/dox_scored_only_resolution_audit.csv.',
    )
    dox_scored_only_parser.add_argument(
        '--smoke-manifest-path',
        help='Output clean smoke manifest CSV. Defaults to raw_data/cohorts/vegfri_dox/metadata/dox_scored_no_mask_smoke_manifest.csv.',
    )
    dox_scored_only_parser.add_argument(
        '--localized-image-root',
        help='Runtime folder where clean smoke images are copied. Defaults to raw_data/cohorts/vegfri_dox/scored_no_mask_smoke/images.',
    )
    dox_scored_only_parser.add_argument(
        '--no-update-manifest',
        action='store_true',
        help='Do not write Dox scored-no-mask transport columns back to the master manifest.',
    )
    dox_scored_only_parser.set_defaults(func=dox_scored_only_resolution_audit_command)

    backup_parser = subparsers.add_parser(
        'backup-project-data',
        help='Create a timestamped backup of project data before migration work',
    )
    backup_parser.add_argument(
        '--project-dir',
        required=True,
        help='Raw cohort/project directory containing images/, masks/, and metadata/subject_metadata.xlsx',
    )
    backup_parser.add_argument(
        '--backup-dir',
        default='backup',
        help='Directory in which to create the snapshot',
    )
    backup_parser.add_argument(
        '--label',
        default='project_backup',
        help='Label prefix for the snapshot directory',
    )
    backup_parser.add_argument(
        '--include-derived',
        action='store_true',
        help='Also copy a derived-data directory into the snapshot',
    )
    backup_parser.add_argument(
        '--derived-dir',
        help='Optional derived-data directory to include when --include-derived is set',
    )
    backup_parser.set_defaults(func=backup_project_data_command)

    # Metadata processing command
    metadata_parser = subparsers.add_parser(
        'metadata-process', help='Process metadata (e.g., glomeruli scoring matrix)'
    )
    metadata_parser.add_argument(
        '--input-file', required=True, help='Path to input metadata file (e.g., .xlsx)'
    )
    metadata_parser.add_argument(
        '--output-dir', required=True, help='Directory to write standardized outputs'
    )
    metadata_parser.add_argument(
        '--file-type',
        default='auto',
        choices=['auto', 'glomeruli_matrix', 'csv', 'json'],
        help='Type of metadata file (default: auto)',
    )
    metadata_parser.set_defaults(func=metadata_process_command)

    # Capabilities command
    capabilities_parser = subparsers.add_parser(
        'capabilities',
        help='Show hardware capabilities and recommendations',
        description='Show hardware capabilities and recommendations',
    )
    capabilities_parser.set_defaults(func=capabilities_command)

    # Mode command
    mode_parser = subparsers.add_parser(
        'mode',
        help='Inspect and manage environment mode',
        description='Inspect and manage environment mode',
    )
    mode_parser.add_argument(
        '--set',
        choices=['auto', 'development', 'production'],
        help='Set the environment mode',
    )
    mode_parser.add_argument(
        '--show',
        action='store_true',
        help='Show current mode and configuration summary',
    )
    mode_parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate current mode against hardware capabilities',
    )
    mode_parser.set_defaults(func=mode_command)

    # Visualization command
    viz_parser = subparsers.add_parser(
        'visualize', help='Visualize masks and images for debugging'
    )
    viz_parser.add_argument('--mask', help='Path to mask file to visualize')
    viz_parser.add_argument(
        '--image', help='Path to image file (for image-mask pair visualization)'
    )
    viz_parser.add_argument('--output', help='Output path for visualization')
    viz_parser.add_argument('--title', help='Title for the visualization')
    viz_parser.add_argument(
        '--batch', nargs='+', help='Multiple mask paths for batch visualization'
    )
    viz_parser.add_argument(
        '--max-masks',
        type=int,
        default=16,
        help='Maximum masks for batch visualization',
    )
    viz_parser.set_defaults(func=visualize_command)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Ensure runtime artifact folders exist up front.
    runtime_dirs = [
        get_runtime_raw_data_path(),
        get_output_path(),
        get_runtime_models_path() / 'segmentation' / 'mitochondria',
        get_runtime_models_path() / 'segmentation' / 'glomeruli',
        get_runtime_output_path(),
    ]
    for runtime_dir in runtime_dirs:
        runtime_dir.mkdir(parents=True, exist_ok=True)

    # Initialize mode manager for the session based on global --mode
    session_mode = (
        EnvironmentMode(args.mode) if getattr(args, 'mode', None) else EnvironmentMode.AUTO
    )

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
            print('🔧 Hardware Info:')
            print(f'   Platform: {hardware_capabilities.platform}')
            print(f'   Backend: {hardware_capabilities.backend_type.value.upper()}')
            print(f'   Memory: {hardware_capabilities.total_memory_gb:.1f}GB')
            print(
                f'   Hardware Tier: {hardware_capabilities.hardware_tier.value.upper()}'
            )
            print(f'   Mode: {mode_manager.current_mode.value.upper()}')
            print()
        else:
            print('⚠️ Hardware detection unavailable')
            print()

    # Set up logging
    log_file = Path(args.log_file) if args.log_file else None
    log_level = (
        logging.DEBUG
        if args.verbose
        else (logging.WARNING if args.quiet else logging.INFO)
    )

    logger = setup_logging(level=log_level, log_file=log_file, verbose=args.verbose)

    # Only log essential info unless verbose mode
    if not args.quiet:
        logger.info(f'Starting eq command: {args.command}')
        if args.verbose:
            logger.debug(f'Arguments: {vars(args)}')
            logger.debug(f'Mode: {mode_manager.current_mode.value}')

    try:
        args.func(args)
        if not args.quiet:
            logger.info('Command completed successfully!')
    except Exception as e:
        logger.error(f'Command failed: {str(e)}')

        # Handle mode-specific error recovery
        _handle_mode_specific_errors(e, mode_manager, args.command)

        if args.verbose:
            import traceback

            logger.error(f'Full traceback:\n{traceback.format_exc()}')
        sys.exit(1)


if __name__ == '__main__':
    main()
