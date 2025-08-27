#!/usr/bin/env python3
"""
Segmentation Pipeline Orchestrator

This script manages the complete segmentation pipeline:
1. Mitochondria pretraining (transfer learning foundation)
2. Glomeruli fine-tuning (production model)

Usage:
    python -m eq.pipeline.segmentation_pipeline --stage mito
    python -m eq.pipeline.segmentation_pipeline --stage glomeruli
    python -m eq.pipeline.segmentation_pipeline --stage all
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

from eq.data_management.loaders import (
    get_scores_from_annotations,
    load_annotations_from_json,
    load_mitochondria_patches,
)
from eq.evaluation.glomeruli_evaluator import evaluate_glomeruli_model
# Training functions moved to eq.training module
# from eq.models.train_glomeruli_transfer_learning import (
#     train_glomeruli_transfer_learning_from_config,
# )
from eq.processing.convert_files import convert_tif_to_jpg
from eq.utils.logger import get_logger


def get_y(x):
    """Get mask path for image path - required for loading pretrained models."""
    return str(x).replace('.jpg', '.png').replace('img_', 'mask_')


class SegmentationPipeline:
    """Orchestrates the complete segmentation pipeline."""
    
    def __init__(self, config_path: str):
        """Initialize pipeline with configuration."""
        self.logger = get_logger("eq.segmentation_pipeline")
        
        # Set up file logging to logs/ folder
        import logging
        from datetime import datetime

        # Create logs directory if it doesn't exist
        Path("logs").mkdir(exist_ok=True)
        
        # Create timestamped log filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"logs/segmentation_pipeline_{timestamp}.log"
        
        # Set up file handler with detailed formatting
        file_handler = logging.FileHandler(log_filename)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.INFO)
        
        # Set up console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.INFO)
        
        # Add both handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        self.logger.setLevel(logging.INFO)
        
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.stage = self.config.get('name', 'unknown')
        
        # Log startup information
        self.logger.info(f"Pipeline initialized: {self.stage}")
        self.logger.info(f"Log file: {log_filename}")
        
        # Add testing indicator if in QUICK_TEST mode
        if os.getenv('QUICK_TEST', 'false').lower() == 'true':
            self.logger.info("TESTING RUN - QUICK_TEST MODE")
            self.logger.info("This is a TESTING run with reduced epochs and parameters.")
            self.logger.info("DO NOT use outputs from this run for production inference!")
            print("TESTING RUN - QUICK_TEST MODE")
            print("This is a TESTING run with reduced epochs and parameters.")
            print("DO NOT use outputs from this run for production inference!")
        
        # Check for FORCE_RERUN mode
        if os.getenv('FORCE_RERUN', 'false').lower() == 'true':
            self.logger.info("🔄 FORCE_RERUN MODE ENABLED")
            self.logger.info("All skip logic will be overridden - complete regeneration from scratch")
            print("🔄 FORCE_RERUN MODE ENABLED")
            print("All skip logic will be overridden - complete regeneration from scratch")
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load YAML configuration file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            self.logger.error(f"Failed to load config {config_path}: {e}")
            raise
    
    def _ensure_directories(self):
        """Create necessary output directories."""
        # Always use production directories - QUICK_TEST only affects training parameters
        standard_dirs = [
            "logs",
            "raw_data",
            "derived_data", 
            "output",
            "models/segmentation/mitochondria",
            "models/segmentation/glomeruli_xfer_learn",
            "models/regression"
        ]
        
        for directory in standard_dirs:
            Path(directory).mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Ensured standard directory exists: {directory}")
        
        # Also ensure config-specified directories exist
        directories = [
            self.config.get('logging', {}).get('plot_dir'),
            os.path.dirname(self.config.get('model', {}).get('checkpoint_path', '')),
        ]
        
        # Add data-specific directories
        data_config = self.config.get('data', {})
        if 'patches' in data_config:
            directories.append(data_config['patches'].get('output_dir'))
        if 'processed' in data_config:
            processed = data_config['processed']
            directories.extend([
                processed.get('train_dir'),
                processed.get('train_mask_dir'),
                processed.get('cache_dir')
            ])
        
        for directory in directories:
            if directory:
                Path(directory).mkdir(parents=True, exist_ok=True)
                self.logger.info(f"Ensured config directory exists: {directory}")
    
    def _validate_paths(self):
        """Validate that required input paths exist."""
        required_paths = []
        
        if self.stage == "mitochondria_pretraining":
            data_config = self.config.get('data', {})
            processed_config = data_config.get('processed', {})
            required_paths.extend([
                processed_config.get('train_dir'),
                processed_config.get('train_mask_dir'),
                processed_config.get('cache_dir')
            ])
        
        elif self.stage == "glomeruli_finetuning":
            data_config = self.config.get('data', {})
            pretrained_config = self.config.get('pretrained_model', {})
            required_paths.extend([
                pretrained_config.get('path'),
                data_config.get('raw_images'),
                data_config.get('annotations', {}).get('json_file'),
            ])
        
        for path in required_paths:
            if path and not os.path.exists(path):
                self.logger.error(f"Required path does not exist: {path}")
                raise FileNotFoundError(f"Required path does not exist: {path}")
    
    def _convert_images(self):
        """Convert TIF images to JPG format."""
        self.logger.info("Converting TIF images to JPG...")
        
        data_config = self.config.get('data', {})
        
        if self.stage == "mitochondria_pretraining":
            # Skip conversion for mitochondria - patches already exist
            self.logger.info("Skipping image conversion - patches already exist")
                
        elif self.stage == "glomeruli_finetuning":
            # Convert preeclampsia TIFs
            raw_images = data_config.get('raw_images')
            train_dir = data_config.get('processed', {}).get('train_dir')
            if raw_images and train_dir:
                convert_tif_to_jpg(raw_images, train_dir)
        
        self.logger.info("Image conversion completed")
    
    def _generate_patches(self):
        """Generate image patches for U-Net training."""
        if self.stage != "mitochondria_pretraining":
            return  # Only needed for mitochondria pretraining
            
        # Skip patch generation - patches already exist
        self.logger.info("Skipping patch generation - patches already exist")
        
        # Verify patches exist
        patch_config = self.config.get('data', {}).get('patches', {})
        output_dir = patch_config.get('output_dir')
        mask_output_dir = patch_config.get('mask_output_dir')
        
        if output_dir and mask_output_dir:
            image_count = len(list(Path(output_dir).glob("*.jpg")))
            mask_count = len(list(Path(mask_output_dir).glob("*.jpg")))
            self.logger.info(f"Found {image_count} image patches and {mask_count} mask patches")
        
        self.logger.info("Patch verification completed")
    
    def _organize_glomeruli_data(self):
        """Organize glomeruli data into clean structure and generate patches."""
        if self.stage != "glomeruli_finetuning":
            return  # Only needed for glomeruli fine-tuning
            
        # Check if derived_data already exists and has content
        derived_base = Path("derived_data/glomeruli_data")
        training_image_patches = derived_base / "training" / "image_patches"
        testing_image_patches = derived_base / "testing" / "image_patches"
        prediction_image_patches = derived_base / "prediction" / "image_patches"
        
        # Check if all directories exist and have files
        if (training_image_patches.exists() and any(training_image_patches.iterdir()) and
            testing_image_patches.exists() and any(testing_image_patches.iterdir()) and
            prediction_image_patches.exists() and any(prediction_image_patches.iterdir())):
            
            # Check if FORCE_RERUN is enabled
            if os.getenv('FORCE_RERUN', 'false').lower() == 'true':
                self.logger.info("🔄 FORCE_RERUN: Overriding skip logic - regenerating all derived data")
                # Remove existing derived data to force regeneration
                import shutil
                if derived_base.exists():
                    shutil.rmtree(derived_base)
                    self.logger.info("Removed existing derived_data directory")
            else:
                self.logger.info("Derived data already exists and has content - skipping patchification")
                self.logger.info(f"Training images: {training_image_patches}")
                self.logger.info(f"Testing images: {testing_image_patches}")
                self.logger.info(f"Prediction images: {prediction_image_patches}")
                return
            
        self.logger.info("Organizing glomeruli data and generating patches...")
        
        # Also support paired patchification (mito-style)
        from eq.utils.paths import ensure_directory
        
        data_config = self.config.get('data', {})
        raw_images_dir = data_config.get('raw_images')
        processed_config = data_config.get('processed', {})
        
        # Define clean derived_data structure (mito-style: separate image_patches and mask_patches directories)
        derived_base = Path("derived_data/glomeruli_data")
        training_base = derived_base / "training"
        testing_base = derived_base / "testing"
        cache_dir = derived_base / "cache"
        
        # Ensure clean directory structure (mito-style)
        ensure_directory(training_base)
        ensure_directory(testing_base)
        ensure_directory(cache_dir)
        
        # Create mito-style patch directories
        training_image_patches = training_base / "image_patches"
        training_mask_patches = training_base / "mask_patches"
        testing_image_patches = testing_base / "image_patches"
        testing_mask_patches = testing_base / "mask_patches"
        
        # Create prediction directory for subjects without masks
        prediction_base = derived_base / "prediction"
        prediction_image_patches = prediction_base / "image_patches"
        prediction_mask_patches = prediction_base / "mask_patches"
        
        ensure_directory(training_image_patches)
        ensure_directory(training_mask_patches)
        ensure_directory(testing_image_patches)
        ensure_directory(testing_mask_patches)
        ensure_directory(prediction_image_patches)
        ensure_directory(prediction_mask_patches)
        
        # Copy annotations to cache
        annotations_file = data_config.get('annotations', {}).get('json_file')
        if annotations_file and Path(annotations_file).exists():
            import shutil
            shutil.copy2(annotations_file, cache_dir / "annotations.json")
            self.logger.info(f"Copied annotations to {cache_dir / 'annotations.json'}")
        
        # Use raw images/masks roots from config (do not require train/test under raw)
        raw_images_dir = Path(raw_images_dir)
        images_dir = raw_images_dir / "images"
        masks_dir = raw_images_dir / "masks"
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        subject_folders = [d for d in images_dir.iterdir() if d.is_dir() and d.name.startswith('T')]
        
        if subject_folders:
            self.logger.info(f"Found {len(subject_folders)} subject folders: {[f.name for f in subject_folders]}")
            
            # Separate subjects with and without masks
            subjects_with_masks = []
            subjects_without_masks = []
            for subject_folder in subject_folders:
                subject_name = subject_folder.name
                subject_masks_dir = masks_dir / subject_name
                if subject_masks_dir.exists() and any(subject_masks_dir.iterdir()):
                    subjects_with_masks.append(subject_folder)
                else:
                    subjects_without_masks.append(subject_folder)
                    self.logger.info(f"No masks for subject {subject_name}; will be used for prediction")
            
            self.logger.info(f"Found {len(subjects_with_masks)} subjects with masks: {[f.name for f in subjects_with_masks]}")
            self.logger.info(f"Found {len(subjects_without_masks)} subjects without masks: {[f.name for f in subjects_without_masks]}")
            
            if len(subjects_with_masks) == 0:
                raise ValueError("No subjects with masks found - cannot proceed with training")
            
            # Split subjects with masks 80/20 into training/testing
            subjects_with_masks_sorted = sorted(subjects_with_masks, key=lambda p: p.name)
            split_idx = int(len(subjects_with_masks_sorted) * 0.8)
            train_subjects = subjects_with_masks_sorted[:split_idx]
            test_subjects = subjects_with_masks_sorted[split_idx:]
            
            self.logger.info(f"Training subjects: {[f.name for f in train_subjects]}")
            self.logger.info(f"Testing subjects: {[f.name for f in test_subjects]}")
            self.logger.info(f"Prediction subjects: {[f.name for f in subjects_without_masks]}")

            def process_subjects_with_masks(subject_list, image_patches_dir: Path, mask_patches_dir: Path, split_name: str):
                for subject_folder in subject_list:
                    subject_name = subject_folder.name
                    self.logger.info(f"Processing subject {subject_name} for {split_name}...")
                    # Masks for this subject (already verified to exist)
                    subject_masks_dir = masks_dir / subject_name
                    try:
                        # Create temporary directory for this subject's patches
                        import tempfile
                        with tempfile.TemporaryDirectory() as temp_dir:
                            temp_dir_path = Path(temp_dir)
                            
                            # Use smart patchification to only create patches with glomeruli content
                            # TODO: Implement smart_patchify_image_and_mask_dirs in eq.processing.image_mask_preprocessing
                            # For now, use the standard patchify function
                            from eq.processing.image_mask_preprocessing import patchify_image_and_mask_dirs

                            # Temporary smart patchify implementation
                            def smart_patchify_image_and_mask_dirs(square_size, image_dir, mask_dir, output_dir, overlap=0, min_foreground_ratio=0):
                                """Temporary implementation - uses standard patchify for now."""
                                # TODO: Add smart logic to skip patches without glomeruli content
                                patchify_image_and_mask_dirs(square_size, image_dir, mask_dir, output_dir)
                                exit()  # Fake return values (created, skipped) - TODO: calculate actual values
                            created, skipped = smart_patchify_image_and_mask_dirs(
                                square_size=224,
                                image_dir=str(subject_folder),
                                mask_dir=str(subject_masks_dir),
                                output_dir=str(temp_dir_path),
                                overlap=0.5,  # 50% overlap to capture more glomeruli
                                min_foreground_ratio=0.01  # Only patches with at least 1% glomeruli
                            )
                            
                            self.logger.info(f"  {subject_name}: Created {created} patches, skipped {skipped} empty patches")
                            
                            # Move image patches to image_patches directory
                            for img_file in temp_dir_path.rglob("*.jpg"):
                                if not img_file.name.endswith('_mask.jpg'):
                                    import shutil
                                    shutil.move(str(img_file), str(image_patches_dir / img_file.name))
                            
                            # Move mask patches to mask_patches directory
                            for mask_file in temp_dir_path.rglob("*_mask.jpg"):
                                import shutil
                                shutil.move(str(mask_file), str(mask_patches_dir / mask_file.name))
                        
                        self.logger.info(f"Generated patches for {subject_name}")
                    except Exception as e:
                        self.logger.error(f"Failed to patchify subject {subject_name}: {e}")

            def process_subjects_without_masks(subject_list, image_patches_dir: Path, split_name: str):
                for subject_folder in subject_list:
                    subject_name = subject_folder.name
                    self.logger.info(f"Processing subject {subject_name} for {split_name}...")
                    try:
                        # Create temporary directory for this subject's patches
                        import tempfile
                        with tempfile.TemporaryDirectory() as temp_dir:
                            temp_dir_path = Path(temp_dir)
                            
                            # Patchify images only (no masks)
                            from eq.processing.image_mask_preprocessing import patchify_image_dir
                            patchify_image_dir(
                                square_size=224,
                                input_dir=str(subject_folder),
                                output_dir=str(temp_dir_path)
                            )
                            
                            # Move image patches to image_patches directory
                            for img_file in temp_dir_path.rglob("*.jpg"):
                                import shutil
                                shutil.move(str(img_file), str(image_patches_dir / img_file.name))
                        
                        self.logger.info(f"Generated patches for {subject_name}")
                    except Exception as e:
                        self.logger.error(f"Failed to patchify subject {subject_name}: {e}")

            process_subjects_with_masks(train_subjects, training_image_patches, training_mask_patches, 'training')
            process_subjects_with_masks(test_subjects, testing_image_patches, testing_mask_patches, 'testing')
            process_subjects_without_masks(subjects_without_masks, prediction_image_patches, 'prediction')
        
        else:
            raise ValueError(f"No subject folders found in {images_dir}")
        
        # Update config to point to the mito-style derived_data structure
        processed_config['train_dir'] = str(training_image_patches)
        processed_config['train_mask_dir'] = str(training_mask_patches)
        processed_config['test_dir'] = str(testing_image_patches)
        processed_config['cache_dir'] = str(cache_dir)
        
        self.logger.info("Glomeruli data organization completed")
        self.logger.info(f"Training images: {training_image_patches}")
        self.logger.info(f"Training masks: {training_mask_patches}")
        self.logger.info(f"Testing images: {testing_image_patches}")
        self.logger.info(f"Testing masks: {testing_mask_patches}")
        self.logger.info(f"Prediction images: {prediction_image_patches}")
        self.logger.info(f"Cache: {cache_dir}")
    
    def _needs_raw_data_organization(self) -> bool:
        """Raw data should NOT be organized - train/test splits only happen in derived_data."""
        return False
    
    def _organize_raw_data_automatically(self):
        """Organize raw data into clean structure WITHOUT train/test splits."""
        try:
            self.logger.info("Running automatic raw data organization...")
            
            # Get paths from config
            data_config = self.config.get('data', {})
            raw_images_dir = data_config.get('raw_images')
            if not raw_images_dir:
                raise ValueError("raw_images path not specified in config")
            
            data_dir = Path(raw_images_dir)
            
            # Check if we already have the clean structure
            images_dir = data_dir / "images"
            masks_dir = data_dir / "masks"
            
            if images_dir.exists() and masks_dir.exists():
                self.logger.info("Clean structure already exists - no organization needed")
                return
            
            # Create clean structure
            images_dir.mkdir(parents=True, exist_ok=True)
            masks_dir.mkdir(parents=True, exist_ok=True)
            
            # Find and move subject directories to clean structure
            subject_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('T')]
            
            for subject_dir in subject_dirs:
                # Check if this is an image or mask directory
                has_images = any(subject_dir.rglob("*.jpg")) or any(subject_dir.rglob("*.tif"))
                has_masks = any(subject_dir.rglob("*_mask.jpg")) or any(subject_dir.rglob("*_mask.tif"))
                
                if has_images and not has_masks:
                    # This is an image directory
                    target_dir = images_dir / subject_dir.name
                    if not target_dir.exists():
                        import shutil
                        shutil.move(str(subject_dir), str(target_dir))
                        self.logger.info(f"Moved {subject_dir.name} to images/")
                elif has_masks:
                    # This is a mask directory
                    target_dir = masks_dir / subject_dir.name
                    if not target_dir.exists():
                        import shutil
                        shutil.move(str(subject_dir), str(target_dir))
                        self.logger.info(f"Moved {subject_dir.name} to masks/")
            
            self.logger.info("Raw data organization completed - clean structure created")
            self.logger.info("Train/test splits will be created in derived_data during processing")
                
        except Exception as e:
            self.logger.error(f"Failed to organize raw data automatically: {e}")
            self.logger.error("Please organize your data manually according to the README structure")
            raise
    
    def _process_annotations(self):
        """Process Label Studio annotations."""
        if self.stage != "glomeruli_finetuning":
            return  # Only needed for glomeruli fine-tuning
            
        self.logger.info("Processing Label Studio annotations...")
        
        annotations_config = self.config.get('data', {}).get('annotations', {})
        json_file = annotations_config.get('json_file')
        cache_dir = self.config.get('data', {}).get('processed', {}).get('cache_dir')
        
        if json_file and cache_dir:
            # Load annotations
            annotations = load_annotations_from_json(json_file)
            
            # Extract scores
            scores = get_scores_from_annotations(annotations)
            
            self.logger.info(f"Processed {len(annotations)} annotations")
            self.logger.info(f"Extracted {len(scores)} endotheliosis scores")
        
        self.logger.info("Annotation processing completed")
    
    def _load_mitochondria_data(self):
        """Load mitochondria patch data for training."""
        if self.stage != "mitochondria_pretraining":
            return
            
        try:
            # Always load the real data - QUICK_TEST only affects training parameters
            self.logger.info("Loading mitochondria dataset")
            processed = self.config.get('data', {}).get('processed', {})
            image_patches_dir = processed.get('train_dir')
            mask_patches_dir = processed.get('train_mask_dir')
            cache_dir = processed.get('cache_dir')
            train_ratio = processed.get('train_ratio', 0.8)
            val_ratio = processed.get('val_ratio', 0.2)
            test_ratio = processed.get('test_ratio', 0.0)

            if not image_patches_dir or not mask_patches_dir or not cache_dir:
                raise ValueError("Processed train_dir, train_mask_dir, and cache_dir must be set in config for mitochondria pretraining")

            data = load_mitochondria_patches(
                image_patches_dir=image_patches_dir,
                cache_dir=cache_dir,
                mask_patches_dir=mask_patches_dir,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                random_seed=processed.get('random_seed', 42)
            )
            
            self.logger.info("Loaded data:")
            self.logger.info(f"  Train: {data['train']['images'].shape}")
            self.logger.info(f"  Val: {data['val']['images'].shape}")
            self.logger.info(f"  Patch size: {data['patch_size']}")
            self.logger.info(f"  Num classes: {data['num_classes']}")
            
            # Store data for training
            self.mitochondria_data = data
            
        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
            raise
    
    def _train_model(self):
        """Train the segmentation model."""
        self.logger.info(f"Training {self.stage} model...")
        
        model_config = self.config.get('model', {})
        training_config = model_config.get('training', {})
        
        if self.stage == "mitochondria_pretraining":
            # For mitochondria pretraining, use the clean training function
            if not hasattr(self, 'mitochondria_data'):
                raise ValueError("Mitochondria data not loaded. Call _load_mitochondria_data first.")
            
            self.logger.info("Starting mitochondria training with clean implementation...")
            
            # Get the loaded data
            train_images = self.mitochondria_data['train']['images']
            train_masks = self.mitochondria_data['train']['masks']
            val_images = self.mitochondria_data['val']['images']
            val_masks = self.mitochondria_data['val']['masks']
            
            self.logger.info(f"Training with {train_images.shape[0]} samples")
            self.logger.info(f"Validating with {val_images.shape[0]} samples")
            
            # Check if we should use pretrained model for evaluation
            # Default to True for QUICK_TEST mode, False for production
            is_quick_test = os.getenv('QUICK_TEST', 'false').lower() == 'true'
            default_pretrained = 'true' if is_quick_test else 'false'
            use_pretrained = os.getenv('USE_PRETRAINED', default_pretrained).lower() == 'true'
            
            if use_pretrained:
                # Load pretrained model for evaluation
                self.logger.info("🚀 Loading pretrained mitochondria model for evaluation...")
                pretrained_model_path = "backups/mito_dynamic_unet_seg_model-e50_b16.pkl"
                
                if not os.path.exists(pretrained_model_path):
                    raise FileNotFoundError(f"Pretrained model not found: {pretrained_model_path}")
                
                # Import and load the pretrained model
                from fastai.vision.all import load_learner

                # Load the pretrained model
                learn = load_learner(pretrained_model_path)
                self.logger.info(f"✅ Loaded pretrained model: {pretrained_model_path}")
                
                # Get output directory for evaluation results
                output_dir = model_config.get('output_dir', 'models/segmentation/mitochondria')
                model_name = 'mito_pretrained_evaluation'
                
                # Run evaluation on validation data
                self.logger.info("🔍 Running evaluation on validation data...")
                self._evaluate_mitochondria_model(learn, val_images, val_masks, output_dir, model_name)
                
            else:
                # Use the clean mitochondria training function
                from eq.models.train_mitochondria_fastai import train_mitochondria_model

                # Get output directory and model name from config
                checkpoint_path = model_config.get('checkpoint_path', '')
                output_dir = os.path.dirname(checkpoint_path) if checkpoint_path else model_config.get('output_dir', 'models/segmentation/mitochondria')
                model_name = os.path.basename(checkpoint_path).replace('.pkl', '') if checkpoint_path else model_config.get('model_name', 'mito_model')
                
                # Check if we're in quick test mode and adjust training parameters
                is_quick_test = os.getenv('QUICK_TEST', 'false').lower() == 'true'
                
                if is_quick_test:
                    # Use smaller parameters for quick testing
                    test_batch_size = min(training_config.get('batch_size', 16), 4)
                    test_epochs = 5  # Explicitly set to 5 for quick testing
                    self.logger.info(f"🔬 QUICK_TEST mode: Using batch_size={test_batch_size}, epochs={test_epochs}")
                else:
                    # Use production parameters
                    test_batch_size = training_config.get('batch_size', 16)
                    test_epochs = training_config.get('epochs', 50)
                    self.logger.info(f"🚀 Production mode: Using batch_size={test_batch_size}, epochs={test_epochs}")
                
                # Train the model
                learn = train_mitochondria_model(
                    train_images=train_images,
                    train_masks=train_masks,
                    val_images=val_images,
                    val_masks=val_masks,
                    output_dir=output_dir,
                    model_name=model_name,
                    batch_size=test_batch_size,
                    epochs=test_epochs,
                    learning_rate=training_config.get('learning_rate', 1e-3),
                    image_size=model_config.get('input_size', [224, 224])[0]
                )
                
                self.logger.info("Mitochondria training completed successfully!")
            
        else:
            # For glomeruli fine-tuning, use the new transfer learning function
            self.logger.info("Starting glomeruli transfer learning...")
            
            # Check if model already exists
            model_config = self.config.get('model', {})
            checkpoint_path = model_config.get('checkpoint_path', '')
            if checkpoint_path and Path(checkpoint_path).exists():
                self.logger.info(f"✅ Model checkpoint already exists: {checkpoint_path}")
                self.logger.info("Model will be loaded instead of retrained")
            
            # Use the new training module
            from eq.training import train_glomeruli
            # TODO: Update to use proper configuration-based training
            # For now, use the basic training function
            segmenter = train_glomeruli(
                data_dir=self.config.get('data', {}).get('processed', {}).get('cache_dir', ''),
                model_dir=self.config.get('model', {}).get('output_dir', 'models/segmentation/glomeruli_xfer_learn'),
                base_model=self.config.get('model', {}).get('base_model', ''),
                epochs=self.config.get('training', {}).get('epochs', 50)
            )
        
        self.logger.info("Model training completed")

        # If we just finished glomeruli fine-tuning, run evaluation mirroring mitochondria pattern
        if self.stage == "glomeruli_finetuning":
            try:
                # Prepare validation data from cache as done in fastai_segmenter
                from eq.utils.common import load_pickled_data
                cache_dir = self.config.get('data', {}).get('processed', {}).get('cache_dir')
                if cache_dir:
                    val_images = load_pickled_data(Path(cache_dir) / 'val_images.pickle')
                    val_masks = load_pickled_data(Path(cache_dir) / 'val_masks.pickle')
                else:
                    raise ValueError("Cache directory not specified in config for validation data")

                # Determine output directory and model name
                model_config = self.config.get('model', {})
                checkpoint_path = model_config.get('checkpoint_path', '')
                output_dir = os.path.dirname(checkpoint_path) if checkpoint_path else model_config.get('output_dir', 'models/segmentation/glomeruli_xfer_learn')
                model_name = os.path.basename(checkpoint_path).replace('.pkl', '') if checkpoint_path else model_config.get('model_name', 'glom_model')

                # load learner if not available from returned segmenter
                learn = None
                try:
                    if segmenter is not None and hasattr(segmenter, 'learn') and segmenter.learn is not None:
                        learn = segmenter.learn
                except Exception:
                    learn = None

                if learn is None:
                    # Try to load from checkpoint path if exists
                    if checkpoint_path and os.path.exists(checkpoint_path):
                        try:
                            # Try safer loading method first
                            from fastai.learner import Learner
                            learn = Learner.load(checkpoint_path, with_opt=False)
                            self.logger.info("✅ Loaded model using safe method")
                        except Exception as e:
                            self.logger.warning(f"Safe loading failed ({e}), falling back to load_learner")
                            from fastai.vision.all import load_learner
                            learn = load_learner(checkpoint_path)
                    else:
                        self.logger.warning("Learner not available for evaluation; skipping glomeruli evaluation step.")
                        return

                self.logger.info("🔍 Running glomeruli evaluation on validation data...")
                _ = evaluate_glomeruli_model(learn, val_images, val_masks, output_dir, f"{model_name}_evaluation")
                self.logger.info("Glomeruli evaluation completed successfully")
            except Exception as e:
                self.logger.warning(f"Glomeruli evaluation step failed: {e}")
    
    def _evaluate_mitochondria_model(self, learn, val_images, val_masks, output_dir, model_name):
        """Evaluate a pretrained mitochondria model on validation data."""
        import os
        from datetime import datetime
        from pathlib import Path

        import matplotlib.pyplot as plt
        import numpy as np

        # Create output directory
        output_path = Path(output_dir) / model_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"🔍 Evaluating mitochondria model on {len(val_images)} validation samples...")
        
        # Calculate quantitative metrics on all validation samples
        from PIL import Image
        
        dice_scores = []
        iou_scores = []
        pixel_accuracies = []
        
        print("Computing quantitative metrics...")
        self.logger.info("Computing quantitative metrics on all validation samples...")
        
        for i in range(len(val_images)):
            img = val_images[i]
            true_mask = val_masks[i]
            
            # Convert to 3-channel if needed
            if img.shape[-1] == 1:
                img = np.repeat(img, 3, axis=-1)
            # Convert to PIL Image
            img_pil = Image.fromarray((img * 255).astype(np.uint8))
            # Make prediction
            pred_result = learn.predict(img_pil)
            
            # Extract prediction mask
            if isinstance(pred_result, tuple) and len(pred_result) >= 2:
                pred_tensor = pred_result[1]
            else:
                pred_tensor = pred_result
                
            # Convert to numpy
            if hasattr(pred_tensor, 'numpy'):
                pred_mask = pred_tensor.numpy()
            elif hasattr(pred_tensor, 'cpu'):
                pred_mask = pred_tensor.cpu().numpy()
            else:
                pred_mask = pred_tensor
            
            # Ensure binary masks for metric calculation
            true_binary = (true_mask.squeeze() > 0.5).astype(np.float32)
            
            # Simple fix: resize prediction to match ground truth
            if pred_mask.shape != true_binary.shape:
                from scipy.ndimage import zoom
                scale_factors = [true_binary.shape[i] / pred_mask.shape[i] for i in range(len(pred_mask.shape))]
                pred_mask = zoom(pred_mask, scale_factors, order=1)
            
            pred_binary = (pred_mask > 0.5).astype(np.float32)
            
            # Calculate Dice score
            intersection = np.sum(true_binary * pred_binary)
            dice = (2.0 * intersection) / (np.sum(true_binary) + np.sum(pred_binary) + 1e-7)
            dice_scores.append(dice)
            
            # Calculate IoU (Jaccard index)
            union = np.sum(true_binary) + np.sum(pred_binary) - intersection
            iou = intersection / (union + 1e-7)
            iou_scores.append(iou)
            
            # Calculate pixel accuracy
            correct_pixels = np.sum(true_binary == pred_binary)
            total_pixels = true_binary.size
            pixel_acc = correct_pixels / total_pixels
            pixel_accuracies.append(pixel_acc)
        
        # Calculate summary statistics
        metrics = {
            'dice_mean': np.mean(dice_scores),
            'dice_std': np.std(dice_scores),
            'iou_mean': np.mean(iou_scores),
            'iou_std': np.std(iou_scores),
            'pixel_acc_mean': np.mean(pixel_accuracies),
            'pixel_acc_std': np.std(pixel_accuracies),
            'num_samples': len(val_images)
        }
        
        # Log metrics
        print("EVALUATION METRICS:")
        print(f"   Dice Score:      {metrics['dice_mean']:.4f} ± {metrics['dice_std']:.4f}")
        print(f"   IoU Score:       {metrics['iou_mean']:.4f} ± {metrics['iou_std']:.4f}")
        print(f"   Pixel Accuracy:  {metrics['pixel_acc_mean']:.4f} ± {metrics['pixel_acc_std']:.4f}")
        print(f"   Samples:         {metrics['num_samples']}")
        
        self.logger.info(f"Dice Score: {metrics['dice_mean']:.4f} ± {metrics['dice_std']:.4f}")
        self.logger.info(f"IoU Score: {metrics['iou_mean']:.4f} ± {metrics['iou_std']:.4f}")
        self.logger.info(f"Pixel Accuracy: {metrics['pixel_acc_mean']:.4f} ± {metrics['pixel_acc_std']:.4f}")
        
        # Generate sample predictions
        predictions_plot = output_path / "sample_predictions.png"
        try:
            # Use our validation data directly (first 4 samples)
            sample_images = val_images[:4]
            sample_masks = val_masks[:4]
            
            # Convert numpy arrays to PIL Images for prediction
            from PIL import Image
            
            preds = []
            for i in range(len(sample_images)):
                img = sample_images[i]
                # Convert to 3-channel if needed
                if img.shape[-1] == 1:
                    img = np.repeat(img, 3, axis=-1)
                # Convert to PIL Image
                img_pil = Image.fromarray((img * 255).astype(np.uint8))
                # Make prediction
                pred = learn.predict(img_pil)
                preds.append(pred)
            
            # Create a grid of images, masks, and predictions
            fig, axes = plt.subplots(4, 3, figsize=(12, 16))
            
            # Add testing indicator to plot title if in QUICK_TEST mode
            is_quick_test = os.getenv('QUICK_TEST', 'false').lower() == 'true'
            if is_quick_test:
                fig.suptitle('TESTING RUN - Pretrained Model Evaluation: Image | Ground Truth | Prediction', fontsize=16)
            else:
                fig.suptitle('Pretrained Model Evaluation: Image | Ground Truth | Prediction', fontsize=16)
            
            for i in range(min(4, len(sample_images))):
                # Original image 
                img = sample_images[i]
                if img.shape[-1] == 1:
                    img_display = img.squeeze()
                else:
                    img_display = img
                
                axes[i, 0].imshow(img_display, cmap='gray')
                axes[i, 0].set_title(f'Image {i+1}')
                axes[i, 0].axis('off')
                
                # Ground truth mask
                mask = sample_masks[i]
                if mask.shape[-1] == 1:
                    mask_display = mask.squeeze()
                else:
                    mask_display = mask
                    
                axes[i, 1].imshow(mask_display, cmap='gray')
                axes[i, 1].set_title(f'Ground Truth {i+1}')
                axes[i, 1].axis('off')
                
                # Prediction - extract the mask from FastAI prediction tuple
                pred = preds[i]
                if isinstance(pred, tuple) and len(pred) >= 2:
                    # FastAI returns (category, tensor, probabilities)
                    pred_tensor = pred[1]
                else:
                    pred_tensor = pred
                
                # Convert tensor to numpy for display
                if hasattr(pred_tensor, 'numpy'):
                    pred_mask = pred_tensor.numpy()
                elif hasattr(pred_tensor, 'cpu'):
                    pred_mask = pred_tensor.cpu().numpy()
                else:
                    pred_mask = pred_tensor
                    
                axes[i, 2].imshow(pred_mask, cmap='gray')
                axes[i, 2].set_title(f'Prediction {i+1}')
                axes[i, 2].axis('off')
            
            plt.tight_layout()
            plt.savefig(predictions_plot, dpi=300, bbox_inches='tight')
            plt.close()
            self.logger.info(f"Sample predictions plot saved to {predictions_plot}")
            
        except Exception as e:
            self.logger.warning(f"Could not save sample predictions plot: {e}")
            import traceback
            self.logger.warning(f"Full error traceback: {traceback.format_exc()}")
        
        # Save evaluation summary
        evaluation_summary = output_path / "evaluation_summary.txt"
        try:
            with open(evaluation_summary, 'w') as f:
                f.write("Mitochondria Segmentation Model Evaluation Summary\n")
                f.write("==================================================\n\n")
                
                # Add testing indicator if in QUICK_TEST mode
                is_quick_test = os.getenv('QUICK_TEST', 'false').lower() == 'true'
                if is_quick_test:
                    f.write("TESTING RUN - QUICK_TEST MODE\n")
                    f.write("This is a TESTING run using a PRETRAINED model.\n")
                    f.write("DO NOT use this evaluation for production inference!\n\n")
                
                f.write(f"Model: {model_name}\n")
                f.write("Model source: backups/mito_dynamic_unet_seg_model-e50_b16.pkl\n")
                f.write(f"Evaluation samples: {len(val_images)}\n")
                f.write(f"Output directory: {output_path}\n")
                f.write(f"Evaluation completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Add quantitative metrics
                f.write("QUANTITATIVE EVALUATION METRICS:\n")
                f.write("================================\n")
                f.write(f"Dice Score:      {metrics['dice_mean']:.4f} ± {metrics['dice_std']:.4f}\n")
                f.write(f"IoU Score:       {metrics['iou_mean']:.4f} ± {metrics['iou_std']:.4f}\n")
                f.write(f"Pixel Accuracy:  {metrics['pixel_acc_mean']:.4f} ± {metrics['pixel_acc_std']:.4f}\n")
                f.write(f"Sample Count:    {metrics['num_samples']}\n\n")
                
                f.write("METRIC EXPLANATIONS:\n")
                f.write("===================\n")
                f.write("- Dice Score: Overlap between prediction and ground truth (0=no overlap, 1=perfect)\n")
                f.write("- IoU Score: Intersection over Union, also called Jaccard index (0=no overlap, 1=perfect)\n")
                f.write("- Pixel Accuracy: Fraction of correctly classified pixels (0=all wrong, 1=all correct)\n\n")
                
                f.write("Files generated:\n")
                f.write(f"- Sample predictions: {predictions_plot}\n")
                f.write(f"- Evaluation summary: {evaluation_summary}\n")
            
            self.logger.info(f"Evaluation summary saved to {evaluation_summary}")
            
        except Exception as e:
            self.logger.warning(f"Could not save evaluation summary: {e}")
        
        self.logger.info("Mitochondria model evaluation completed successfully!")
        print("Mitochondria model evaluation completed successfully!")
    
    def run(self):
        """Execute the complete pipeline."""
        # Check if we're in quick test mode and display banner
        is_quick_test = os.getenv('QUICK_TEST', 'false').lower() == 'true'
        
        if is_quick_test:
            print("\n" + "="*80)
            print("QUICK_TEST MODE ENABLED")
            print("="*80)
            print("Training with LIMITED DATA (20 samples)")
            print("Reduced epochs (5 instead of 50)")
            print("Smaller batch size (4 instead of 16)")
            print("Same 224x224 patches, just faster training")
            print("="*80 + "\n")
            self.logger.info("QUICK_TEST MODE ENABLED - Limited training for faster testing")
        else:
            print("\n" + "="*80)
            print("PRODUCTION MODE")
            print("="*80)
            print("Training with FULL DATASET")
            print("Full epochs (50)")
            print("Full batch size (16)")
            print("Production 224x224 training parameters")
            print("="*80 + "\n")
            self.logger.info("PRODUCTION MODE - Full training")
        
        self.logger.info(f"Starting {self.stage} pipeline...")
        
        try:
            # Setup
            self._ensure_directories()
            self._validate_paths()
            
            # Data processing
            self._convert_images()
            self._generate_patches()
            self._process_annotations()
            
            # Load mitochondria data if needed
            if self.stage == "mitochondria_pretraining":
                self._load_mitochondria_data()
            
            # Organize glomeruli data if needed
            if self.stage == "glomeruli_finetuning":
                self._organize_glomeruli_data()
            
            # Model training
            self._train_model()
            
            self.logger.info(f"{self.stage} pipeline completed successfully!")
            
            # Log next steps
            next_step = self.config.get('next_step')
            if next_step:
                self.logger.info(f"Next step: {next_step.get('description')}")
                self.logger.info(f"Config file: {next_step.get('config_file')}")
        
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Segmentation Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m eq.pipeline.segmentation_pipeline --stage mito
  python -m eq.pipeline.segmentation_pipeline --stage glomeruli
  python -m eq.pipeline.segmentation_pipeline --stage all
        """
    )
    
    parser.add_argument(
        '--stage',
        choices=['mito', 'glomeruli', 'all'],
        required=True,
        help='Pipeline stage to run'
    )
    
    parser.add_argument(
        '--config-dir',
        default='configs',
        help='Directory containing configuration files (default: configs)'
    )
    
    args = parser.parse_args()
    
    # Determine config files
    config_files = []
    if args.stage in ['mito', 'all']:
        config_files.append(os.path.join(args.config_dir, 'mito_pretraining_config.yaml'))
    if args.stage in ['glomeruli', 'all']:
        config_files.append(os.path.join(args.config_dir, 'glomeruli_finetuning_config.yaml'))
    
    # Run pipeline stages
    for config_file in config_files:
        if not os.path.exists(config_file):
            print(f"Error: Config file not found: {config_file}")
            sys.exit(1)
        
        print(f"\n{'='*60}")
        print(f"Running pipeline with config: {config_file}")
        print(f"{'='*60}")
        
        pipeline = SegmentationPipeline(config_file)
        pipeline.run()
    
    print(f"\n{'='*60}")
    print("All pipeline stages completed successfully!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
