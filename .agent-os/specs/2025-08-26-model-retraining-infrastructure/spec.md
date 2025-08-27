# Spec Requirements Document

> Spec: Model Retraining and Training Infrastructure Organization
> Created: 2025-08-26

## Overview

Implement FastAI v2 compatible retraining scripts for both mitochondria and glomeruli models, and reorganize the training infrastructure to separate training scripts from pipeline execution scripts. This will address the critical performance issues caused by FastAI v1 vs v2 incompatibility and establish a clean, maintainable training codebase.

ALWAYS LOAD `conda activate eq` FIRST
If new packages are needed, install with `mamba install <package>`

## User Stories

### Retrain Models for Production Performance

As a **machine learning engineer**, I want to retrain both segmentation models with FastAI v2, so that I can achieve the expected 70-90%+ detection rates needed for production use. The current models show 7.98% (mitochondria) and 0% (glomeruli) detection rates, which are unacceptable for a trained model.

**Workflow**: Create mitochondria retraining script based on historical `segment_mitochondria.py`, retrain mitochondria model first, then retrain glomeruli model using the new mitochondria model as base, validate performance meets production thresholds.

### Organize Training Infrastructure

As a **developer**, I want to properly organize training scripts separate from pipeline execution scripts, so that I can maintain clear separation of concerns and avoid confusion between training and production workflows.

**Workflow**: Rename retrain_* scripts to train_*, determine optimal organization structure, move training scripts to appropriate location, update imports and references, ensure pipeline scripts remain focused on orchestration.

## Spec Scope

1. **Mitochondria Retraining Script** - Create `train_mitochondria.py` based on historical `segment_mitochondria.py` with FastAI v2 compatibility
2. **Training Infrastructure Reorganization** - Establish clean separation of concerns with dedicated directories for training, inference, and pipeline orchestration
3. **Retraining Process Setup** - Establish proper training workflow for both models with validation and performance monitoring
4. **Model Integration Testing** - Verify new models integrate properly with existing inference pipeline

## Current Infrastructure Problems Identified

The `src/eq/pipeline/` directory contains mixed concerns that need reorganization:

**Training Scripts (should be in `training/`):**
- `retrain_glomeruli_original.py` - Model training logic

**Inference Scripts (should be in `inference/`):**
- `run_glomeruli_prediction.py` - Model prediction/inference
- `run_glomeruli_prediction_fixed.py` - Duplicate fixed version (should be removed)
- `gpu_inference.py` - GPU inference utilities
- `historical_glomeruli_inference.py` - Legacy inference (should be archived)

**Pipeline Orchestration (should stay in `pipeline/`):**
- `segmentation_pipeline.py` - Main pipeline orchestration
- `run_production_pipeline.py` - Production pipeline coordination

**Utility Functions (should be in `utils/`):**
- `extract_model_weights.py` - Model weight extraction utility

**Demo Scripts (should be removed or consolidated):**
- `run_complete_mito_pipeline.py` - Demo script (unnecessary complexity)

## Complete Reorganization Plan

### **Current Directory Structure (Before Reorganization):**

**EXISTING DIRECTORY STRUCTURE IN `src/eq/` RIGHT NOW:**

```
src/eq/
├── __main__.py         # ✅ MAIN CLI ENTRY POINT (30KB, 734 lines) - KEEP AS IS
├── __init__.py         # ✅ PACKAGE INITIALIZATION (2.7KB, 85 lines) - KEEP AS IS
├── core/               # ❌ MIXED CONCERNS: Contains functions that should be in other directories
│   ├── __init__.py     # Package initialization
│   ├── data_loading.py # ❌ MOVE: To data_management/ (data loading functions)
│   ├── preprocessing.py # ❌ MOVE: To processing/ (preprocessing functions)
│   ├── constants.py    # ✅ KEEP: Core constants and configurations (BINARY_P2C=[0,1], mask threshold 127)
│   ├── model_loading.py # ❌ MOVE: To data_management/ (model loading functions)
│   └── __pycache__/   # Python cache directory
├── data/               # ✅ EXISTING: Basic data management (needs consolidation)
│   ├── __init__.py     # Package initialization
│   ├── preprocessing.py # Data preprocessing functions (4.7KB, 158 lines) - REDUNDANT with core/
│   ├── config.py       # Data configuration (1.1KB, 55 lines)
│   ├── loaders.py      # Data loading utilities (14KB, 356 lines)
│   └── __pycache__/   # Python cache directory
├── processing/         # ✅ EXISTING: Basic image processing (needs consolidation)
│   ├── __init__.py     # Package initialization
│   ├── convert_files.py # TIF to JPG conversion utility (1.8KB, 51 lines)
│   ├── patchify.py     # Image patchification (3.8KB, 86 lines) - NEEDS BETTER NAME
│   └── __pycache__/   # Python cache directory
├── models/             # ❌ MIXED CONCERNS: Training scripts + model definitions + data loaders
│   ├── __init__.py     # Package initialization (0.0B, 0 lines) - EMPTY
│   ├── fastai_segmenter.py                    # ✅ KEEP: Model architecture definition (29KB, 801 lines)
│   ├── train_segmenter_fastai.py              # ❌ MOVE: Redundant training approach (14KB, 366 lines)
│   ├── train_mitochondria_fastai.py           # ❌ MOVE: To training/ (18KB, 487 lines)
│   ├── train_glomeruli_transfer_learning.py   # ❌ MOVE: Redundant training approach (14KB, 386 lines)
│   ├── data_loader.py                         # ❌ MOVE: To data_management/ (15KB, 463 lines)
│   └── __pycache__/   # Python cache directory
├── pipeline/           # ❌ MIXED CONCERNS: Training + inference + orchestration + utilities
│   ├── __init__.py     # Package initialization (0.0B, 0 lines) - EMPTY
│   ├── segmentation_pipeline.py               # ✅ KEEP: Main pipeline orchestration (49KB, 1053 lines)
│   ├── run_production_pipeline.py             # ✅ KEEP: Production pipeline coordination (18KB, 502 lines)
│   ├── retrain_glomeruli_original.py          # ❌ MOVE: To training/ (rename to train_glomeruli.py) (10KB, 259 lines)
│   ├── run_complete_mito_pipeline.py          # ❌ REMOVE: Demo script (unnecessary complexity) (7.4KB, 201 lines)
│   ├── run_glomeruli_prediction.py            # ❌ MOVE: To inference/ (8.1KB, 238 lines)
│   ├── run_glomeruli_prediction_fixed.py      # ❌ REMOVE: Duplicate fixed version (7.2KB, 210 lines)
│   ├── gpu_inference.py                       # ❌ MOVE: To inference/ (14KB, 387 lines)
│   ├── historical_glomeruli_inference.py      # ❌ REMOVE: Legacy inference (13KB, 361 lines)
│   ├── extract_model_weights.py               # ❌ MOVE: To utils/ (6.5KB, 207 lines)
│   └── __pycache__/   # Python cache directory
├── evaluation/         # ✅ WELL-ORGANIZED: Metrics and evaluators
│   ├── __init__.py     # Package initialization (0.0B, 0 lines) - EMPTY
│   ├── quantification_metrics.py              # Quantification evaluation metrics (8.4KB, 254 lines)
│   ├── segmentation_metrics.py                # Segmentation evaluation metrics (10KB, 316 lines)
│   ├── evaluate_glomeruli_model.py            # Glomeruli model evaluation (16KB, 431 lines)
│   ├── glomeruli_evaluator.py                 # Glomeruli evaluation utilities (7.3KB, 186 lines)
│   └── __pycache__/   # Python cache directory
├── utils/              # ❌ MIXED CONCERNS: Pure utilities + data processing + data organization
│   ├── __init__.py     # Package initialization (568B, 21 lines)
│   ├── hardware_detection.py                  # ✅ KEEP: Hardware detection utilities (13KB, 343 lines)
│   ├── config_manager.py                      # ✅ KEEP: Configuration management (7.5KB, 222 lines)
│   ├── backend_manager.py                     # ✅ KEEP: Backend management (6.5KB, 186 lines)
│   ├── mode_manager.py                        # ✅ KEEP: Mode management (12KB, 356 lines)
│   ├── logger.py                              # ✅ KEEP: Logging utilities (5.7KB, 185 lines)
│   ├── paths.py                               # ✅ KEEP: Path utilities (2.0KB, 96 lines)
│   ├── create_mitochondria_patches.py         # ❌ MOVE: To processing/ (mitochondria-specific patches) (15KB, 425 lines)
│   ├── organize_raw_data.py                   # ❌ REMOVE: Pre-eclampsia specific (9.6KB, 278 lines)
│   ├── reorganize_preeclampsia_data.py        # ❌ REMOVE: Pre-eclampsia specific (4.6KB, 137 lines)
│   ├── organize_lucchi_dataset.py             # ❌ MOVE: To data_management/ (mitochondria dataset) (8.2KB, 217 lines)
│   ├── output_manager.py                      # ❌ MOVE: To data_management/ (output management) (8.5KB, 245 lines)
│   ├── metadata_processor.py                  # ❌ MOVE: To data_management/ (glomeruli scoring) (16KB, 414 lines)
│   ├── auto_suggestion.py                     # ❌ REMOVE: Hardware suggestions (redundant) (11KB, 289 lines)
│   ├── common.py                              # ❌ REMOVE: Generic utilities (redundant) (1.8KB, 58 lines)
│   ├── env_check.py                           # ❌ REMOVE: Environment validation (redundant) (2.7KB, 86 lines)
│   ├── runtime_check.py                       # ❌ REMOVE: Runtime validation (redundant) (3.0KB, 106 lines)
│   └── __pycache__/   # Python cache directory
└── __pycache__/        # Python cache directory
```

**CURRENT STATE ANALYSIS:**

**✅ WELL-ORGANIZED (Keep as-is):**
- `evaluation/` - Well-organized metrics and evaluators
- `__main__.py` - Main CLI entry point

**❌ MIXED CONCERNS (Need reorganization):**
- `core/` - Contains functions that should be in other directoriesll 
- `models/` - Contains training scripts that should be in `training/`
- `pipeline/` - Contains training, inference, and utilities mixed with orchestration
- `utils/` - Contains data processing mixed with pure utilities
- `data/` - Has redundant preprocessing with `core/`

**📊 FILE SIZE SUMMARY:**
- **Total files to reorganize**: 25+ files across multiple directories
- **Largest files**: `segmentation_pipeline.py` (49KB), `fastai_segmenter.py` (29KB)
- **Training scripts**: 4 files totaling ~56KB
- **Inference scripts**: 3 files totaling ~35KB
- **Data processing**: 8 files totaling ~85KB
- **Utilities**: 8 files totaling ~70KB

**🔍 SPECIFIC PROBLEMS IDENTIFIED:**
1. **Training scripts scattered** across `models/` and `pipeline/`
2. **Inference scripts mixed** with pipeline orchestration
3. **Data processing scattered** across `core/`, `data/`, `processing/`, and `utils/`
4. **Redundant functionality** between `core/preprocessing.py` and `data/preprocessing.py`
5. **Core directory mixed concerns** - contains functions that should be in other directories
6. **Empty `__init__.py` files** in several directories
7. **Mixed concerns** in every directory except `evaluation/`

### **Proposed Clean Organization (After):**
```
src/eq/
├── core/               # 🧹 CLEANED: Pure constants, types, and abstract interfaces only
│   ├── constants.py    # Core constants and configurations (BINARY_P2C=[0,1], mask threshold 127)
│   ├── types.py        # Shared type definitions and interfaces
│   └── __init__.py
├── data_management/    # 🆕 CONSOLIDATED: All data management and loading functionality
│   ├── config.py                          # From data/ (existing)
│   ├── loaders.py                         # From data/ (existing)
│   ├── data_loading.py                    # From core/ (core data loading functions)
│   ├── model_loading.py                   # From core/ (core model loading functions)
│   ├── organize_lucchi_dataset.py         # From utils/ (ESSENTIAL: mitochondria dataset)
│   ├── output_manager.py                  # From utils/ (ESSENTIAL: output management)
│   ├── metadata_processor.py              # From utils/ (ESSENTIAL: glomeruli scoring and downstream analysis)
│   └── __init__.py
├── processing/         # 🆕 CONSOLIDATED: All image processing and preprocessing functionality
│   ├── preprocessing.py                    # From core/ (core preprocessing functions)
│   ├── image_mask_preprocessing.py        # From processing/ (existing patchification - renamed for clarity)
│   ├── convert_files.py                    # From processing/ (TIF to JPG conversion)
│   ├── create_mitochondria_patches.py     # From utils/ (mitochondria-specific patches)
│   └── __init__.py
├── training/           # 🆕 NEW: Dedicated training scripts (only 2 we need)
│   ├── train_mitochondria.py              # From models/train_mitochondria_fastai.py
│   ├── train_glomeruli.py                 # From pipeline/retrain_glomeruli_original.py
│   └── __init__.py
├── inference/          # 🆕 NEW: Dedicated inference/prediction scripts
│   ├── run_mitochondria_prediction.py     # 🆕 NEW: Need to create (missing)
│   ├── run_glomeruli_prediction.py        # From pipeline/run_glomeruli_prediction.py
│   ├── gpu_inference.py                   # From pipeline/gpu_inference.py
│   └── __init__.py
├── models/             # 🧹 CLEANED: Only model architecture definitions
│   ├── fastai_segmenter.py                # ✅ KEEP: Model architecture
│   └── __init__.py
├── pipeline/           # 🧹 CLEANED: Only pipeline orchestration
│   ├── segmentation_pipeline.py           # ✅ KEEP: Main pipeline orchestration
│   ├── run_production_pipeline.py         # ✅ KEEP: Production pipeline coordination
│   └── __init__.py
├── evaluation/         # 🧹 CONSOLIDATED: Metrics and evaluators (eliminate duplication)
│   ├── quantification_metrics.py              # Quantification evaluation metrics
│   ├── segmentation_metrics.py                # Pure metrics functions (reusable)
│   ├── evaluate_glomeruli_model.py            # Main glomeruli evaluator (production-ready)
│   └── __init__.py
├── utils/              # 🧹 CLEANED: Pure utility functions only
│   ├── hardware_detection.py              # ✅ KEEP: Hardware detection
│   ├── config_manager.py                  # ✅ KEEP: Configuration management
│   ├── backend_manager.py                 # ✅ KEEP: Backend management
│   ├── mode_manager.py                    # ✅ KEEP: Mode management
│   ├── logger.py                          # ✅ KEEP: Logging utilities
│   ├── paths.py                           # ✅ KEEP: Path utilities
│   ├── extract_model_weights.py           # From pipeline/ (model utility)
│   └── __init__.py
└── __main__.py         # ✅ KEEP: CLI entry point
```

### **Files to Remove (Unnecessary/Duplicate):**
- `pipeline/run_complete_mito_pipeline.py` - Demo script (unnecessary complexity)
- `pipeline/run_glomeruli_prediction_fixed.py` - Duplicate fixed version
- `pipeline/historical_glomeruli_inference.py` - Legacy inference (archived)
- `models/train_segmenter_fastai.py` - Redundant training approach
- `models/train_glomeruli_transfer_learning.py` - Redundant training approach
- `models/data_loader.py` - Moved to data_management/ directory
- `evaluation/glomeruli_evaluator.py` - Duplicate functionality, merge into evaluate_glomeruli_model.py

### **Data Management Files - Essential vs. Legacy:**

**✅ ESSENTIAL (Keep and Move to `data_management/`):**
- `organize_lucchi_dataset.py` - Core functionality for mitochondria dataset organization
- `output_manager.py` - Core utility for managing output directories
- `metadata_processor.py` - Core functionality for glomeruli scoring and downstream analysis

**❌ LEGACY/SPECIFIC (Remove or Archive):**
- `organize_raw_data.py` - Pre-eclampsia specific data organization (not needed for current task)
- `reorganize_preeclampsia_data.py` - Pre-eclampsia specific cleanup (not needed for current task)
- `auto_suggestion.py` - Hardware suggestion system (redundant with existing utils)
- `common.py` - Generic utilities (likely redundant)
- `env_check.py` - Environment validation (redundant with existing utils)
- `runtime_check.py` - Runtime validation (redundant with existing utils)

**✅ ESSENTIAL FOR COMPLETE PIPELINE (Keep and Move to `data/`):**
- `metadata_processor.py` - **Critical for glomeruli scoring and downstream analysis** (enables complete workflow)

**Rationale:** Focus only on the essential data management functions needed for mitochondria and glomeruli training. Remove legacy code that was specific to pre-eclampsia research or provides redundant functionality.

### **Thoughtful Merging Strategy - Preserve Important Information:**

**Before Removing Any File, We Must:**
1. **Analyze functionality overlap** - Identify what each file does uniquely
2. **Extract valuable approaches** - Preserve useful algorithms, configurations, or methodologies
3. **Merge complementary features** - Combine the best parts of redundant files
4. **Document merged functionality** - Ensure no information is lost
5. **Test merged implementations** - Verify functionality is preserved

**Specific Merging Examples:**

**Training Scripts:**
- `models/train_mitochondria_fastai.py` + `pipeline/retrain_glomeruli_original.py` → `training/train_mitochondria.py`
- **Preserve:** Best training approaches, hyperparameters, data augmentation strategies
- **Merge:** Complementary validation methods, error handling, logging approaches
- **Result:** Single, robust training script with best practices from both sources

**Data Loaders:**
- `models/data_loader.py` + existing data loaders → consolidated `data_management/loaders.py`
- **Preserve:** Efficient data loading patterns, memory optimization techniques
- **Merge:** Support for different data formats, error handling strategies
- **Result:** Comprehensive data loading system with no functionality loss

**Image Processing Consolidation:**
- `core/preprocessing.py` + `data/preprocessing.py` + `processing/patchify.py` + `utils/create_mitochondria_patches.py` → consolidated `processing/` module
- **Preserve:** All preprocessing functions, patchification logic, mitochondria-specific processing
- **Merge:** Unified image processing API, consistent error handling, shared utilities
- **Result:** Single, comprehensive processing module with no functionality loss
- **Eliminates redundancy:** No more scattered image processing functions across multiple directories

**Core Directory Consolidation:**
- `core/data_loading.py` + `core/model_loading.py` → consolidated `data_management/` module
- **Preserve:** All core data and model loading functionality
- **Merge:** Unified data management API with consistent error handling
- **Result:** Single, comprehensive data management system
- **Core becomes:** Pure constants, types, and abstract interfaces only

**Utility Functions:**
- Redundant validation functions → single, comprehensive validation module
- **Preserve:** All validation logic, edge case handling, error messages
- **Merge:** Consistent API, unified error handling, shared configuration
- **Result:** More robust utilities with broader coverage

**Documentation Requirements:**
- **Before deletion:** Document what functionality is being merged
- **After merging:** Verify all features work and are accessible
- **Migration guide:** Document how to use the new consolidated functions

### **Key Principles:**
1. **Training scripts** go in `training/` - only the 2 we need (mito + glom)
2. **Inference scripts** go in `inference/` - both mito and glom inference
3. **Data management** consolidates in `data_management/` - all data processing, organization, and loading
4. **Image processing** consolidates in `processing/` - all preprocessing, patchification, and conversion
5. **Core directory** contains only constants, types, and abstract interfaces
6. **Pipeline scripts** only handle orchestration - no training or inference logic
7. **Models directory** only contains model architecture definitions
8. **Utils directory** only contains pure utility functions
9. **Thoughtful merging** - preserve important information while eliminating redundancy

**Pipeline Definition:** Pipeline scripts should only handle **orchestration** - coordinating the flow between training, inference, and other components. They should not contain actual training or inference logic.

## Out of Scope

- WSL2 and NVIDIA 3080 setup (user will handle infrastructure)
- Feature extraction pipeline implementation (separate phase)
- Endotheliosis quantification model (separate phase)
- Data preprocessing changes (use existing data structures)

## Expected Deliverable

1. Working `train_mitochondria.py` script that achieves >70% validation accuracy
2. Reorganized training infrastructure with clear separation from pipeline scripts
3. Updated `train_glomeruli.py` script that uses new mitochondria model as base
4. Tested integration of new models with existing inference pipeline
