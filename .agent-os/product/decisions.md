# Decisions Log

## Platform & Performance
- Must run on macOS Apple Silicon (M1) for inference and light workflows.
- Retain optional Windows (WSL2) + CUDA path (RTX 3080) for training; document when required.

## Framework
- **DECISION MADE 2025-08-21**: Migrate to fastai/PyTorch for all segmentation and training workflows.
- Leverage existing working fastai implementations from notebooks folder.
- Use PyTorch with MPS (Metal Performance Shaders) for Apple Silicon and CUDA for RTX 3080.
- This provides superior cross-platform support and leverages proven working code.

## Packaging & Structure
- Rename repository to `endotheliosis_quantifier`.
- Adopt `src/eq` package with CLI entrypoints for the 4 pipeline steps.
- Centralize configuration; remove hard-coded paths.

## Reproducibility & Standards
- Use Conda env `eq` exclusively for installs/execution.
- Follow `.agent-os/standards/code-style/python-style.md`.
- Separate test scripts for new/updated logic; assert specific outputs on test data.

## 2025-08-21: Segmentation Pipeline Architecture Implementation
**ID:** DEC-005
**Status:** Accepted
**Category:** technical

### Decision
- Implement two-stage segmentation pipeline: mitochondria pretraining followed by glomeruli fine-tuning
- Use transfer learning approach with mitochondria EM data as foundation for general image segmentation features
- Create separate configuration files for each stage with programmatic train/test splits
- Implement pipeline orchestrator script for automated execution of both stages

### Rationale
- Transfer learning from mitochondria data improves performance on glomeruli segmentation task
- Clear separation of concerns between pretraining and fine-tuning stages
- Programmatic data splits ensure reproducibility across different datasets
- Configuration-driven approach enables easy adaptation to new data sources

### Implementation Details
1. **Stage 1 - Mitochondria Pretraining**: 
   - Raw TIF stack → individual TIFs → JPG conversion → patches → U-Net training
   - Output: `mito_dynamic_unet_seg_model-e50_b16.pkl` (228MB)
   - Config: `configs/mito_pretraining_config.yaml`
   - Train/test split: 80/20 (seed=42)

2. **Stage 2 - Glomeruli Fine-tuning**:
   - Preeclampsia TIFs → Label Studio annotations → fine-tuning from mitochondria model
   - Output: `glomerulus_segmentation_model-dynamic_unet-e50_b16_s84.pkl` (250MB)
   - Config: `configs/glomeruli_finetuning_config.yaml`
   - Train/val/test split: 70/15/15 (seed=84, different from mito)

3. **Pipeline Orchestration**:
   - Script: `src/eq/pipeline/segmentation_pipeline.py`
   - Commands: `--stage mito`, `--stage glomeruli`, `--stage all`
   - Automatic directory creation and path validation

### Benefits Achieved
- **Reproducibility**: Clear data flow and configuration for each stage
- **Extensibility**: Easy to add new datasets or tissue types
- **Documentation**: Comprehensive workflow documentation in product technical README
- **Automation**: Single script orchestrates complete pipeline execution

### Context
This decision establishes the foundation for reproducible segmentation model training and enables researchers to easily extend the pipeline with new data sources. The transfer learning approach leverages the proven effectiveness of pretraining on EM data for histology applications.

## 2025-08-21: Repository Hygiene and Utilities Consolidation
**ID:** DEC-001
**Status:** Accepted
**Category:** technical

### Decision
- Consolidate `scripts/utils/` to a minimal, well-documented helper set; remove or merge redundant utilities.
- Do not track large/ephemeral folders: `notebooks/`, `earlier_models/`. Update `.gitignore` accordingly.

### Context
Improves maintainability, reduces confusion for users, and avoids committing bulky, non-source artifacts. Aligns with packaging goals and cross-platform reproducibility.

## 2025-01-27: CLI Pipeline System Implementation
**ID:** DEC-002
**Status:** Accepted
**Category:** technical

### Decision
- Implement comprehensive CLI pipeline system with `eq` command and subcommands for each pipeline step.
- Use `src/eq/__main__.py` as the main CLI entry point with argparse-based command parsing.
- Create wrapper functions in each module to handle CLI arguments and orchestrate pipeline steps.
- Install package in development mode with console script entry point.

### Commands Implemented
- `eq data-load` - Load and preprocess data with configurable paths
- `eq train-segmenter` - Train segmentation models with hyperparameter control
- `eq extract-features` - Extract features from images using trained models
- `eq quantify` - Run endotheliosis quantification pipeline
- `eq pipeline` - Execute complete end-to-end pipeline

### Context
Provides a centralized, user-friendly interface for running the endotheliosis quantification pipeline. Eliminates the need for users to understand internal module structure or manually orchestrate pipeline steps. Enables both step-by-step execution and full pipeline runs with consistent parameter handling.

## 2025-01-27: fastai/PyTorch Migration Decision
**ID:** DEC-003
**Status:** Completed
**Category:** framework

### Decision
- **COMPLETED**: Full migration from TensorFlow to fastai/PyTorch for all segmentation and training workflows.
- **COMPLETED**: TensorFlow packages completely removed from environment to eliminate conflicts.
- Convert existing working notebook implementations to production-ready Python modules.
- Implement dual-environment architecture with explicit mode selection between development (M1 Mac/MPS) and production (RTX 3080/CUDA).
- Use PyTorch's native MPS support for Apple Silicon and CUDA for NVIDIA GPUs.

### Rationale
- Existing fastai implementations in notebooks are proven and working
- PyTorch provides superior cross-platform support with native MPS and CUDA backends
- fastai offers higher-level API that works seamlessly across both backends
- **RESOLVED**: Eliminated TensorFlow Metal compatibility issues and OpenMP conflicts
- Leverages existing developer expertise and working code

### Implementation Progress
1. ✅ **COMPLETED**: Updated environment dependencies from TensorFlow to fastai/PyTorch
2. ✅ **COMPLETED**: Removed all TensorFlow packages (tensorflow-macos, tensorflow-metal, tensorflow-hub, tensorflow-estimator)
3. ✅ **COMPLETED**: Implemented comprehensive hardware detection and mode selection system
4. ✅ **COMPLETED**: Created dual-environment architecture foundation with 42 passing tests
5. ✅ **COMPLETED**: Convert working notebook implementations to Python modules (68 passing tests)
6. ✅ **COMPLETED**: Implemented dual-environment architecture with mode selection and backend abstraction (117 passing tests)
7. ✅ **COMPLETED**: CLI pipeline updated to support dual-environment architecture with mode selection and hardware detection
8. ✅ **COMPLETED**: Full TensorFlow to fastai migration completed with comprehensive test coverage (162 passing tests)

### 2025-08-21: Production Heuristic Tightening
- **Change**: Production mode is now only suggested/valid on CUDA systems, or on Apple Silicon (MPS) with ≥32GB unified memory.
- **Rationale**: Typical M1/M2 8-16GB machines are excellent for development/inference but underpowered for sustained production training.
- **Effect**:
  - AUTO will default to DEVELOPMENT on M1 with <32GB.
  - Validation rejects PRODUCTION on CPU-only or MPS <32GB; users can still explicitly choose DEVELOPMENT.

### Context
This decision addresses the dual-environment architecture requirements while leveraging existing working implementations. The fastai/PyTorch approach provides better cross-platform compatibility and has eliminated the TensorFlow Metal issues that were previously encountered. Environment is now clean with no TensorFlow conflicts.

## 2025-08-21: TensorFlow to fastai Migration Completion
**ID:** DEC-004
**Status:** Completed
**Category:** framework

### Decision
- **COMPLETED**: Successfully migrated entire segmentation system from TensorFlow to fastai/PyTorch
- **COMPLETED**: Created comprehensive fastai segmentation module with U-Net support and mode-specific optimizations
- **COMPLETED**: Implemented advanced data pipeline with cache integration and augmentation
- **COMPLETED**: Added mode-specific training strategies (production, development, auto) with hardware-aware optimization
- **COMPLETED**: Comprehensive test coverage with 46 fastai segmentation tests and 162 total tests passing

### Implementation Details
1. **FastaiSegmenter Class**: Complete segmentation implementation with multiple U-Net architectures (ResNet18, 34, 50, 101)
2. **Data Pipeline**: Enhanced data preparation with support for cached pickle files and advanced augmentation
3. **Training Strategies**: Mode-specific training with production (conservative), development (aggressive), and auto (balanced) approaches
4. **Hardware Integration**: Automatic device selection (MPS/CUDA/CPU) with optimal batch sizing and mode validation
5. **Callback System**: Comprehensive monitoring with early stopping, learning rate scheduling, and model checkpointing
6. **Migration Script**: Created `train_segmenter_fastai.py` as drop-in replacement for TensorFlow implementation

### Benefits Achieved
- **Modern Stack**: Replaced TensorFlow with fastai/PyTorch for superior cross-platform support
- **M1 Optimization**: Native MPS acceleration for Apple Silicon with automatic fallback
- **Training Efficiency**: Mode-specific optimizations and advanced augmentation pipeline
- **Maintainability**: Comprehensive test suite (46 tests) ensuring reliability
- **Hardware Awareness**: Automatic detection and optimization for different configurations
- **Backward Compatibility**: Maintains existing data format and CLI interface

### Context
This migration completes the framework transition decision and provides a production-ready fastai segmentation system. The implementation leverages existing working notebook code while adding enterprise-grade features like mode-specific training, hardware optimization, and comprehensive testing. All 162 tests pass, confirming the migration's success and system stability.

## DEC-006: Mitochondria Data Source Documentation and Workflow Guide

**Date**: 2025-08-22  
**Status**: COMPLETED  
**Priority**: HIGH  

### Context
The mitochondria segmentation pipeline required clear documentation of the data source and complete workflow for reproducibility. The original documentation was incomplete regarding where the mitochondria data originated from.

### Decision
1. **Data Source Identified**: The mitochondria data comes from the [Lucchi et al. 2012 benchmark dataset](http://rhoana.rc.fas.harvard.edu/dataset/lucchi.zip) from Harvard's Rhoana server, a widely-used EM dataset for mitochondria segmentation research.

2. **Complete Workflow Documentation**: Added comprehensive "Complete Workflow Guide" section to README.md that documents:
   - Data acquisition and download instructions
   - Step-by-step processing pipeline
   - Model training workflow
   - Transfer learning to glomeruli
   - Reproducibility features
   - Quick start guide for new users

3. **Reproducibility Focus**: Emphasized that users should start with raw data download and process it themselves, rather than hosting large datasets on GitHub.

### Rationale
- **Reproducibility**: Complete workflow documentation ensures other researchers can reproduce results from scratch
- **Data Attribution**: Proper citation of the Lucchi et al. 2012 dataset acknowledges the original research
- **User Experience**: Step-by-step guide makes the pipeline accessible to new users
- **Maintainability**: Clear documentation reduces support burden and improves code adoption

### Implementation
- Updated README.md with comprehensive workflow guide
- Updated technical documentation with correct data source
- Maintained focus on reproducibility rather than data hosting

### Benefits
- **Research Impact**: Other researchers can easily reproduce and build upon the work
- **Academic Integrity**: Proper attribution to original dataset creators
- **Code Adoption**: Clear documentation increases likelihood of code reuse
- **Maintenance**: Reduced need for individual user support

### Next Steps
- Consider adding data validation scripts to verify downloaded dataset integrity
- Add performance benchmarks against published results from Lucchi et al. 2012
- Document any preprocessing differences from the original paper

---

## DEC-007: Patch Size Standardization to 224x224

**Date**: 2025-08-22  
**Status**: COMPLETED  
**Priority**: HIGH  

### Context
During evaluation testing of the pretrained mitochondria model, we encountered a shape mismatch error: `ValueError: operands could not be broadcast together with shapes (256,256) (224,224)`. This revealed that the pretrained model was trained on 224x224 patches, but the pipeline was configured for 256x256.

### Decision
**Standardize entire pipeline to use 224x224 pixel patches** instead of 256x256 to ensure compatibility with existing pretrained models.

### Rationale
- **Model Compatibility**: Existing pretrained models (e.g., `backups/mito_dynamic_unet_seg_model-e50_b16.pkl`) expect 224x224 input
- **Evaluation Accuracy**: Proper metrics calculation without resizing artifacts
- **Consistency**: Uniform patch size throughout entire pipeline
- **Performance**: Slightly faster training due to smaller patches

### Implementation
1. **Configuration Files Updated**:
   - `configs/mito_pretraining_config.yaml`: patches.size: 224, input_size: [224, 224]
   - `configs/glomeruli_finetuning_config.yaml`: resize_to: [224, 224], input_size: [224, 224], mask_resize: [224, 224]

2. **Python Code Updated**:
   - All training functions default to `image_size=224`
   - All config defaults updated to `[224, 224]`
   - Pipeline banners show "224x224 patches"
   - Data loaders default to 224x224

3. **Documentation Updated**:
   - README.md: All examples and references updated to 224x224
   - Function signatures and examples updated
   - Console output updated

### Backward Compatibility
- **Automatic Resizing**: Pipeline includes resize logic for mixed scenarios
- **Existing Patches**: 256x256 patches work during evaluation with automatic resizing
- **New Training**: Generates 224x224 patches natively

### Benefits
- **Perfect Model Alignment**: Works seamlessly with pretrained models
- **Quantified Evaluation**: Proper metrics without shape mismatches
- **Future Consistency**: All new training uses correct patch size
- **Performance**: Slightly faster training and inference

### Impact
- **Evaluation**: ✅ Working with quantified metrics (Dice: 0.60, IoU: 0.44, Pixel Acc: 95%)
- **Training**: ✅ Updated to 224x224 defaults
- **Documentation**: ✅ All references updated
- **Configs**: ✅ All files aligned

### Next Steps
1. **Optional**: Regenerate existing training patches at 224x224 for optimal performance
2. **Future**: All new models will use 224x224 natively
3. **Consider**: Updating existing datasets to 224x224 for consistency

---
