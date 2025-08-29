# Spec Tasks

> Spec: Model Retraining and Training Infrastructure Organization
> Created: 2025-08-26

## Tasks

- [x] 1. **Reorganize Core Directory and Eliminate Mixed Concerns**
  - [x] 1.1 Write tests for new directory structure and script organization
  - [x] 1.2 Move `core/data_loading.py` to `data_management/data_loading.py`
  - [x] 1.3 Move `core/preprocessing.py` to `processing/preprocessing.py`
  - [x] 1.4 Move `core/model_loading.py` to `data_management/model_loading.py`
  - [x] 1.5 Keep only `core/constants.py` and create `core/types.py` for abstract interfaces
  - [x] 1.6 Update imports and references throughout the codebase
  - [x] 1.7 Verify core directory now contains only constants, types, and abstract interfaces
  - [x] 1.8 Verify all moved functions work in their new locations

- [x] 2. **Consolidate Data Management and Eliminate Redundancy**
  - [x] 2.1 Create dedicated `src/eq/data_management/` directory
  - [x] 2.2 Move `data/loaders.py` to `data_management/loaders.py`
  - [x] 2.3 Move `data/config.py` to `data_management/config.py`
  - [x] 2.4 Move `utils/organize_lucchi_dataset.py` to `data_management/`
  - [x] 2.5 Move `utils/output_manager.py` to `data_management/`
  - [x] 2.6 Move `utils/metadata_processor.py` to `data_management/`
  - [x] 2.7 Remove redundant `data/preprocessing.py` (functionality moved to processing/)
  - [x] 2.8 Update imports and references throughout the codebase
  - [x] 2.9 Verify data_management directory consolidation works correctly

t- [x] 2.5. **Clean Up EQ Directory and Remove Redundancy**
  - [x] 2.5.1 Remove entire `src/eq/data/` directory (legacy compatibility layer)
  - [x] 2.5.2 Remove legacy utils files: `organize_raw_data.py`, `reorganize_preeclampsia_data.py`, `auto_suggestion.py`, `common.py`, `env_check.py`, `runtime_check.py`
  - [x] 2.5.3 Move `utils/create_mitochondria_patches.py` to `processing/`
  - [x] 2.5.4 Move `utils/model_loader.py` to `data_management/`
  - [x] 2.5.5 Update all imports and references throughout the codebase
  - [x] 2.5.6 Verify clean directory structure with no redundant files or empty directories
  - [x] 2.5.7 Ensure all tests pass after cleanup

- [x] 3. **Consolidate Image Processing and Eliminate Scattered Functions**
  - [x] 3.1 Move `data/preprocessing.py` to `processing/preprocessing.py`
  - [x] 3.2 Rename `processing/patchify.py` to `processing/image_mask_preprocessing.py`
  - [x] 3.3 Move `utils/create_mitochondria_patches.py` to `processing/`
  - [x] 3.4 Keep `processing/convert_files.py` in place
  - [x] 3.5 Consolidate all preprocessing functions into unified API
  - [x] 3.6 Update imports and references throughout the codebase
  - [x] 3.7 Verify processing directory consolidation works correctly

- [x] 4. **Create Training Infrastructure and Move Training Scripts**
  - [x] 4.1 Create dedicated `src/eq/training/` directory
  - [x] 4.2 Move `models/train_mitochondria_fastai.py` to `training/train_mitochondria.py`
  - [x] 4.3 Move `pipeline/retrain_glomeruli_original.py` to `training/train_glomeruli.py`
  - [x] 4.4 Remove redundant `models/train_segmenter_fastai.py`
  - [x] 4.5 Remove redundant `models/train_glomeruli_transfer_learning.py`
  - [x] 4.6 Update imports and references throughout the codebase
  - [x] 4.7 Verify training directory structure works correctly

- [x] 5. **Create Inference Infrastructure and Move Inference Scripts**
  - [x] 5.1 Create dedicated `src/eq/inference/` directory
  - [x] 5.2 Move `pipeline/run_glomeruli_prediction.py` to `inference/`
  - [x] 5.3 Move `pipeline/gpu_inference.py` to `inference/`
  - [x] 5.4 Create new `inference/run_mitochondria_prediction.py` (missing script)
  - [x] 5.5 Remove duplicate `pipeline/run_glomeruli_prediction_fixed.py`
  - [x] 5.6 Remove legacy `pipeline/historical_glomeruli_inference.py`
  - [x] 5.7 Update imports and references throughout the codebase
  - [x] 5.8 Verify inference directory structure works correctly

- [x] 6. **Clean Up Models and Pipeline Directories**
  - [x] 6.1 Keep only `models/fastai_segmenter.py` (model architecture)
  - [x] 6.2 Move `models/data_loader.py` to `data_management/`
  - [x] 6.3 Keep only `pipeline/segmentation_pipeline.py` and `pipeline/run_production_pipeline.py`
  - [x] 6.4 Remove demo script `pipeline/run_complete_mito_pipeline.py`
  - [x] 6.5 Move `pipeline/extract_model_weights.py` to `utils/`
  - [x] 6.6 Update imports and references throughout the codebase
  - [x] 6.7 Verify models and pipeline directories are clean and focused

- [x] 7. **Consolidate Evaluation Directory and Remove Duplication**
  - [x] 7.1 Keep `evaluation/evaluate_glomeruli_model.py` (production-ready evaluator)
  - [x] 7.2 Keep `evaluation/segmentation_metrics.py` (pure metrics functions)
  - [x] 7.3 Keep `evaluation/quantification_metrics.py` (quantification metrics)
  - [x] 7.4 Remove duplicate `evaluation/glomeruli_evaluator.py` (merge functionality into main evaluator)
  - [x] 7.5 Update imports and references throughout the codebase
  - [x] 7.6 Verify evaluation directory consolidation works correctly

- [ ] 8. **CRITICAL: Consolidate Duplicated Code and Eliminate Overlap**
  - [x] 8.1 Write tests for consolidated metric calculation system
  - [x] 8.2 Consolidate all metric calculations into `evaluation/segmentation_metrics.py` only
  - [x] 8.3 Remove duplicate metric methods from `evaluate_glomeruli_model.py`, `gpu_inference.py`, and `run_glomeruli_prediction.py`
  - [x] 8.4 Consolidate model loading logic into single `data_management/model_loading.py` module
  - [x] 8.5 Consolidate prediction logic into single `inference/prediction_core.py` module
  - [x] 8.6 Update all modules to use consolidated functions instead of duplicated code
  - [x] 8.7 Verify no functionality is lost during consolidation
  - [x] 8.8 Ensure all tests pass with consolidated code 
  # ERROR: [PARTIALLY COMPLETE 8.8]

- [x] 9. **Fix Broken Imports and Clean Up Remaining Issues**
  - [x] 9.1 Fix broken import in `segmentation_pipeline.py`: `from eq.utils.common import load_pickled_data` (file doesn't exist)
  - [x] 9.2 Replace broken `load_pickled_data` function with proper data loading from `eq.data_management.loaders`
  - [x] 9.3 Remove the TODO comment about temporary evaluation fix (lines 717-723) after Task 8 consolidation
  - [x] 9.4 Verify all imports in `segmentation_pipeline.py` resolve correctly
  - [x] 9.5 Ensure no other broken imports exist across the codebase
  - [x] 9.6 Verify the pipeline runs without import errors

- [x] 10. **Create FastAI v2 Compatible Mitochondria Training Script**
  - [x] 10.1 Write tests for mitochondria training script functionality
  - [x] 10.2 Research and locate historical `segment_mitochondria.py` approach (check archives, git history)
  - [x] 10.3 Create `src/eq/training/train_mitochondria.py` based on historical approach with FastAI v2 compatibility
  - [x] 10.4 Implement proper data loading, model architecture, and training loop
  - [x] 10.5 Add validation metrics and performance monitoring
  - [x] 10.6 Verify mitochondria training script tests pass

- [x] 11. **Update Glomeruli Training Script for FastAI v2 and New Architecture**
  - [x] 11.1 Write tests for updated glomeruli training script
  - [x] 11.2 Refactor `train_glomeruli.py` to use new mitochondria model as base
  - [x] 11.3 Ensure FastAI v2 compatibility throughout the training process
  - [x] 11.4 Implement proper transfer learning workflow from mitochondria to glomeruli
  - [x] 11.5 Add comprehensive validation and performance metrics
  - [x] 11.6 Verify glomeruli training script tests pass

- [x] 12. **Integrate Training Scripts with CLI and Pipeline System**
  - [x] 12.1 Write tests for CLI integration of training commands
  - [x] 12.2 Update CLI commands to use new training script locations
  - [x] 12.3 Ensure training scripts integrate properly with existing inference pipeline
  - [x] 12.4 Add training-specific CLI options and configuration
  - [x] 12.5 Verify CLI integration tests pass

## Technical Dependencies

- **Tasks 1-7** must be completed before **Task 8** (evaluation consolidation must be complete)
- **Task 8** must be completed before **Tasks 9-10** (consolidation must be complete before training setup)
- **Task 9** must be completed before **Task 10** (mitochondria model needed for glomeruli training)
- **Tasks 9-10** must be completed before **Task 11** (training scripts must exist before CLI integration)
- **Task 11** must be completed before **Task 12** (CLI must work before validation tests)
- **CRITICAL**: Task 8 (consolidation) should be completed immediately after Task 7 to eliminate maintenance burden

## Success Criteria

- Training scripts achieve >70% validation accuracy for both models
- Clear separation between training, inference, data management, and pipeline orchestration
- FastAI v2 compatibility throughout the training pipeline
- Proper integration with existing inference pipeline
- All tests pass with 100% success rate
- Training infrastructure is maintainable and well-organized
- **CRITICAL**: No duplicate functionality across directories
- **CRITICAL**: Single source of truth for all metric calculations, model loading, and prediction logic
- Clean, logical directory structure with single responsibilities
- **CONSOLIDATION COMPLETE**: All duplicated code eliminated, single implementation for each functionality
