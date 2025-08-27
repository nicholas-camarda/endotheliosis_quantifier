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

- [x] 2.5. **Clean Up EQ Directory and Remove Redundancy**
  - [x] 2.5.1 Remove entire `src/eq/data/` directory (legacy compatibility layer)
  - [x] 2.5.2 Remove legacy utils files: `organize_raw_data.py`, `reorganize_preeclampsia_data.py`, `auto_suggestion.py`, `common.py`, `env_check.py`, `runtime_check.py`
  - [x] 2.5.3 Move `utils/create_mitochondria_patches.py` to `processing/`
  - [x] 2.5.4 Move `utils/model_loader.py` to `data_management/`
  - [x] 2.5.5 Update all imports and references throughout the codebase
  - [x] 2.5.6 Verify clean directory structure with no redundant files or empty directories
  - [x] 2.5.7 Ensure all tests pass after cleanup

- [ ] 3. **Consolidate Image Processing and Eliminate Scattered Functions**
  - [ ] 3.1 Move `data/preprocessing.py` to `processing/preprocessing.py`
  - [ ] 3.2 Rename `processing/patchify.py` to `processing/image_mask_preprocessing.py`
  - [ ] 3.3 Move `utils/create_mitochondria_patches.py` to `processing/`
  - [ ] 3.4 Keep `processing/convert_files.py` in place
  - [ ] 3.5 Consolidate all preprocessing functions into unified API
  - [ ] 3.6 Update imports and references throughout the codebase
  - [ ] 3.7 Verify processing directory consolidation works correctly

- [ ] 4. **Create Training Infrastructure and Move Training Scripts**
  - [ ] 4.1 Create dedicated `src/eq/training/` directory
  - [ ] 4.2 Move `models/train_mitochondria_fastai.py` to `training/train_mitochondria.py`
  - [ ] 4.3 Move `pipeline/retrain_glomeruli_original.py` to `training/train_glomeruli.py`
  - [ ] 4.4 Remove redundant `models/train_segmenter_fastai.py`
  - [ ] 4.5 Remove redundant `models/train_glomeruli_transfer_learning.py`
  - [ ] 4.6 Update imports and references throughout the codebase
  - [ ] 4.7 Verify training directory structure works correctly

- [ ] 5. **Create Inference Infrastructure and Move Inference Scripts**
  - [ ] 5.1 Create dedicated `src/eq/inference/` directory
  - [ ] 5.2 Move `pipeline/run_glomeruli_prediction.py` to `inference/`
  - [ ] 5.3 Move `pipeline/gpu_inference.py` to `inference/`
  - [ ] 5.4 Create new `inference/run_mitochondria_prediction.py` (missing script)
  - [ ] 5.5 Remove duplicate `pipeline/run_glomeruli_prediction_fixed.py`
  - [ ] 5.6 Remove legacy `pipeline/historical_glomeruli_inference.py`
  - [ ] 5.7 Update imports and references throughout the codebase
  - [ ] 5.8 Verify inference directory structure works correctly

- [ ] 6. **Clean Up Models and Pipeline Directories**
  - [ ] 6.1 Keep only `models/fastai_segmenter.py` (model architecture)
  - [ ] 6.2 Move `models/data_loader.py` to `data_management/`
  - [ ] 6.3 Keep only `pipeline/segmentation_pipeline.py` and `pipeline/run_production_pipeline.py`
  - [ ] 6.4 Remove demo script `pipeline/run_complete_mito_pipeline.py`
  - [ ] 6.5 Move `pipeline/extract_model_weights.py` to `utils/`
  - [ ] 6.6 Update imports and references throughout the codebase
  - [ ] 6.7 Verify models and pipeline directories are clean and focused

- [ ] 7. **Consolidate Evaluation Directory and Remove Duplication**
  - [ ] 7.1 Keep `evaluation/evaluate_glomeruli_model.py` (production-ready evaluator)
  - [ ] 7.2 Keep `evaluation/segmentation_metrics.py` (pure metrics functions)
  - [ ] 7.3 Keep `evaluation/quantification_metrics.py` (quantification metrics)
  - [ ] 7.4 Remove duplicate `evaluation/glomeruli_evaluator.py` (merge functionality into main evaluator)
  - [ ] 7.5 Update imports and references throughout the codebase
  - [ ] 7.6 Verify evaluation directory consolidation works correctly

- [ ] 8. **Clean Up Utils Directory and Remove Mixed Concerns**
  - [ ] 8.1 Keep pure utility functions: `hardware_detection.py`, `config_manager.py`, `backend_manager.py`, `mode_manager.py`, `logger.py`, `paths.py`
  - [ ] 8.2 Move `extract_model_weights.py` from pipeline to utils
  - [ ] 8.3 Remove pre-eclampsia specific files: `organize_raw_data.py`, `reorganize_preeclampsia_data.py`
  - [ ] 8.4 Remove redundant utilities: `auto_suggestion.py`, `common.py`, `env_check.py`, `runtime_check.py`
  - [ ] 8.5 Update imports and references throughout the codebase
  - [ ] 8.6 Verify utils directory contains only pure utility functions

- [ ] 9. **Create FastAI v2 Compatible Mitochondria Training Script**
  - [ ] 9.1 Write tests for mitochondria training script functionality
  - [ ] 9.2 Research and locate historical `segment_mitochondria.py` approach (check archives, git history)
  - [ ] 9.3 Create `src/eq/training/train_mitochondria.py` based on historical approach with FastAI v2 compatibility
  - [ ] 9.4 Implement proper data loading, model architecture, and training loop
  - [ ] 9.5 Add validation metrics and performance monitoring
  - [ ] 9.6 Verify mitochondria training script tests pass

- [ ] 10. **Update Glomeruli Training Script for FastAI v2 and New Architecture**
  - [ ] 10.1 Write tests for updated glomeruli training script
  - [ ] 10.2 Refactor `train_glomeruli.py` to use new mitochondria model as base
  - [ ] 10.3 Ensure FastAI v2 compatibility throughout the training process
  - [ ] 10.4 Implement proper transfer learning workflow from mitochondria to glomeruli
  - [ ] 10.5 Add comprehensive validation and performance metrics
  - [ ] 10.6 Verify glomeruli training script tests pass

- [ ] 11. **Integrate Training Scripts with CLI and Pipeline System**
  - [ ] 11.1 Write tests for CLI integration of training commands
  - [ ] 11.2 Update CLI commands to use new training script locations
  - [ ] 11.3 Ensure training scripts integrate properly with existing inference pipeline
  - [ ] 11.4 Add training-specific CLI options and configuration
  - [ ] 11.5 Verify CLI integration tests pass

- [ ] 12. **Validate Model Performance and Integration**
  - [ ] 12.1 Write tests for model performance validation
  - [ ] 12.2 Test mitochondria model achieves >70% validation accuracy
  - [ ] 12.3 Test glomeruli model achieves >70% validation accuracy using mitochondria as base
  - [ ] 12.4 Verify new models integrate properly with existing inference pipeline
  - [ ] 12.5 Run end-to-end integration tests
  - [ ] 12.6 Verify all validation and integration tests pass

## Technical Dependencies

- **Tasks 1-8** must be completed before **Tasks 9-10** (infrastructure setup)
- **Task 9** must be completed before **Task 10** (mitochondria model needed for glomeruli training)
- **Tasks 9-10** must be completed before **Task 11** (training scripts must exist before CLI integration)
- **Task 11** must be completed before **Task 12** (CLI must work before running validation tests)

## Success Criteria

- Training scripts achieve >70% validation accuracy for both models
- Clear separation between training, inference, data management, and pipeline orchestration
- FastAI v2 compatibility throughout the training pipeline
- Proper integration with existing inference pipeline
- All tests pass with 100% success rate
- Training infrastructure is maintainable and well-organized
- No duplicate functionality across directories
- Clean, logical directory structure with single responsibilities
