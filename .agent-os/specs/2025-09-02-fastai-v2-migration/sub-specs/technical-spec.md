# Technical Specification

This is the technical specification for the FastAI v2 migration spec detailed in @.agent-os/specs/2025-09-02-fastai-v2-migration/spec.md

## Technical Requirements

### 0. Data Processing Validation (PREREQUISITE)
- **Validate existing data processing components** (Completed 2025-09-03)
  - Test image loading with real medical image data
  - Test image preprocessing and normalization
  - Test image patching/segmentation logic
  - Test mask generation and validation
  - Ensure data processing pipeline produces correct outputs
  - Use subset of real data for validation
  - Result: `eq process-data` creates `derived_data/` with auto-detected masks; 256x256 patches; binary masks via JPEG(100)

### 1. Data Pipeline Migration
- **Convert SegmentationDataLoaders.from_folder() to DataBlock approach**
  - Replace `SegmentationDataLoaders.from_folder()` calls with `DataBlock(blocks=(ImageBlock, MaskBlock))`
  - Update data loading patterns to use `get_items`, `splitter`, and `get_y` functions
  - Ensure mask handling works correctly with FastAI v2 PILMask format
  - Test with minimal sample data before full implementation

### 2. Training Module Updates
- **Update unet_learner calls**
  - Add required `n_out` parameter to all `unet_learner()` calls
  - Verify `n_out=1` for binary segmentation tasks
  - Test model initialization and training start
- **Fix import statements**
  - Update all `from fastai.vision.all import *` statements
  - Replace deprecated function calls with current v2 equivalents
  - Ensure all required classes and functions are accessible
- **Transfer Learning Namespace Issue** (CRITICAL)
  - FastAI v2 model loading requires custom functions in global namespace
  - When loading saved models, custom functions like `get_y` must be available
  - Need to implement proper namespace setup for transfer learning between models
  - Current workaround: Train glomeruli from scratch instead of transfer learning

### 3. Callback Event Updates
- **Update event naming conventions**
  - Change `begin_fit` → `before_fit`
  - Change `begin_validate` → `before_validate`
  - Change `begin_epoch` → `before_epoch`
  - Change `begin_train` → `before_train`
  - Change `begin_batch` → `before_batch`
- **Verify callback compatibility**
  - Test all custom callbacks with new event names
  - Ensure training progress and logging work correctly

### 4. Inference Pipeline Updates
- **Update prediction code**
  - Fix any FastAI v1 specific prediction methods
  - Ensure model loading works with v2 APIs
  - Test end-to-end prediction pipeline
- **Validate output formats**
  - Verify prediction tensors are correctly formatted
  - Ensure mask extraction and processing work

### 5. Testing and Validation
- **Component-level testing**
  - Test data loading with sample images and masks
  - Test training initialization and basic training steps
  - Test model saving and loading
- **Integration testing**
  - Test complete training pipeline end-to-end
  - Test complete inference pipeline end-to-end
  - Validate results against expected outputs

## External Dependencies

### FastAI v2.7.13
- **Purpose**: Modern deep learning library with PyTorch 2.1.0 compatibility
- **Justification**: Required for current package compatibility and modern features
- **Version Requirements**: 2.7.13 or later for PyTorch 2.1.0 support

### PyTorch 2.1.0
- **Purpose**: Backend deep learning framework
- **Justification**: Current stable version with CUDA 11.5 support for WSL2
- **Version Requirements**: 2.1.0 for environment stability

### Medical Image Libraries
- **Purpose**: Image I/O and processing for histology data
- **Justification**: Required for loading and preprocessing medical images
- **Version Requirements**: Current working versions (OpenCV, TiffFile, Pillow)

## Migration Strategy

### Phase 0: Data Processing Validation (Priority 0 - PREREQUISITE)
1. Test existing image loading and preprocessing components (done)
2. Validate patching/segmentation logic with real data (done)
3. Ensure data processing pipeline works correctly (done)
4. Implemented `eq process-data` CLI with auto mask detection (done)
5. Document any issues found (none blocking)

### Phase 1: Data Pipeline (Priority 1)
1. Create minimal test script with DataBlock approach
2. Test with small dataset to validate mask handling
3. Update all data loading modules
4. Validate preprocessing pipeline

### Phase 2: Training Modules (Priority 2)
1. Update mitochondria training script
2. Update glomeruli training script
3. Test training initialization
4. Test basic training steps

### Phase 3: Inference Pipeline (Priority 3)
1. Update prediction code
2. Test model loading
3. Test end-to-end prediction
4. Validate results

### Phase 4: Integration Testing (Priority 4)
1. Test complete training pipeline
2. Test complete inference pipeline
3. Validate with real medical data
4. Document any remaining issues

## Success Criteria

- [x] Data processing pipeline works correctly with real medical images
- [x] All training scripts run without FastAI v1 errors
- [x] Data loading works with DataBlock approach
- [x] Training produces valid models
- [ ] Inference pipeline produces correct results
- [x] End-to-end pipeline completes successfully (from scratch training)
- [x] No deprecated FastAI v1 code remains
- [x] **TRANSFER LEARNING**: Mitochondria → glomeruli transfer learning works (namespace issue resolved)

## Current Status Summary

**✅ COMPLETED**:
- Data processing pipeline (3,960 mitochondria patches, 17,640 glomeruli patches)
- Mitochondria training (5 epochs, dice score 0.97)
- Glomeruli training from scratch (3 epochs, 8,640 train samples)
- FastAI v2 DataBlock migration
- CLI interfaces for both training modules
- End-to-end pipeline validation

**✅ TRANSFER LEARNING RESOLVED**:
- Transfer learning from mitochondria to glomeruli now works with proper namespace handling
- Implemented standardized getter functions and weight compatibility checking
- Tested successfully: 2 epochs, dice score 0.985
