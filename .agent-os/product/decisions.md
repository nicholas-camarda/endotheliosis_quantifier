# Architecture and Development Decisions

## Repository Structure Decisions

### Package Organization
- **Decision**: Use `src/eq` package structure for clean separation
- **Rationale**: Follows modern Python packaging best practices, enables proper testing and distribution
- **Status**: ✅ Implemented

### Module Separation
- **Decision**: Separate modules for data, training, inference, evaluation, and pipeline
- **Rationale**: Enables independent development and testing, clear separation of concerns
- **Status**: ✅ Implemented

## Technology Stack Decisions

### FastAI v1 → v2 Migration
- **Decision**: Modernize from FastAI v1 to v2 for current compatibility
- **Rationale**: FastAI v1 is deprecated, v2 provides better PyTorch integration and performance
- **Status**: 🚧 In Progress - Priority 1

### PyTorch Version
- **Decision**: Use PyTorch 2.2+ with CUDA 12.1 support
- **Rationale**: Latest stable version with optimal GPU performance and FastAI v2 compatibility
- **Status**: 🔄 Needs Testing

### Environment Management
- **Decision**: Use conda/mamba for environment management
- **Rationale**: Better dependency resolution for scientific computing packages
- **Status**: ✅ Implemented

## Architecture Decisions

### Two-Stage Training Approach
- **Decision**: Pre-train on mitochondria EM data, then transfer learn on glomeruli H&E
- **Rationale**: EM data provides robust segmentation features, transfer learning improves H&E performance
- **Status**: ✅ Implemented (needs v2 update)

### Binary Segmentation
- **Decision**: Use binary segmentation (0=background, 1=glomeruli) with threshold 127
- **Rationale**: Proven approach for medical image segmentation, simple and effective
- **Status**: ✅ Implemented

### Hardware Detection
- **Decision**: Automatic detection of MPS/CUDA/CPU with fallbacks
- **Rationale**: Enables code to run on different hardware without modification
- **Status**: ✅ Implemented

## Data Management Decisions

### Caching Strategy
- **Decision**: Implement pickle-based caching for processed data
- **Rationale**: Improves training speed by avoiding repeated preprocessing
- **Status**: ✅ Implemented

### Image Format Support
- **Decision**: Support TIF, JPG, PNG formats with TiffFile for high-performance TIFF handling
- **Rationale**: Medical images often in TIFF format, need efficient handling
- **Status**: ✅ Implemented

### Patchification
- **Decision**: Use 224x224 patches for training with configurable sizes
- **Rationale**: Standard size for many pre-trained models, enables batch processing
- **Status**: ✅ Implemented

## Training Decisions

### Model Architecture
- **Decision**: Use UNet for segmentation tasks
- **Rationale**: Proven architecture for medical image segmentation, good for small datasets
- **Status**: ✅ Implemented

### Transfer Learning
- **Decision**: Use mitochondria model as base for glomeruli training
- **Rationale**: Leverages learned features from similar segmentation task
- **Status**: ✅ Implemented (needs v2 update)

### Batch Size Strategy
- **Decision**: Dynamic batch size adjustment based on available GPU memory
- **Rationale**: Optimizes training performance while avoiding out-of-memory errors
- **Status**: ✅ Implemented

## Modernization Priorities

### Phase 1: FastAI v2 Compatibility
- **Priority**: Highest - blocking all other development
- **Approach**: Update imports and API calls systematically
- **Risk**: Breaking changes in FastAI v2 API
- **Mitigation**: Start with minimal examples, test incrementally

### Phase 2: Pipeline Validation
- **Priority**: High - ensure end-to-end functionality
- **Approach**: Test complete pipeline with real data
- **Risk**: Hidden bugs in data flow
- **Mitigation**: Use real medical data from the start

### Phase 3: Quantification Model
- **Priority**: Medium - complete the missing piece
- **Approach**: Implement feature extraction and ML models
- **Risk**: Feature engineering complexity
- **Mitigation**: Start with proven approaches (Random Forest)

## Testing Strategy Decisions

### Test Data
- **Decision**: Use real medical data for testing
- **Rationale**: Ensures compatibility with actual use case
- **Status**: 🔄 Needs Implementation

### Test Coverage
- **Decision**: Focus on integration and end-to-end testing
- **Rationale**: More valuable for pipeline validation than unit tests
- **Status**: 🔄 Needs Implementation

### Validation Approach
- **Decision**: Use expert annotations as ground truth
- **Rationale**: Medical accuracy is critical for clinical applications
- **Status**: 🔄 Needs Implementation

## Configuration Management Decisions

### Central Configuration
- **Decision**: Single configuration file for all parameters
- **Rationale**: Makes pipeline transparent and easy to modify
- **Status**: 🔄 Needs Implementation

### Path Management
- **Decision**: Centralized path handling for user data
- **Rationale**: Simplifies user setup and reduces hardcoded paths
- **Status**: 🔄 Needs Implementation

### Environment Variables
- **Decision**: Use environment variables for runtime overrides
- **Rationale**: Enables flexible deployment without code changes
- **Status**: 🔄 Needs Implementation

## Documentation Decisions

### Lab Notebook Style
- **Decision**: GitHub README as lab notebook with methodology and results
- **Rationale**: Makes research transparent and reproducible
- **Status**: 🔄 Needs Implementation

### Code Documentation
- **Decision**: Comprehensive docstrings and type hints
- **Rationale**: Enables maintainability and collaboration
- **Status**: 🔄 Needs Implementation

### User Guides
- **Decision**: Step-by-step tutorials for common workflows
- **Rationale**: Enables new users to run pipeline independently
- **Status**: 🔄 Needs Implementation
