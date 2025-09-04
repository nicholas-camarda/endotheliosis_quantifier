# Technical Architecture

## Core Technology Stack

### Programming Language
- **Python 3.10**: Primary language with type hints and modern syntax

### Deep Learning Framework
- **FastAI v2**: Modern deep learning library for medical image analysis
- **PyTorch 2.2+**: Backend framework with CUDA 12.1 support
- **TorchVision**: Image processing and augmentation utilities

### Image Processing
- **OpenCV**: Computer vision operations and image manipulation
- **Pillow (PIL)**: Image I/O and basic processing
- **TiffFile**: High-performance TIFF image handling
- **Albumentations**: Advanced image augmentation for training

### Data Management
- **NumPy 1.26**: Numerical computing and array operations
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning utilities and metrics
- **LightGBM/XGBoost**: Gradient boosting for quantification models

### Development Tools
- **Conda/Mamba**: Environment management and package installation
- **Ruff**: Fast Python linter and formatter
- **PyTest**: Testing framework for validation
- **PyRight**: Type checking and static analysis

## Architecture Patterns

### Modular Design
- **Core Module**: Constants, types, and shared utilities
- **Data Management**: Loading, preprocessing, and caching
- **Training**: Separate modules for mitochondria and glomeruli models
- **Inference**: Prediction engines with GPU optimization
- **Pipeline**: Orchestration and end-to-end execution
- **Evaluation**: Metrics calculation and performance assessment

### Configuration Management
- **Central Configuration**: Single source of truth for all parameters
- **YAML Configs**: Human-readable configuration files
- **Environment Variables**: Runtime configuration overrides
- **Path Management**: Centralized path handling for user data

### Hardware Optimization
- **CUDA Support**: NVIDIA GPU acceleration for training and inference
- **MPS Support**: Apple Silicon optimization with fallbacks
- **CPU Fallback**: Automatic fallback for development and testing
- **Batch Size Optimization**: Dynamic adjustment based on available memory

## Data Flow Architecture

### Training Pipeline
```
Raw Images → Preprocessing → Patchification → Binary Masks → UNet Training → Model Checkpoints
```

### Transfer Learning
```
Mitochondria Model → Feature Extraction → Glomeruli Fine-tuning → Glomeruli Model
```

### Production Pipeline
```
Input Images → Segmentation → ROI Extraction → Feature Engineering → Quantification → Results
```

## Development Standards

### Code Quality
- **Type Hints**: Full type annotation for maintainability
- **Docstrings**: Comprehensive documentation for all functions
- **Error Handling**: Robust error handling with meaningful messages
- **Logging**: Structured logging throughout the pipeline

### Testing Strategy
- **Unit Tests**: Individual function validation
- **Integration Tests**: Component interaction testing
- **End-to-End Tests**: Complete pipeline validation
- **Real Data Testing**: Validation on actual medical images

### Performance Requirements
- **Training Speed**: Efficient training with GPU utilization
- **Inference Speed**: Real-time prediction capabilities
- **Memory Efficiency**: Optimized for large image datasets
- **Scalability**: Handle datasets from 100s to 1000s of images

## Deployment Architecture

### Local Development
- **Conda Environment**: Isolated Python environment
- **Development Mode**: Fast iteration with reduced data
- **Debug Mode**: Comprehensive logging and error reporting

### Production Deployment
- **GPU Optimization**: Maximum performance for model training
- **Batch Processing**: Efficient handling of large datasets
- **Result Persistence**: Comprehensive output and logging
- **Reproducibility**: Deterministic results with fixed seeds
