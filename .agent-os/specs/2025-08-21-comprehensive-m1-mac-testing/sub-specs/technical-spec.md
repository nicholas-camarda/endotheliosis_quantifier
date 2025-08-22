# Technical Specification

This is the technical specification for the spec detailed in @~/.agent-os/specs/2025-08-21-comprehensive-m1-mac-testing/2025-08-21-comprehensive-m1-mac-testing.md

## Technical Requirements

### Explicit Mode Selection
- **CLI Mode Flags**: Explicit --mode=development/production command-line options
- **Mode Validation**: Validate mode selection against hardware capabilities
- **Default Mode**: Automatic suggestion of appropriate mode based on hardware
- **Mode Override**: Allow users to override automatic suggestions
- **Mode Persistence**: Remember user's mode preference across sessions

### Hardware Capability Detection
- **GPU Detection**: Detect NVIDIA GPU vs Apple Silicon GPU with specific model identification
- **Memory Assessment**: Determine available memory and compute capabilities
- **Performance Benchmarking**: Assess relative performance capabilities
- **Capability Reporting**: Generate detailed capability reports for users
- **Hardware Classification**: Classify hardware into capability tiers (basic, standard, powerful)

### Automatic Suggestion System
- **Mode Recommendation**: Suggest appropriate mode based on detected capabilities
- **Capability Analysis**: Analyze hardware against mode requirements
- **Performance Prediction**: Predict performance characteristics for each mode
- **Suggestion Rationale**: Provide clear explanation for mode suggestions
- **Override Warnings**: Warn users when overriding suggestions might cause issues

### Backend Abstraction Layer
- **Mode-Based Backend Selection**: Select Metal vs CUDA backend based on chosen mode
- **Hardware-Aware Configuration**: Configure backend based on actual hardware capabilities
- **Memory Management**: Environment-specific memory allocation strategies
- **Batch Size Optimization**: Automatic batch size adjustment based on mode and hardware
- **Error Handling**: Mode-specific error handling and recovery

### Configuration Management
- **Mode-Specific Configuration**: Separate configuration for development and production modes
- **Hardware-Aware Settings**: Adjust configuration based on detected hardware
- **Environment Variables**: Override configuration via environment variables
- **CLI Overrides**: Command-line options to override configuration
- **Configuration Validation**: Validate configuration compatibility with selected mode

### Data Scaling Logic
- **Mode-Based Scaling**: Scale data processing based on selected mode
- **Hardware-Aware Processing**: Adjust processing based on actual hardware capabilities
- **Memory-Aware Processing**: Scale based on available memory
- **Performance Optimization**: Optimize for detected hardware performance
- **Caching Strategy**: Mode-specific caching and storage strategies

### Unified CLI Interface
- **Mode Selection Commands**: CLI commands for mode selection and capability reporting
- **Command Consistency**: Identical core commands work in both modes
- **Mode Transparency**: Clear indication of current mode and capabilities
- **Status Reporting**: Report current mode, hardware capabilities, and configuration
- **Error Handling**: Mode-specific error messages and recovery

### Capability Reporting
- **Hardware Summary**: Clear summary of detected hardware capabilities
- **Mode Recommendations**: Specific recommendations for each detected hardware type
- **Performance Estimates**: Performance estimates for each mode
- **Requirement Analysis**: Analysis of hardware against mode requirements
- **Upgrade Suggestions**: Suggestions for hardware upgrades if needed

### Pipeline Organization
- **Package Structure**: All scripts organized in `src/eq/` with proper package structure
- **Clean Root Directory**: No Python files scattered around root directory
- **Pipeline Runner Location**: Main pipeline runner in obvious location (`src/eq/pipeline/run_pipeline.py`)
- **Module Imports**: Proper import structure for all pipeline components
- **Script Organization**: Logical grouping of related pipeline functionality

### Output Directory System
- **Data-Driven Naming**: Output directories named based on input data directory name
- **Run Type Organization**: Separate outputs for quick, full production, and smoke test runs
- **Timestamp Integration**: Include timestamps in output directory names for versioning
- **Structured Outputs**: Organized subdirectories for models, plots, results, and reports
- **Path Management**: Consistent path handling across different pipeline stages

### Enhanced Visualization Pipeline
- **Multi-Level Visualization**: Show progression through each pipeline stage
- **Input Visualization**: Raw medical images with contrast enhancement
- **Segmentation Visualization**: Predicted masks vs. ground truth with overlay
- **ROI Extraction Visualization**: Visual representation of region of interest extraction
- **Feature Computation Visualization**: Visualization of numerical feature extraction
- **Regression Visualization**: Training curves, predictions, and confidence intervals
- **Interactive Elements**: Notebook-style outputs with interactive visualizations

### Comprehensive Reporting
- **Pipeline Progression Tracking**: Clear indication of current pipeline stage
- **Stage-Specific Reports**: Detailed reports for each pipeline stage
- **Performance Metrics**: Training metrics, validation scores, and processing times
- **Quality Assessment**: Quality metrics for segmentation and feature extraction
- **Error Reporting**: Comprehensive error reporting with recovery suggestions
- **Summary Reports**: Executive summaries of pipeline execution and results

### ROI and Feature Visualization
- **ROI Extraction Display**: Visual representation of extracted regions of interest
- **Patch Visualization**: Display of image patches extracted from segmented regions
- **Feature Distribution**: Histograms and statistical plots of extracted features
- **Feature Correlation**: Correlation matrices and feature importance visualization
- **Quality Metrics**: Visualization of ROI quality and feature reliability

### Regression Model Visualization
- **Training Curves**: Loss and accuracy curves during model training
- **Validation Plots**: Cross-validation results and model performance
- **Prediction Visualization**: Predicted vs. actual endotheliosis scores
- **Confidence Intervals**: Visualization of prediction uncertainty and confidence
- **Feature Importance**: Feature importance plots and coefficient analysis
- **Model Comparison**: Comparison of different regression model performances

## External Dependencies

### Hardware Detection
- **psutil** - System information and hardware detection
- **torch** - Backend detection and MPS/CUDA availability checking
- **numpy** - Hardware capability assessment
- **platform** - Platform and architecture detection
- **GPUtil** - GPU detection and capability assessment

### Mode Management
- **click** - CLI mode selection and command management
- **pydantic** - Mode configuration validation
- **python-dotenv** - Environment variable management
- **configparser** - Configuration file management

### Backend Management
- **torch** - PyTorch core framework with MPS and CUDA support
- **fastai** - High-level API for seamless cross-platform training
- **torchvision** - Computer vision utilities for PyTorch
- **torchaudio** - Audio processing utilities for PyTorch

### Visualization and Reporting
- **matplotlib** - Core plotting and visualization library
- **seaborn** - Statistical data visualization
- **plotly** - Interactive visualizations and dashboards
- **PIL/Pillow** - Image processing and manipulation
- **opencv-python** - Computer vision and image processing
- **pandas** - Data manipulation and analysis for reporting
- **jupyter** - Interactive notebook support for visualization outputs

### Testing Frameworks
- **pytest** - Primary testing framework
- **pytest-mock** - Mocking for mode and hardware testing
- **pytest-cov** - Code coverage reporting
- **pytest-benchmark** - Performance benchmarking

### Justification
- **Hardware detection**: Essential for accurate mode suggestions and capability reporting
- **Mode management**: Critical for explicit user control over workflow selection
- **Backend management**: Required for seamless MPS vs CUDA switching using PyTorch
- **Visualization libraries**: Essential for comprehensive pipeline visualization and reporting
- **Testing frameworks**: Necessary for validating mode selection and hardware detection
- **fastai/PyTorch**: Leverages existing working implementations and provides superior cross-platform support
- **Pipeline organization**: Critical for maintainable and scalable codebase structure
