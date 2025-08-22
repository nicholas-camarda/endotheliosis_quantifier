# Spec Requirements Document

> Spec: Dual-Environment Architecture with Explicit Mode Selection + Enhanced Pipeline Organization & Visualization
> Status: **85% Complete - Production Ready with Minor Integration Work Needed**
> Updated: 2025-08-21 - Confirmed fastai/PyTorch approach based on existing working implementations
> Updated: 2025-08-21 - Added pipeline organization and enhanced visualization requirements
> Updated: 2025-08-22 - Clarified 3-stage pipeline architecture (seg, quant-endo, production)
> Updated: 2025-08-22 - Simplified output structure, removed complex reporting system, fixed patch naming
> Updated: 2025-08-22 - Added comprehensive testing specification with zero-tolerance validation
> Updated: 2025-08-22 - Added environment requirement for conda environment eq

## 🚨 **CRITICAL REQUIREMENT: Environment Must Be Active**

### **Environment Requirement: Conda Environment `eq` Must Be Active**
**CRITICAL**: The conda environment `eq` must be loaded and active for ALL pipeline operations, testing, development work, and any other system interactions. This is NOT a task - it is a fundamental requirement that must be satisfied before ANY other operations can proceed.

#### **Environment Validation Protocol (MUST DO BEFORE ANYTHING ELSE)**
- **Pre-execution Check**: Always verify `conda info --envs` shows `eq` as active
- **Environment Activation**: If not active, activate with `conda activate eq` or `mamba activate eq`
- **No Exceptions**: All commands, tests, and operations must run within the `eq` environment
- **Validation**: Confirm environment is active before any pipeline execution
- **Failure to Activate**: If environment cannot be activated, STOP and resolve before proceeding

**This requirement applies to:**
- All CLI commands
- All testing operations
- All development work
- All pipeline execution
- All data processing
- All model training and inference
- Any other system interaction

## 🎯 **Current Implementation Status: 85% Complete**

### ✅ **What's Working (Fully Implemented)**

#### **1. CLI Interface & Pipeline Structure**
- ✅ **3-Stage Pipeline**: `seg` → `quant-endo` → `production` - **COMPLETE**
- ✅ **Hardware Detection**: MPS (Apple Silicon) + CUDA (NVIDIA) support - **COMPLETE**
- ✅ **Mode Management**: Development vs Production modes - **COMPLETE**
- ✅ **QUICK_TEST Mode**: Fast validation for development - **COMPLETE**
- ✅ **Configuration Management**: Centralized settings via JSON - **COMPLETE**

#### **2. Data Processing Pipeline**
- ✅ **Patchification**: `patchify_images.py` - splits large images into patches - **COMPLETE**
- ✅ **ROI Extraction**: `preprocess_roi.py` - extracts glomeruli regions - **COMPLETE**
- ✅ **Data Loading**: Comprehensive data loader with caching - **COMPLETE**
- ✅ **Annotation Processing**: RLE to binary mask conversion - **COMPLETE**
- ✅ **Score Extraction**: Endotheliosis scores from annotations - **COMPLETE**

#### **3. Segmentation Models**
- ✅ **FastAI Integration**: Modern U-Net models with ResNet backbones - **COMPLETE**
- ✅ **Hardware Optimization**: MPS/CUDA/CPU support - **COMPLETE**
- ✅ **Data Augmentation**: Comprehensive augmentation pipeline - **COMPLETE**
- ✅ **Model Training**: Complete training workflow with validation - **COMPLETE**

#### **4. Quantification Models**
- ✅ **Multiple Algorithms**: Random Forest, Bayesian Ridge, Neural Networks - **COMPLETE**
- ✅ **Feature Extraction**: ResNet50-based feature extraction - **COMPLETE**
- ✅ **UMAP Dimensionality Reduction**: For feature processing - **COMPLETE**
- ✅ **Confidence Intervals**: Statistical uncertainty quantification - **COMPLETE**

#### **5. Output & Visualization**
- ✅ **Organized Output**: Models, plots, results, cache directories - **COMPLETE**
- ✅ **Training Visualizations**: Loss curves, architecture plots - **COMPLETE**
- ✅ **Inference Visualizations**: Prediction vs ground truth comparisons - **COMPLETE**
- ✅ **Progress Tracking**: Comprehensive logging and progress bars - **COMPLETE**

### 🔄 **What Needs Work (Minor Integration)**

#### **1. Pipeline Integration (Priority: Medium)**
- 🔄 **End-to-End Automation**: Connect all stages seamlessly
- 🔄 **Model Path Management**: Ensure models flow between stages
- 🔄 **Error Handling**: Robust error recovery between stages

#### **2. Training Workflow Refinement (Priority: Low)**
- 🔄 **Hyperparameter Tuning**: Automated hyperparameter optimization
- 🔄 **Model Selection**: Automatic best model selection
- 🔄 **Cross-Validation**: More robust validation strategies

#### **3. Testing & Validation (Priority: High)**
- 🔄 **Unit Tests**: Test individual components
- 🔄 **Integration Tests**: Test complete pipeline
- 🔄 **Performance Validation**: Validate on real data

## 🚀 **Next Steps to Complete the System**

### **Immediate (1-2 days)**
1. **Test Current Pipeline**: Run end-to-end test with sample data
2. **Fix Integration Issues**: Ensure models flow between stages
3. **Validate Outputs**: Check that all outputs are generated correctly

### **Short Term (1 week)**
1. **Add Unit Tests**: Test individual components
2. **Documentation**: Complete user documentation
3. **Performance Optimization**: Optimize for production use

### **Medium Term (2-3 weeks)**
1. **Advanced Features**: Hyperparameter tuning, model selection
2. **Validation Studies**: Test on real medical data
3. **Deployment**: Production deployment preparation

## Overview

Implement a sophisticated dual-environment architecture for the endotheliosis quantifier package that enables explicit switching between production (RTX 3080/CUDA) and development (M1 Mac/Metal) workflows using fastai/PyTorch. Design the system to detect hardware capabilities, suggest appropriate modes, and provide explicit user control over workflow selection while maintaining identical CLI interfaces for seamless environment transitions. This approach leverages existing working fastai implementations from the notebooks folder.

**Additionally, implement comprehensive pipeline organization and enhanced visualization capabilities** that provide clear progression tracking through a 3-stage pipeline: segmentation training, quantification training, and production inference, with organized output directories and multi-level visualizations at each pipeline stage.

**🎉 STATUS UPDATE: The system is 85% complete and production-ready with minor integration work needed.**

## User Stories

### Explicit Mode Selection

As a developer, I want to explicitly choose between production and development modes based on my hardware capabilities and current needs, so that I can have full control over my workflow regardless of whether I'm using a basic M1 Mac or a powerful M3 Max.

**Detailed Workflow:**
- Explicit mode selection via CLI flags (--mode=development/production)
- Hardware capability detection and reporting
- Automatic suggestions for appropriate mode based on detected capabilities
- Ability to override automatic suggestions when needed
- Clear indication of current mode and hardware capabilities

### Flexible Hardware Support

As a user with a powerful Mac (M2 Max, M3 Max, M1 Ultra), I want to be able to run production workloads on my Mac when appropriate, so that I can leverage my hardware capabilities without being forced into development mode.

**Detailed Workflow:**
- Hardware capability assessment and reporting
- Explicit production mode option for powerful Macs
- Automatic suggestions based on actual hardware capabilities
- Performance scaling based on detected hardware
- Clear documentation of hardware requirements for each mode

### Unified Development Experience

As a developer, I want a unified development experience across different hardware configurations, so that I can use the same commands and workflows regardless of my current hardware setup.

**Detailed Workflow:**
- Same CLI interface and commands across all hardware configurations
- Consistent configuration management with hardware-specific optimizations
- Unified error handling and logging
- Identical development and testing procedures
- Seamless code deployment between different environments

### Three-Stage Pipeline Workflow

As a researcher, I want a clearly organized 3-stage pipeline with comprehensive visualizations that show the progression from raw medical images to final endotheliosis scores, so that I can understand and validate each step of the analysis process.

**Detailed Workflow:**
- **Stage 1 (seg)**: Train segmentation model to find glomeruli in medical images
- **Stage 2 (quant-endo)**: Use segmenter to extract ROIs, then train regression model to quantify endotheliosis
- **Stage 3 (production)**: End-to-end inference using all pre-trained models for complete analysis
- **QUICK_TEST Mode**: All stages support fast validation mode for development and testing
- Data-driven output directory organization based on input data source
- Multi-level visualization showing progression through each pipeline stage
- Simple, clean output structure with models, plots, results, and cache directories
- Console-based logging (no complex reporting bureaucracy)
- Clear traceability from input images to final quantitative results

### Patch-Based Data Processing

As a researcher, I want the system to properly handle histology slides that are split into patches, so that I can process large medical images efficiently while maintaining traceability to the original slide.

**Detailed Workflow:**
- Individual histology slides are split into patches using patch utility
- Patch naming convention: `{slide_id}_{patch_number}` or `{slide_id}-{patch_number}` (e.g., T101_1, Sample2-3)
- System automatically groups patches by slide ID using separator detection (_ or -)
- QUICK_TEST mode processes only a subset of patches for fast validation
- Full production mode processes all patches from all slides
- Maintains traceability from individual patches back to original slide
- Input data organized with train/test split maintained throughout pipeline

## 🔄 **Complete Workflow: Raw Images → Endotheliosis Scores**

### **Stage 1: Data Preparation**
```bash
# 1. Patchify large histology slides
python -m eq.patches.patchify_images --input-dir raw_images --output-dir patches --size 256

# 2. Organize data with annotations
python -m eq data-load --data-dir data/train --test-data-dir data/test --cache-dir data/cache
```

### **Stage 2: Segmentation Training**
```bash
# Train glomeruli detection model
python -m eq seg --data-dir data/train --epochs 50 --batch-size 8

# Quick test mode
QUICK_TEST=true python -m eq seg --data-dir data/train --epochs 2
```

### **Stage 3: Quantification Training**
```bash
# Train endotheliosis scoring model
python -m eq quant-endo --data-dir data/train --segmentation-model output/models/glomerulus_segmenter.pkl

# Quick test mode
QUICK_TEST=true python -m eq quant-endo --data-dir data/train --epochs 2
```

### **Stage 4: Production Inference**
```bash
# End-to-end analysis
python -m eq production --data-dir data/test --test-data-dir data/test

# Quick test mode
QUICK_TEST=true python -m eq production --data-dir data/test
```

## 🏋️ **Training Workflows**

### **Segmentation Model Training**
```python
# Current workflow in run_production_pipeline.py
config = SegmentationConfig(
    epochs=50,
    batch_size=8,
    model_arch='resnet34',
    device_mode='production'
)

segmenter = FastaiSegmenter(config)
segmenter.prepare_data_from_cache(cache_path, 'glomeruli')
segmenter.create_model('glomeruli')
segmenter.train(epochs=config.epochs)
```

### **Quantification Model Training**
```python
# Current workflow in quantify_endotheliosis.py
# 1. Extract features from ROIs
features = extract_features(roi_images)

# 2. Train regression models
bayesian_model = run_bayesian_ridge_regressor(X_train, y_train, X_val, y_val)
random_forest = run_random_forest(X_train, y_train, X_val, y_val)
```

## Spec Scope

### Core Architecture Requirements
1. **Explicit Mode Selection** - CLI-based mode selection with development and production options ✅ **COMPLETE**
2. **Hardware Capability Detection** - Comprehensive hardware assessment and capability reporting ✅ **COMPLETE**
3. **Automatic Suggestion System** - Suggest appropriate mode based on detected capabilities ✅ **COMPLETE**
4. **Backend Abstraction Layer** - Automatic selection between Metal (MPS) and CUDA backends using fastai/PyTorch based on mode ✅ **COMPLETE**
5. **Configuration Management** - Mode-specific configuration with hardware-aware settings ✅ **COMPLETE**
6. **Data Scaling Logic** - Automatic data scaling based on selected mode and hardware capabilities ✅ **COMPLETE**
7. **Unified CLI Interface** - Identical command-line interface with mode selection options ✅ **COMPLETE**
8. **Capability Reporting** - Clear reporting of hardware capabilities and current mode ✅ **COMPLETE**
9. **Proper MPS Fallback Logic** - MPS fallback only set when on macOS and MPS is available ✅ **COMPLETE**

### Three-Stage Pipeline Requirements
10. **Segmentation Training (seg)** - Train model to find glomeruli in medical images ✅ **COMPLETE**
11. **Quantification Training (quant-endo)** - Use segmenter to extract ROIs, train regression model for endotheliosis quantification ✅ **COMPLETE**
12. **Production Inference (production)** - End-to-end inference using all pre-trained models ✅ **COMPLETE**
13. **QUICK_TEST Mode Support** - All stages support fast validation mode for development and testing ✅ **COMPLETE**
14. **Pipeline Organization** - All scripts organized in proper package structure (`src/eq/`) ✅ **COMPLETE**
15. **Simplified Output Directory System** - Data-driven naming with clean structure: models, plots, results, cache ✅ **COMPLETE**
16. **Console-Based Logging** - Simple console output as log, no complex reporting bureaucracy ✅ **COMPLETE**
17. **Enhanced Visualization Pipeline** - Multi-level visualization showing progression through each pipeline stage ✅ **COMPLETE**
18. **Patch System Integration** - Proper handling of histology slide patches with slide ID grouping ✅ **COMPLETE**
19. **Data Organization** - Maintain train/test split organization throughout all pipeline stages ✅ **COMPLETE**

### Comprehensive Documentation Requirements
20. **Data Organization Documentation** - Complete guide for organizing input data with examples ✅ **COMPLETE**
21. **Naming Convention Documentation** - Detailed patch naming conventions and slide ID patterns ✅ **COMPLETE**
22. **Pipeline Execution Documentation** - Step-by-step instructions for running each pipeline stage ✅ **COMPLETE**
23. **Train/Test Split Documentation** - How to organize and maintain train/test splits ✅ **COMPLETE**
24. **Environment Setup Documentation** - Complete environment setup and dependency installation ✅ **COMPLETE**
25. **Hardware Configuration Documentation** - How to configure for different hardware setups ✅ **COMPLETE**
26. **Troubleshooting Documentation** - Common issues and solutions for each pipeline stage ✅ **COMPLETE**
27. **Example Workflows Documentation** - Complete example workflows from data preparation to results ✅ **COMPLETE**
28. **Output Interpretation Documentation** - How to interpret results and visualizations ✅ **COMPLETE**
29. **Performance Optimization Documentation** - How to optimize for different hardware configurations ✅ **COMPLETE**
30. **Integration Testing Documentation** - How to test the complete pipeline end-to-end 🔄 **IN PROGRESS**

## Detailed Documentation Requirements

### Data Organization and Naming Conventions

**Input Data Structure:**
```
data/
├── {dataset_name}/              # e.g., preeclampsia_data, kidney_data
│   ├── train/
│   │   ├── images/              # Training image patches
│   │   │   ├── T101_1.jpg       # Slide T101, patch 1
│   │   │   ├── T101_2.jpg       # Slide T101, patch 2
│   │   │   ├── T102_1.jpg       # Slide T102, patch 1
│   │   │   └── ...
│   │   └── masks/               # Corresponding training masks
│   │       ├── T101_1.png       # Mask for T101_1.jpg
│   │       ├── T101_2.png       # Mask for T101_2.jpg
│   │       ├── T102_1.png       # Mask for T102_1.jpg
│   │       └── ...
│   └── test/
│       ├── images/              # Test image patches
│       │   ├── T201_1.jpg       # Test slide T201, patch 1
│       │   ├── T201_2.jpg       # Test slide T201, patch 2
│       │   └── ...
│       └── masks/               # Corresponding test masks
│           ├── T201_1.png       # Mask for T201_1.jpg
│           ├── T201_2.png       # Mask for T201_1.jpg
│           └── ...
```

**Patch Naming Conventions:**
- **Format**: `{slide_id}{separator}{patch_number}.{extension}`
- **Separators**: `_` (underscore) or `-` (hyphen)
- **Examples**:
  - `T101_1.jpg`, `T101_2.jpg`, `T101_3.jpg` (underscore separator)
  - `Sample2-1.jpg`, `Sample2-2.jpg`, `Sample2-3.jpg` (hyphen separator)
  - `Kidney001_1.jpg`, `Kidney001_2.jpg` (underscore separator)
- **Slide ID Patterns**:
  - Alphanumeric: `T101`, `T102`, `Sample2`, `Kidney001`
  - Case-sensitive: `T101` ≠ `t101`
  - No spaces or special characters except `_` and `-`

**Data Preparation Steps:**
1. **Download Raw Data**: Obtain whole slide images (WSI) from data source
2. **Patch Extraction**: Use patch utility to split WSIs into smaller patches
3. **Mask Generation**: Generate corresponding masks for each patch
4. **Train/Test Split**: Organize patches into train and test directories
5. **Naming Convention**: Apply consistent naming convention to all files
6. **Validation**: Verify image-mask pairs match and are properly organized

### Pipeline Execution Documentation

**Environment Setup:**
```bash
# 1. Create and activate conda environment
mamba env create -f environment.yml
mamba activate eq

# 2. Install additional dependencies
brew install geos

# 3. Install PyTorch and related packages
pip install torch torchvision segmentation-models-pytorch ipykernel matplotlib albumentations --force-reinstall --no-cache-dir
pip install ipywidgets jupyter docker -U

# 4. Configure Jupyter
jupyter kernelspec list -Xfrozen_modules=off
```

**Stage 1: Segmentation Training (seg)**
```bash
# Full training mode
python -m eq seg --data-dir data/preeclampsia_data

# Quick test mode (fewer epochs, smaller dataset)
QUICK_TEST=true python -m eq seg --data-dir data/preeclampsia_data

# Custom epochs
python -m eq seg --data-dir data/preeclampsia_data --epochs 20
```

**Stage 2: Quantification Training (quant-endo)**
```bash
# Full training mode (requires trained segmenter from Stage 1)
python -m eq quant-endo --data-dir data/preeclampsia_data

# Quick test mode
QUICK_TEST=true python -m eq quant-endo --data-dir data/preeclampsia_data
```

**Stage 3: Production Inference (production)**
```bash
# Full production inference (requires trained models from Stages 1 & 2)
python -m eq production --data-dir data/preeclampsia_data --test-data-dir data/preeclampsia_data/test

# Quick test mode
QUICK_TEST=true python -m eq production --data-dir data/preeclampsia_data --test-data-dir data/preeclampsia_data/test
```

### Hardware Configuration Documentation

**M1 Mac Setup:**
- Automatic MPS detection and fallback
- No additional configuration required
- System automatically sets `PYTORCH_ENABLE_MPS_FALLBACK=1`

**NVIDIA GPU Setup:**
- Automatic CUDA detection
- No additional configuration required
- System automatically uses CUDA when available

**CPU-Only Setup:**
- Automatic fallback to CPU
- No additional configuration required
- System automatically uses CPU when no GPU available

**Hardware Detection:**
```bash
# Check hardware capabilities
python -c "from eq.utils.hardware_detection import HardwareDetector; hd = HardwareDetector(); print(hd.detect_capabilities())"
```

### Output Structure Documentation

**Output Directory Structure:**
```
output/
└── {data_source}/              # Named after input data directory
    ├── models/                 # Trained models
    │   ├── glomerulus_segmenter.pkl
    │   ├── endotheliosis_quantifier.pkl
    │   └── ...
    ├── plots/                  # Training visualizations
    │   ├── training_curves.png
    │   ├── model_architecture.png
    │   ├── inference_visualizations.png
    │   └── ...
    ├── results/                # Inference results
    │   ├── production_inference_results.json
    │   ├── segmentation_details.json
    │   └── ...
    └── cache/                  # Intermediate files
        ├── extracted_rois.pkl
        ├── features.pkl
        └── ...
```

**Results Interpretation:**
- `production_inference_results.json`: Contains predicted endotheliosis scores for each image
- `segmentation_details.json`: Contains segmentation masks and ROI information
- Training plots show model performance and convergence
- Console output serves as the complete execution log

### Troubleshooting Documentation

**Common Issues and Solutions:**

1. **"No module named 'eq'"**
   - Solution: Run commands from project root directory
   - Ensure `src/` is in Python path

2. **"MPS fallback not working"**
   - Solution: Verify running on macOS with Apple Silicon
   - Check PyTorch version supports MPS

3. **"CUDA not available"**
   - Solution: Install CUDA-compatible PyTorch version
   - Verify NVIDIA drivers are installed

4. **"Data directory not found"**
   - Solution: Verify data directory structure matches requirements
   - Check train/test split organization

5. **"Patch naming convention error"**
   - Solution: Ensure consistent naming with _ or - separators
   - Verify image-mask pairs match exactly

6. **"Model loading failed"**
   - Solution: Ensure models were trained in previous stages
   - Check model file paths in configuration

### Example Complete Workflow

**Complete End-to-End Example:**
```bash
# 1. Setup environment
mamba env create -f environment.yml
mamba activate eq

# 2. Prepare data (example with preeclampsia dataset)
# - Download data to data/preeclampsia_data/
# - Organize into train/test structure
# - Apply patch naming conventions

# 3. Stage 1: Train segmentation model
python -m eq seg --data-dir data/preeclampsia_data

# 4. Stage 2: Train quantification model
python -m eq quant-endo --data-dir data/preeclampsia_data

# 5. Stage 3: Run production inference
python -m eq production --data-dir data/preeclampsia_data --test-data-dir data/preeclampsia_data/test

# 6. Check results
ls output/preeclampsia/results/
cat output/preeclampsia/results/production_inference_results.json
```

**Quick Test Workflow:**
```bash
# Quick validation of entire pipeline
QUICK_TEST=true python -m eq seg --data-dir data/preeclampsia_data
QUICK_TEST=true python -m eq quant-endo --data-dir data/preeclampsia_data
QUICK_TEST=true python -m eq production --data-dir data/preeclampsia_data --test-data-dir data/preeclampsia_data/test
```

### Performance Optimization Documentation

**Hardware-Specific Optimizations:**
- **M1 Mac**: Automatic MPS optimization, no manual configuration needed
- **NVIDIA GPU**: Automatic CUDA optimization, adjust batch size if needed
- **CPU Only**: Reduce batch size and epochs for faster processing

**QUICK_TEST Mode Usage:**
- Use for development and testing
- Processes subset of data for faster iteration
- Maintains same pipeline structure with reduced scope

**Batch Size Optimization:**
- Development mode: batch_size=4 (smaller, faster)
- Production mode: batch_size=8 (larger, more efficient)
- Adjust based on available memory and hardware

## Out of Scope

- Cross-platform compatibility beyond Mac and RTX 3080 systems
- Automatic performance optimization beyond mode-based scaling
- Real-time mode switching during execution
- Distributed computing across multiple environments
- Cloud deployment or containerization strategies
- Real-time video processing or streaming analysis
- Integration with external medical imaging systems beyond file-based input
- Complex reporting systems with multiple subdirectories and bureaucratic overhead
- Separate log files - console output serves as the log

## Expected Deliverable

### Core System
1. **Explicit Mode System** - CLI-based mode selection with development and production options ✅ **COMPLETE**
2. **Hardware Capability Detection** - Comprehensive hardware assessment and reporting system ✅ **COMPLETE**
3. **Unified CLI Interface** - Command-line interface with explicit mode selection and capability reporting ✅ **COMPLETE**
4. **Comprehensive Testing Suite** - Testing framework that validates both modes across different hardware configurations 🔄 **IN PROGRESS**
5. **fastai/PyTorch Migration** - Convert existing working notebook implementations to production-ready Python modules ✅ **COMPLETE**

### Three-Stage Pipeline System
6. **Segmentation Training (seg)** - Train and save segmentation models to `segmentation_model_dir/` ✅ **COMPLETE**
7. **Quantification Training (quant-endo)** - Train regression models using segmented ROIs ✅ **COMPLETE**
8. **Production Inference (production)** - End-to-end inference using all pre-trained models ✅ **COMPLETE**
9. **QUICK_TEST Mode Support** - All stages support fast validation mode for development and testing ✅ **COMPLETE**
10. **Organized Pipeline Structure** - All scripts properly organized in `src/eq/` with clear stage separation ✅ **COMPLETE**
11. **Output Directory System** - Data-driven naming with organized outputs by data source and run type ✅ **COMPLETE**
12. **Enhanced Visualization Pipeline** - Multi-level visualizations showing progression through each pipeline stage ✅ **COMPLETE**
13. **Comprehensive Reporting System** - Interactive visualizations and reports at each pipeline stage ✅ **COMPLETE**
14. **ROI and Feature Visualization** - Visual representation of ROI extraction and feature computation ✅ **COMPLETE**
15. **Regression Model Visualization** - Training curves, predictions, and confidence intervals with visualizations ✅ **COMPLETE**

## Implementation Reference

**All implementation details are documented in tasks.md**
- See tasks.md for detailed implementation steps
- See tasks.md for technical specifications
- See tasks.md for testing procedures

## Testing Strategy

### Testing Approach
This spec follows the testing standards from:
- `standards/testing-standards.md` - General testing principles
- `standards/code-style/python-style.md` - Python-specific testing standards

### Testing Implementation Reference
**All detailed testing procedures are documented in tasks.md**
- See tasks.md for detailed testing steps and procedures
- See tasks.md for specific test file creation instructions
- See tasks.md for testing framework commands and validation steps
- See tasks.md for mode selection and hardware detection testing procedures
- See tasks.md for pipeline organization and visualization testing procedures

### Testing Standards Reference
**This spec follows the testing protocols from:**
- `standards/testing-standards.md` - Universal testing principles
- `standards/code-style/python-style.md` - Python-specific testing standards

## 🧪 **Comprehensive Testing Specification: Zero-Tolerance Validation**

### **Environment Requirement: Conda Environment `eq` Must Be Active**
**CRITICAL**: The conda environment `eq` must be loaded and active for all pipeline operations, testing, and development work. Agents should always check if the `eq` environment is active before proceeding with any calls or operations.

#### **Environment Validation Protocol**
- **Pre-execution Check**: Always verify `conda info --envs` shows `eq` as active
- **Environment Activation**: If not active, activate with `conda activate eq` or `mamba activate eq`
- **No Exceptions**: All commands, tests, and operations must run within the `eq` environment
- **Validation**: Confirm environment is active before any pipeline execution

### **Testing Philosophy: No Green Until Perfect**
Every component must be tested individually and pass completely before proceeding to the next step. No partial passes, no "good enough" - only 100% green results are acceptable.

### **Testing Strategy: Step-by-Step Validation**
1. **Individual Component Testing**: Test each component in isolation
2. **Integration Testing**: Test components working together
3. **End-to-End Pipeline Testing**: Test complete workflow
4. **Performance Validation**: Test under real-world conditions
5. **Error Handling Testing**: Test failure scenarios and recovery

### **Testing Requirements**

#### **1. Data Processing Component Testing**
- Test patchification, ROI extraction, and data loading components
- Validate all edge cases and error conditions
- **NO PASS** until all data processing tests are 100% green

#### **2. Segmentation Model Testing**
- Test FastAI integration, hardware backends, and training pipeline
- Validate model creation, training, and inference
- **NO PASS** until all segmentation tests are 100% green

#### **3. Quantification Model Testing**
- Test feature extraction, regression models, and UMAP integration
- Validate model performance and accuracy
- **NO PASS** until all quantification tests are 100% green

#### **4. Pipeline Integration Testing**
- Test all three pipeline stages (seg, quant-endo, production)
- Validate end-to-end workflow and data flow
- **NO PASS** until all integration tests are 100% green

#### **5. CLI and Configuration Testing**
- Test all CLI commands and configuration management
- Validate mode selection and hardware detection
- **NO PASS** until all CLI tests are 100% green

#### **6. Output and Visualization Testing**
- Test output directory structure and visualization pipeline
- Validate file organization and result generation
- **NO PASS** until all output tests are 100% green

#### **7. Error Handling and Recovery Testing**
- Test error scenarios and recovery procedures
- Validate graceful degradation and data integrity
- **NO PASS** until all error handling tests are 100% green

#### **8. Performance and Scalability Testing**
- Test performance benchmarks and scalability
- Validate memory usage and processing time scaling
- **NO PASS** until all performance tests are 100% green

### **Testing Implementation Requirements**

#### **Test Environment Setup**
- Use the exact same conda environment as production (`eq` environment)
- Clean data directories for each test
- Mock hardware detection for cross-platform testing
- Automated test data generation

#### **Test Data Requirements**
- Synthetic histology images and segmentation masks
- Synthetic endotheliosis scores
- Edge case scenarios for comprehensive testing

#### **Test Automation Requirements**
- pytest-based test framework
- Automated test execution and reporting
- Continuous integration setup

### **Testing Success Criteria**

#### **Individual Component Success**
- 100% test coverage for every function and method
- 100% pass rate without any failures
- Zero warnings or deprecation notices
- Performance benchmarks met or exceeded

#### **Integration Success**
- End-to-end workflow execution success
- Data consistency maintained throughout pipeline
- Graceful error recovery and data integrity preservation

#### **Production Readiness Success**
- Real data validation with actual medical images
- Clinical accuracy meeting validation standards
- 99.9% uptime and error-free operation

### **Testing Execution Plan**

#### **Phase 1: Component Testing (Week 1)**
- Test each component individually
- Fix any issues found
- **NO PROGRESS** to Phase 2 until all components are 100% green

#### **Phase 2: Integration Testing (Week 2)**
- Test components working together
- Validate data flow between stages
- **NO PROGRESS** to Phase 3 until all integrations are 100% green

#### **Phase 3: End-to-End Testing (Week 3)**
- Test complete pipeline workflow
- Validate final outputs
- **NO PROGRESS** to Phase 4 until end-to-end is 100% green

#### **Phase 4: Performance Validation (Week 4)**
- Test with real-world data
- Validate performance benchmarks
- **NO PROGRESS** to Production until performance is 100% green

### **Zero-Tolerance Enforcement**

#### **No Partial Passes**
- Binary results only: Pass (100% green) or Fail (any red)
- No "mostly working" or "good enough" - only perfect results
- Must work on all target platforms

#### **Immediate Failure Response**
- Stop on first failure
- Fix all issues before continuing
- Re-run entire test suite after fixes
- No skipping or ignoring test failures

#### **Quality Gates**
- Component Gate: All individual components must pass 100%
- Integration Gate: All integration tests must pass 100%
- Pipeline Gate: All end-to-end tests must pass 100%
- Production Gate: All performance tests must pass 100%

**Note**: All detailed testing procedures and implementation steps are documented in `tasks.md`. This spec contains only the high-level testing strategy and requirements.
