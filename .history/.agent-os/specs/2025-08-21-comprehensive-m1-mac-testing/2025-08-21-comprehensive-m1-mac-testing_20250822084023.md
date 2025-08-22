# Spec Requirements Document

> Spec: Dual-Environment Architecture with Explicit Mode Selection + Enhanced Pipeline Organization & Visualization
> Status: In Progress
> Updated: 2025-08-21 - Confirmed fastai/PyTorch approach based on existing working implementations
> Updated: 2025-08-21 - Added pipeline organization and enhanced visualization requirements
> Updated: 2025-08-22 - Clarified 3-stage pipeline architecture (seg, quant-endo, production)

## Overview

Implement a sophisticated dual-environment architecture for the endotheliosis quantifier package that enables explicit switching between production (RTX 3080/CUDA) and development (M1 Mac/Metal) workflows using fastai/PyTorch. Design the system to detect hardware capabilities, suggest appropriate modes, and provide explicit user control over workflow selection while maintaining identical CLI interfaces for seamless environment transitions. This approach leverages existing working fastai implementations from the notebooks folder.

**Additionally, implement comprehensive pipeline organization and enhanced visualization capabilities** that provide clear progression tracking through a 3-stage pipeline: segmentation training, quantification training, and production inference, with organized output directories and multi-level visualizations at each pipeline stage.

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
- Comprehensive reporting with interactive visualizations at each stage
- Clear traceability from input images to final quantitative results

## Spec Scope

### Core Architecture Requirements
1. **Explicit Mode Selection** - CLI-based mode selection with development and production options
2. **Hardware Capability Detection** - Comprehensive hardware assessment and capability reporting
3. **Automatic Suggestion System** - Suggest appropriate mode based on detected capabilities
4. **Backend Abstraction Layer** - Automatic selection between Metal (MPS) and CUDA backends using fastai/PyTorch based on mode
5. **Configuration Management** - Mode-specific configuration with hardware-aware settings
6. **Data Scaling Logic** - Automatic data scaling based on selected mode and hardware capabilities
7. **Unified CLI Interface** - Identical command-line interface with mode selection options
8. **Capability Reporting** - Clear reporting of hardware capabilities and current mode

### Three-Stage Pipeline Requirements
9. **Segmentation Training (seg)** - Train model to find glomeruli in medical images
10. **Quantification Training (quant-endo)** - Use segmenter to extract ROIs, train regression model for endotheliosis quantification
11. **Production Inference (production)** - End-to-end inference using all pre-trained models
12. **QUICK_TEST Mode Support** - All stages support fast validation mode for development and testing
13. **Pipeline Organization** - All scripts organized in proper package structure (`src/eq/`)
14. **Output Directory System** - Data-driven naming with organized outputs by data source and run type
15. **Enhanced Visualization Pipeline** - Multi-level visualization showing progression through each pipeline stage
16. **Comprehensive Reporting** - Interactive visualizations and reports at each pipeline stage
17. **ROI Extraction Visualization** - Visual representation of region of interest extraction and processing
18. **Feature Computation Visualization** - Visualization of numerical feature extraction from ROIs
19. **Regression Model Visualization** - Training curves, predictions, and confidence intervals with visualizations

## Out of Scope

- Cross-platform compatibility beyond Mac and RTX 3080 systems
- Automatic performance optimization beyond mode-based scaling
- Real-time mode switching during execution
- Distributed computing across multiple environments
- Cloud deployment or containerization strategies
- Real-time video processing or streaming analysis
- Integration with external medical imaging systems beyond file-based input

## Expected Deliverable

### Core System
1. **Explicit Mode System** - CLI-based mode selection with development and production options
2. **Hardware Capability Detection** - Comprehensive hardware assessment and reporting system
3. **Unified CLI Interface** - Command-line interface with explicit mode selection and capability reporting
4. **Comprehensive Testing Suite** - Testing framework that validates both modes across different hardware configurations
5. **fastai/PyTorch Migration** - Convert existing working notebook implementations to production-ready Python modules

### Three-Stage Pipeline System
6. **Segmentation Training (seg)** - Train and save segmentation models to `segmentation_model_dir/`
7. **Quantification Training (quant-endo)** - Train regression models using segmented ROIs
8. **Production Inference (production)** - End-to-end inference using all pre-trained models
9. **QUICK_TEST Mode Support** - All stages support fast validation mode for development and testing
10. **Organized Pipeline Structure** - All scripts properly organized in `src/eq/` with clear stage separation
11. **Output Directory System** - Data-driven naming with organized outputs by data source and run type
12. **Enhanced Visualization Pipeline** - Multi-level visualizations showing progression through each pipeline stage
13. **Comprehensive Reporting System** - Interactive visualizations and reports at each pipeline stage
14. **ROI and Feature Visualization** - Visual representation of ROI extraction and feature computation
15. **Regression Model Visualization** - Training curves, predictions, and confidence intervals with visualizations

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
