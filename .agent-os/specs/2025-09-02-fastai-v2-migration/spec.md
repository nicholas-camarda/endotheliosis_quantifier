# Spec Requirements Document

> Spec: FastAI v2 Migration
> Created: 2025-09-02

## Overview

Migrate the entire Endotheliosis Quantifier codebase from FastAI v1 to v2 to enable modern deep learning capabilities, fix compatibility issues, and establish a foundation for future development. This migration addresses critical breaking changes in the FastAI API and ensures all training, inference, and data pipeline components work with current package versions.

Status update (2025-09-03):
- Data processing prerequisite validated on real data; patching generates derived_data correctly
- New CLI command `eq process-data` creates ready-to-analyze `derived_data/` with auto mask detection
- Global logging deduplicated (no duplicate logs regardless of verbosity)
- **END-TO-END PIPELINE VALIDATED**: Complete pipeline from raw data → derived data → mitochondria training → glomeruli training works successfully
- **TRANSFER LEARNING CHALLENGE IDENTIFIED**: FastAI v2 model loading requires custom function namespace setup for transfer learning between models

Note: always run `mamba activate eq`
Note: Mito raw data is always in `raw_data/mnt/coxfs01/vcg_connectomics/mitochondria/Lucchi/`


## User Stories

### Medical Researcher Workflow

As a medical researcher, I want to run the complete endotheliosis quantification pipeline without encountering FastAI v1 compatibility errors, so that I can focus on analyzing results rather than debugging technical issues.

**Detailed Workflow**: Download the package, point it at histology image data, run the complete pipeline from segmentation to quantification, and receive reproducible results with transparent parameters.

### Developer Maintenance

As a developer maintaining the codebase, I want all FastAI imports and API calls to use the current v2 syntax, so that I can leverage modern features, maintain compatibility with current dependencies, and avoid deprecated code patterns.

**Detailed Workflow**: Update all training modules, data loading pipelines, and inference engines to use FastAI v2 APIs, test each component individually, and validate the complete end-to-end pipeline.

## Spec Scope

1. **Data Processing Validation (PREREQUISITE)** - Validate existing image loading, preprocessing, and patching components work correctly with real data
2. **Data Pipeline Migration** - Convert all data loading from SegmentationDataLoaders to DataBlock approach
3. **Training Module Updates** - Update mitochondria and glomeruli training scripts with FastAI v2 APIs
4. **Inference Pipeline Updates** - Modernize prediction and inference code for v2 compatibility
5. **Import Statement Updates** - Fix all module relocations and function name changes
6. **Callback Compatibility** - Update any custom callbacks to use new event naming conventions
7. **End-to-End Testing** - Validate complete pipeline functionality with real medical data

## Out of Scope

- Backward compatibility with FastAI v1 saved models
- Performance optimization beyond basic functionality
- New feature development unrelated to v2 compatibility
- Documentation updates beyond technical migration notes
- User interface improvements beyond fixing broken functionality

## Expected Deliverable

1. **Complete FastAI v2 Migration** - All training scripts run without v1 errors and produce valid models
2. **Working Data Pipeline** - Data loading, preprocessing, and augmentation work with v2 APIs
3. **Functional Training Pipeline** - End-to-end training from data to saved models completes successfully
4. **Validated Inference Pipeline** - Prediction and quantification produce correct results on test data
