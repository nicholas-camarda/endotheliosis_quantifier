# Technical Specification

This is the technical specification for the spec detailed in @~/.agent-os/specs/2025-08-21-migrate-src-eq/2025-08-21-migrate-src-eq.md

## Technical Requirements
- Target tree under src/eq:
  - eq/io, eq/augment, eq/patches, eq/models, eq/segmentation, eq/features, eq/metrics, eq/pipeline, eq/utils
- Move candidates:
  - scripts/utils: convert_files_to_jpg.py, patchify_images.py, generate_augmented_dataset.py, preprocess_ROI_then_extract_features.py → eq/io, eq/patches, eq/augment, eq/features
  - scripts/main: feature_extractor_helper_functions.py, quantify_endotheliosis.py → eq/features, eq/pipeline
- Add __init__.py to all packages
- Keep CLI entry scripts; replace logic with imports from eq.*
- Tests under tests/: add minimal behavior checks for moved functions

## External Dependencies (Conditional)
- None beyond existing environment

