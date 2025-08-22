# Spec Requirements Document

> Spec: migrate-src-eq
> Created: 2025-08-21
> Status: In Progress (85% Complete)

## Overview

Migrate legacy scripts into a maintainable Python package under src/eq with clear module boundaries, stable imports, and tests, minimizing breakage for existing entry scripts.

## Progress Summary

**Completed (85%):**
- ✅ Package structure established (`src/eq/` with all submodules)
- ✅ Core utilities migrated (IO, patches, features, augmentation)
- ✅ Compatibility shims created and removed after verification
- ✅ Comprehensive test suite (8/8 tests passing)
- ✅ Data directory renamed (`Lauren_PreEclampsia_Data` → `preeclampsia_data`)
- ✅ All code references updated and verified
- ✅ ROI extraction confirmed working (T20: 26 ROIs)

**Remaining (15%):**
- [ ] Feature extraction and pipeline logic migration (Task 3)
- [ ] CLI wrappers and documentation alignment (Task 4)
- [ ] Cleanup, linters, and final documentation overhaul (Task 6)

## User Stories

### As a developer, I want reusable modules

As a developer, I want to import data, preprocessing, models, and pipeline utilities from eq.* so that scripts are thin wrappers and code reuse is consistent.

### As a maintainer, I want a clear layout

As a maintainer, I want a standardized package tree (eq/io, eq/augment, eq/models, etc.) so that new code lands in predictable places.

## Spec Scope

1. ✅ **Create src/eq package structure** - establish directories and __init__.py
2. ✅ **Consolidate utils into modules** - move functions from scripts/utils and scripts/main
3. ✅ **Add import shims** - keep existing scripts working by importing eq.* (shims removed after verification)
4. ✅ **Root tests** - add/adjust tests for moved modules
5. ✅ **Data directory rename** - rename all references from `Lauren_PreEclampsia_Data` to `preeclampsia_data` (case and path consistent)
6. [ ] **README overhaul** - refresh with current setup, professional structure, and clear project ethos (after major changes)

## Out of Scope

- Model architecture redesign
- Dataset changes or new annotations

## Expected Deliverable

1. ✅ Importable eq package with modules for IO, preprocessing, augmentation, models, segmentation, metrics, and pipeline.
2. ✅ Updated scripts that import from eq.*; tests pass and runtime_check succeeds.
3. ✅ All code/docs use `preeclampsia_data` for the data directory.
4. [ ] README professionally rewritten to reflect current setup and ethos.

## Testing Strategy

- Use pytest at root tests/
- Verify behavior on real small data slices
- Smoke test pipeline via existing runtime_check and smoke_test_pipeline scripts

## Data Locations

- Training data root: `data/preeclampsia_data/` ✅ **COMPLETED**
  - Images: `data/preeclampsia_data/train/images/`
  - Masks:  `data/preeclampsia_data/train/masks/`
  - ROIs:   `data/preeclampsia_data/train/rois/`
  - Cache:  `data/preeclampsia_data/cache/`

**Migration Status**: All code references updated from `Lauren_PreEclampsia_Data` to `preeclampsia_data`
- ✅ Directory renamed using `mv`
- ✅ All script references updated (runtime_check.py, smoke_test_pipeline.py, main scripts)
- ✅ .gitattributes updated for LFS tracking
- ✅ Tests pass with new data paths
- ✅ ROI extraction verified (T20: 26 ROIs produced)


