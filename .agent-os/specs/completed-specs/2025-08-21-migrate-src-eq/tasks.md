# Spec Tasks

## Tasks

- [x] 1. Establish `src/eq` package structure and imports
  - [x] 1.1 Write tests: verify package discovery/imports for `eq` and subpackages
  - [x] 1.2 Create directories and `__init__.py`: `eq/io`, `eq/augment`, `eq/patches`, `eq/models`, `eq/segmentation`, `eq/features`, `eq/metrics`, `eq/pipeline`, `eq/utils`
  - [x] 1.3 Ensure `pyproject.toml` discovers `eq*` packages; add missing `__init__.py` if any
  - [x] 1.4 Verify: run tests for task 1 and confirm imports succeed

- [x] 2. Migrate IO and preprocessing utilities (non-destructive with `mv` + shims)
  - [x] 2.1 Write tests: explicit assertions for `patchify_images`, `convert_files_to_jpg`, `preprocess_roi`
  - [x] 2.2 Move files with `mv`:
        `scripts/utils/convert_files_to_jpg.py` → `src/eq/io/convert_files_to_jpg.py`
        `scripts/utils/patchify_images.py` → `src/eq/patches/patchify_images.py`
        `scripts/utils/generate_augmented_dataset.py` → `src/eq/augment/augment_dataset.py`
        `scripts/utils/preprocess_ROI_then_extract_features.py` → `src/eq/features/preprocess_roi.py`
  - [x] 2.3 Add compatibility shims in original locations importing from `eq.*` to avoid breaking existing calls
  - [x] 2.4 Verify: run tests for task 2 and confirm behavior on small real data slices
  - [x] 2.5 Remove shim scripts and fully adopt new `eq` package structure

- [x] 3. Migrate feature extraction and pipeline logic
  - [x] 3.1 Write tests: explicit assertions for `feature_extractor_helper_functions` and `quantify_endotheliosis` key outputs on small inputs
  - [x] 3.2 Move files with `mv`:
        `scripts/main/feature_extractor_helper_functions.py` → `src/eq/features/helpers.py`
        `scripts/main/4_quantify_endotheliosis.py` → `src/eq/pipeline/quantify_endotheliosis.py`
  - [x] 3.3 Update `scripts/main/*.py` to import from `eq.*` while keeping CLI params unchanged
  - [x] 3.4 Verify: run tests, then `mamba run -n eq python scripts/utils/smoke_test_pipeline.py`

- [x] 4. CLI wrappers and documentation alignment
  - [x] 4.1 Write tests: wrappers import-calls `eq.*` functions and return expected values
  - [x] 4.2 Create/adjust thin wrappers in `scripts/main` and `scripts/utils` to call into `eq.*` (temporary - should be removed after migration)
  - [x] 4.3 Update `README.md` examples to prefer `python -m eq.pipeline.quantify_endotheliosis` where appropriate
  - [x] 4.4 Verify: `scripts/utils/runtime_check.py` and wrappers run successfully

- [x] 5. Data directory rename consolidation
  - [x] 5.1 Write tests: grep/assert no references to `Lauren_PreEclampsia_Data`; assert new `preeclampsia_data` references in code/docs
  - [x] 5.2 Update code/scripts/tests/docs to use `preeclampsia_data`
  - [x] 5.3 Verify: runtime_check.py path checks pass with new directory name

- [x] 6. Cleanup, linters, and documentation overhaul
  - [x] 6.1 Write tests for docs: examples reference `eq.*`, no stale paths
  - [x] 6.2 Remove temporary CLI wrappers (quantify_endotheliosis.py) - encourage direct eq package usage
  - [x] 6.3 Move `scripts/main/unused/` to an archive folder; ensure no imports break
  - [x] 6.4 Add `eq/utils/logging.py` and `eq/utils/paths.py` if needed; integrate minimal logging and path handling
  - [x] 6.5 Overhaul README: professional structure, current setup, refine project ethos (minimize emojis)
  - [x] 6.6 Verify: run full tests; ruff lint; smoke test pipeline


