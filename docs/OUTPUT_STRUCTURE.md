# Output Structure

This document describes the recommended local directory layout for data, derived artifacts, trained models, and general outputs.

## Principles

- Keep source code and configuration in Git.
- Keep datasets, trained models, logs, and generated outputs local-only.
- Prefer consistent repo-relative paths across WSL, Windows, and macOS.

## Recommended Layout

```text
project_root/
├── configs/
├── data/
│   ├── raw_data/
│   │   ├── lucchi/
│   │   └── <project_name>/
│   │       ├── images/
│   │       ├── masks/
│   │       ├── annotations/
│   │       │   └── annotations.json
│   │       └── subject_metadata.xlsx
│   └── derived_data/
│       ├── mitochondria_data/
│       └── <project_name>/
├── logs/
├── models/
│   └── segmentation/
│       ├── mitochondria/
│       └── glomeruli/
├── output/
└── test_output/
```

## Purpose Of Each Directory

- `data/raw_data/`
  Original source datasets and annotations. For the current preeclampsia quantification baseline, this includes image/mask pairs plus a Label Studio JSON export. When segmentation training consumes paired image/mask files directly, the canonical curated trainable roots also live under `raw_data`, for example:

  ```text
  data/raw_data/preeclampsia_project/
  ├── clean_backup/
  ├── training_pairs/
  │   ├── images/
  │   └── masks/
  ├── annotations/
  └── subject_metadata.xlsx
  ```

- `data/derived_data/`
  Processed outputs such as extracted images, metadata exports, manifests, audits, caches, metrics, or cached intermediates. Do not use `derived_data` as the canonical location for curated glomeruli training pairs.

Current mitochondria training data uses:

  ```text
  data/derived_data/mitochondria_data/
  ├── training/
  │   ├── images/
  │   └── masks/
  └── testing/
      ├── images/
      └── masks/
  ```

  The `training/` root is used for dynamic training and its internal train/validation split. The `testing/` root is held out for explicit evaluation.

  This mitochondria layout is an installed Lucchi exception to the general rule above: curated glomeruli training pairs belong under `raw_data`, while generated artifacts belong under `derived_data`.

  Static patch datasets are retired runtime artifacts under the configured runtime `_retired/` directory, not active training inputs:

  ```text
  <runtime_root>/_retired/
  ├── glomeruli_static_patch_datasets_2026-04-22/
  └── mitochondria_static_patch_datasets_2026-04-22/
  ```

- `models/segmentation/mitochondria/`
  Trained mitochondria segmentation models and associated training artifacts.

- `models/segmentation/glomeruli/`
  Trained glomeruli segmentation models and associated training artifacts.

- `logs/`
  Run logs and temporary experiment logs.

- `output/`
  General-purpose generated outputs such as visualizations, quantification review reports, and one-off analysis exports.

For quantification runs, a typical output subtree now looks like:

```text
output/quantification/<project_name>/
├── labelstudio_scores/
├── scored_examples/
├── roi_crops/
├── embeddings/
└── ordinal_model/
    ├── ordinal_predictions.csv
    ├── ordinal_metrics.json
    ├── ordinal_confusion_matrix.csv
    └── review_report/
        ├── ordinal_review.html
        ├── selected_examples.csv
        └── assets/
```

- `test_output/`
  Temporary files created by tests or debugging scripts.

## Path Conventions In Code

Current code and documentation should prefer the following repo-relative paths:

- `data/raw_data`
- `data/derived_data`
- `models/segmentation`
- `logs`
- `output`

Avoid older machine-specific absolute paths and legacy directory names such as bare `derived_data/` at the repo root unless you are working on explicit backward-compatibility code.

## Environment Overrides

The project supports a small set of path overrides when needed:

- `EQ_DATA_PATH`
- `EQ_OUTPUT_PATH`
- `EQ_CACHE_PATH`
- `EQ_MODEL_PATH`

These should be the exception rather than the default.
