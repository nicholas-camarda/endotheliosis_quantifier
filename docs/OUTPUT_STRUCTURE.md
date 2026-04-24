# Output Structure

This document describes the recommended local directory layout for data, derived artifacts, trained models, and general outputs.

## Principles

- Keep source code and configuration in Git.
- Keep datasets, trained models, logs, and generated outputs local-only.
- Use the configured runtime root for heavy data and generated artifacts. This checkout's local default is recorded in `analysis_registry.yaml`.
- Keep repo-local `data/` directories as portable placeholders unless `EQ_RUNTIME_ROOT` points the workflow elsewhere.

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
  ├── data/
  │   ├── images/
  │   └── masks/
  ├── annotations/
  └── subject_metadata.xlsx
  ```

- `data/derived_data/`
  Processed outputs such as extracted images, metadata exports, manifests, audits, caches, metrics, or cached intermediates. Do not use `derived_data` as the canonical location for curated glomeruli training inputs.

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

## Runtime Scored Cohort Layout

Scored quantification cohorts use the active runtime root, not the repo checkout's placeholder `data/` tree:

```text
$EQ_RUNTIME_ROOT/
├── raw_data/
│   └── cohorts/
│       ├── manifest.csv
│       ├── metadata/
│       │   └── manifest_summary.json
│       ├── masked_core/
│       ├── vegfri_dox/
│       │   ├── images/
│       │   ├── masks/
│       │   └── metadata/
│       └── vegfri_mr/
└── output/
    └── cohorts/
        └── <cohort_id>/
            ├── transport_audit/
            ├── predicted_roi/
            ├── embeddings/
            ├── grader_outputs/
            └── review/
```

`raw_data/cohorts/manifest.csv` is the dataset-wide table. It records runtime-local paths, cohort identity, score linkage, lane assignment, hashes, verification state, and admission state. It does not carry original PhD or cloud source paths.

The supported admission states are explicit:

- `unresolved`: linkage or required evidence is still incomplete after the recorded discovery surfaces.
- `pending_discovery`: a row is recoverable in principle, but the declared cohort discovery surfaces are not exhausted yet.
- `excluded`: a contradiction, unreadable asset, ambiguous mapping, or failed gate blocks the row.
- `foreign`: the row came from a mixed export and does not belong to the intended cohort.
- `admitted`: mapping verification passed and the row is usable for the lane stated in `lane_assignment`.
- `pending_mask_quality`: a recovered masked external row needs mask-quality review before segmentation improvement.
- `pending_transport_audit`: a scored-only row needs cohort-specific segmentation transport review before predicted-ROI grading use.
- `evaluation_only`: MR phase 1 rows are available for transport/concordance evaluation, not training-set expansion.

Current runtime cohort counts are recorded in `raw_data/cohorts/metadata/manifest_summary.json`:

- `masked_core`: 88 `manual_mask` rows, all `admitted`.
- `vegfri_dox`: 864 rows total, with 619 decoded `masked_external` rows admitted after mask-quality review, 7 decoded rows missing scores, 228 foreign mixed-export rows, and 10 scored-only rows without decoded runtime images.
- `vegfri_mr`: 127 `mr_concordance_only` rows, with 126 localized whole-field TIFF rows marked `evaluation_only` and one unresolved workbook row, `8570-5`, without a matching discovered TIFF.

The current Dox runtime surface contains copied images, copied recovered brushlabel masks, copied score exports, `metadata/decoded_brushlabel_masks.csv`, and `metadata/mask_quality_audit.csv` under `raw_data/cohorts/vegfri_dox/`. Verified Dox masked-external rows are eligible for segmentation augmentation through the manifest-backed `raw_data/cohorts` training root.

MR is handled as a whole-field TIFF cohort. Manifest rows are image-level, workbook replicates are reduced to a human image-level median, raw replicate vectors are preserved in sidecar ingest artifacts, external-drive source provenance is recorded under `raw_data/cohorts/vegfri_mr/metadata/`, and phase 1 output is concordance/evaluation under `output/cohorts/vegfri_mr/`.

MR phase 1 inference has an explicit contract: whole-field TIFF tiling, glomerulus segmentation, component-area filtering, accepted ROI extraction, ROI grading, image-level median aggregation, and human-versus-inferred concordance. Rows with zero accepted inferred ROIs are non-evaluable, not silently admitted.

Lucchi and other segmentation-install datasets remain outside `raw_data/cohorts/manifest.csv`.

## Retired Runtime Input Surfaces

The active scored-cohort input surface is only `raw_data/cohorts/manifest.csv` plus `raw_data/cohorts/<cohort_id>/`. During the current rollout no overlapping active quantification-input directory was found under the runtime root. Retired historical static-patch trees remain under `<runtime_root>/_retired/` and are reference surfaces only, not active cohort or training inputs.

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

Current code and documentation should resolve paths through `src/eq/utils/paths.py`. The path helpers resolve the active runtime root and cohort surfaces consistently:

- `data/raw_data`
- `data/derived_data`
- `models/segmentation`
- `logs`
- `output`
- `<runtime_root>/raw_data/cohorts/manifest.csv`
- `<runtime_root>/raw_data/cohorts/<cohort_id>/`
- `<runtime_root>/output/cohorts/<cohort_id>/`

Avoid older machine-specific absolute paths and legacy directory names such as bare `derived_data/` at the repo root unless you are working on explicit backward-compatibility code.

## Environment Overrides

The project supports a small set of path overrides when needed:

- `EQ_DATA_PATH`
- `EQ_OUTPUT_PATH`
- `EQ_CACHE_PATH`
- `EQ_MODEL_PATH`
- `EQ_RUNTIME_ROOT`
- `EQ_RUNTIME_OUTPUT_PATH`
- `EQ_RUNTIME_MODEL_PATH`
- `EQ_DOX_LABEL_STUDIO_EXPORT`
- `EQ_MR_SCORE_WORKBOOK`
- `EQ_MR_IMAGE_ROOT`

These should be the exception rather than the default.
