# Output Structure

This document describes the recommended local directory layout for data, derived artifacts, trained models, and general outputs.

## Principles

- Keep source code and configuration in Git.
- Keep datasets, trained models, logs, and generated outputs local-only.
- Use the configured runtime root for heavy data and generated artifacts. This checkout's local default is recorded in `analysis_registry.yaml`.
- Do not use repo-root `raw_data/`, `derived_data/`, `models/`, `logs/`, `output/`, `data/`, `archive/`, `backups/`, `cache/`, or `test_output/` as active artifact roots.

## Recommended Layout

```text
endotheliosis_quantifier/
├── configs/
├── docs/
├── openspec/
├── src/eq/
└── tests/

$EQ_RUNTIME_ROOT/
├── raw_data/
│   ├── cohorts/
│   │   ├── manifest.csv
│   │   └── <cohort_id>/
│   │       ├── images/
│   │       ├── masks/
│   │       ├── scores/
│   │       └── metadata/
│   ├── lucchi/
│   └── mitochondria_data/
│       ├── training/
│       │   ├── images/
│       │   └── masks/
│       └── testing/
│           ├── images/
│           └── masks/
├── derived_data/
│   ├── cohort_manifest/
│   │   └── manifest_summary.json
│   ├── segmentation_cache/
│   └── <project_name>/
├── logs/
│   └── run_config/<run_id>/
├── models/
│   └── segmentation/
│       ├── mitochondria/<run_id>/
│       └── glomeruli/
│           ├── transfer/<run_id>/
│           └── scratch/<run_id>/
└── output/
```

## Purpose Of Each Directory

- `$EQ_RUNTIME_ROOT/raw_data/`
  Original source datasets and localized cohort inputs. For scored cohorts, active image/mask pairs and Label Studio-derived score material live in localized cohort directories:

  ```text
  $EQ_RUNTIME_ROOT/raw_data/cohorts/<cohort_id>/
  ├── images/
  ├── masks/
  ├── scores/
  │   ├── labelstudio_annotations.json
  │   └── labelstudio_scores.csv
  └── metadata/
      └── subject_metadata.xlsx
  ```

  Do not document or use separate active project trees for segmentation training. Single-cohort training uses `raw_data/cohorts/<cohort_id>`; all-available masked training uses `raw_data/cohorts`.

- `$EQ_RUNTIME_ROOT/raw_data/mitochondria_data/`
  Installed Lucchi mitochondria full-image training and held-out testing data. This dataset is a segmentation-install dataset, not a scored quantification cohort, so it stays outside `raw_data/cohorts/manifest.csv`.

  ```text
  $EQ_RUNTIME_ROOT/raw_data/mitochondria_data/
  ├── training/
  │   ├── images/
  │   └── masks/
  └── testing/
      ├── images/
      └── masks/
  ```

  The `training/` root is used for dynamic training and its internal train/validation split. The `testing/` root is held out for explicit evaluation.

- `$EQ_RUNTIME_ROOT/derived_data/`
  Generated outputs such as manifest summaries, audits, caches, metrics, or cached intermediates. Do not use `derived_data` as the canonical location for curated glomeruli training inputs. `derived_data/glomeruli_data` and `derived_data/mitochondria_data` are not supported active training-data roots.

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
  Run logs and temporary experiment logs. `eq run-config` writes a tee of the workflow commands and subprocess output to `logs/run_config/<run_id>/`.

- `output/`
  General-purpose generated outputs such as visualizations, quantification review reports, and one-off analysis exports.

## Runtime Scored Cohort Layout

Scored quantification cohorts use the active runtime root, not the repo checkout's placeholder `data/` tree:

```text
$EQ_RUNTIME_ROOT/
├── raw_data/
│   └── cohorts/
│       ├── manifest.csv
│       └── <cohort_id>/
│           ├── images/
│           ├── masks/
│           ├── scores/
│           └── metadata/
├── derived_data/
│   └── cohort_manifest/
│       └── manifest_summary.json
└── output/
    ├── segmentation_evaluation/
    │   ├── mitochondria/<run_id>/
    │   ├── glomeruli/<run_id>/
    │   └── glomeruli_candidate_comparison/<run_id>/
    ├── predictions/
    │   ├── mitochondria/<model_run_id>/<input_set>/
    │   └── glomeruli/<model_run_id>/<input_set>/
    └── quantification_results/
        └── <cohort_id>/
            ├── predicted_roi/
            ├── embeddings/
            ├── grader_outputs/
            └── review/
```

`raw_data/cohorts/manifest.csv` is the dataset-wide table. It records runtime-local paths, cohort identity, score linkage, lane assignment, hashes, verification state, and admission state. It does not carry original source or cloud source paths.

The naming contract is:

- `cohort_id` is the project or biological cohort identity.
- `lane_assignment` is the workflow/admission lane, such as `manual_mask_core`, `manual_mask_external`, `scored_only`, or `mr_concordance_only`.
- `manual_mask_core` and `manual_mask_external` are both first-class manual-mask glomeruli training lanes when admitted. They preserve source provenance and workflow role.
- Do not encode mask state in cohort identity; preserve the source cohort ID and use `lane_assignment` for workflow role.

The supported admission states are explicit:

- `unresolved`: linkage or required evidence is still incomplete after the recorded discovery surfaces.
- `pending_discovery`: a row is recoverable in principle, but the declared cohort discovery surfaces are not exhausted yet.
- `excluded`: a contradiction, unreadable asset, ambiguous mapping, or failed gate blocks the row.
- `foreign`: the row came from a mixed export and does not belong to the intended cohort.
- `admitted`: mapping verification passed and the row is usable for the lane stated in `lane_assignment`.
- `pending_transport_audit`: a scored-only row needs cohort-specific segmentation transport review before predicted-ROI grading use.
- `evaluation_only`: rows are available for evaluation or concordance checks, not training-set expansion.

Generated cohort summaries are written under `derived_data/cohort_manifest/`. The current local cohort snapshot, including concrete cohort IDs, row counts, and unresolved-source notes, is recorded in [TECHNICAL_LAB_NOTEBOOK.md](TECHNICAL_LAB_NOTEBOOK.md#local-cohort-manifest-snapshot).

Lucchi and other segmentation-install datasets remain outside `raw_data/cohorts/manifest.csv`.

## Runtime Input Surfaces

The active scored-cohort input surface is `raw_data/cohorts/manifest.csv` plus `raw_data/cohorts/<cohort_id>/`. Active training and quantification docs should point to those surfaces.

For quantification runs, a typical output subtree looks like:

```text
output/quantification_results/<cohort_id>/
├── labelstudio_scores/
├── scored_examples/
├── roi_crops/
├── embeddings/
├── burden_model/
│   ├── primary_model/
│   │   ├── burden_model.joblib
│   │   ├── burden_predictions.csv
│   │   ├── final_model_predictions.csv
│   │   └── burden_metrics.json
│   ├── validation/
│   │   ├── threshold_metrics.csv
│   │   ├── threshold_support.csv
│   │   ├── grouping_audit.json
│   │   ├── validation_design.json
│   │   └── cohort_stability.csv
│   ├── calibration/
│   │   ├── calibration_bins.csv
│   │   └── uncertainty_calibration.json
│   ├── summaries/
│   │   ├── cohort_metrics.csv
│   │   ├── final_model_cohort_metrics.csv
│   │   ├── group_summary_intervals.csv
│   │   └── final_model_group_summary_intervals.csv
│   ├── evidence/
│   │   ├── prediction_explanations.csv
│   │   ├── nearest_examples.csv
│   │   └── morphology_feature_review/
│   │       ├── feature_review.html
│   │       ├── feature_review_cases.csv
│   │       ├── operator_adjudication_template.csv
│   │       ├── operator_adjudication_agreement.json
│   │       └── assets/
│   ├── candidates/
│   │   ├── signal_comparator_metrics.csv
│   │   ├── subject_level_candidate_predictions.csv
│   │   ├── precision_candidate_summary.json
│   │   ├── morphology_candidate_metrics.csv
│   │   ├── subject_morphology_candidate_predictions.csv
│   │   └── morphology_candidate_summary.json
│   ├── diagnostics/
│   │   └── morphology_feature_diagnostics.json
│   ├── feature_sets/
│   │   ├── morphology_features.csv
│   │   ├── morphology_feature_metadata.json
│   │   └── subject_morphology_features.csv
│   └── learned_roi/
│       ├── feature_sets/
│       │   ├── learned_roi_features.csv
│       │   └── learned_roi_feature_metadata.json
│       ├── candidates/
│       │   ├── learned_roi_candidate_metrics.csv
│       │   └── learned_roi_candidate_summary.json
│       ├── validation/
│       │   ├── learned_roi_predictions.csv
│       │   └── learned_roi_subject_predictions.csv
│       ├── calibration/
│       │   └── learned_roi_calibration.json
│       ├── summaries/
│       │   └── learned_roi_subject_summary_intervals.json
│       ├── diagnostics/
│       │   ├── provider_audit.json
│       │   ├── learned_roi_feature_diagnostics.json
│       │   └── cohort_confounding_diagnostics.json
│       └── evidence/
│           ├── learned_roi_review.html
│           ├── learned_roi_review_examples.csv
│           ├── learned_roi_nearest_examples.csv
│           ├── learned_roi_attribution_status.json
│           └── assets/
├── ordinal_model/
│   ├── ordinal_predictions.csv
│   ├── ordinal_metrics.json
│   ├── ordinal_confusion_matrix.csv
│   └── review_report/
│       ├── ordinal_review.html
│       ├── selected_examples.csv
│       └── assets/
└── quantification_review/
    ├── quantification_review.html
    ├── review_examples.csv
    ├── results_summary.md
    ├── results_summary.csv
    ├── readme_results_snippet.md
    └── assets/
```

`burden_model/primary_model/burden_model.joblib` is the serialized exploratory burden model artifact. Candidate-screen files under `burden_model/candidates/` are comparison and planning artifacts, not separately deployed models. Use `burden_model/primary_model/burden_predictions.csv` for held-out validation evidence and `burden_model/primary_model/final_model_predictions.csv` for final full-cohort fitted summaries. Morphology features under `burden_model/feature_sets/` are deterministic review features; inspect `burden_model/evidence/morphology_feature_review/feature_review.html` and the operator adjudication output before treating them as biologically reliable evidence.

`burden_model/learned_roi/` contains the capped learned-ROI candidate screen. Phase 1 fits only the current glomeruli encoder embeddings, simple ROI QC features, and their hybrid. Optional backbone or foundation providers are recorded in `diagnostics/provider_audit.json` as audit-only, unavailable, or failed; they are not fitted candidates. Use `candidates/learned_roi_candidate_summary.json`, `calibration/learned_roi_calibration.json`, and `diagnostics/cohort_confounding_diagnostics.json` to determine whether any learned ROI track is README/docs-ready. If readiness gates fail, the learned ROI outputs are failure evidence and review artifacts, not a promoted quantification claim.

- `test_output/`
  Temporary files created by tests or debugging scripts.

## Path Conventions In Code

Current code and documentation should resolve paths through `src/eq/utils/paths.py`. The path helpers resolve the active runtime root and cohort surfaces consistently:

- `<runtime_root>/raw_data`
- `<runtime_root>/derived_data`
- `<runtime_root>/models/segmentation`
- `<runtime_root>/models/segmentation/mitochondria/<run_id>/`
- `<runtime_root>/models/segmentation/glomeruli/{transfer,scratch}/<run_id>/`
- `<runtime_root>/logs`
- `<runtime_root>/logs/run_config/<run_id>/`
- `<runtime_root>/output`
- `<runtime_root>/raw_data/cohorts/manifest.csv`
- `<runtime_root>/raw_data/cohorts/<cohort_id>/`
- `<runtime_root>/raw_data/mitochondria_data/`
- `<runtime_root>/derived_data/cohort_manifest/manifest_summary.json`
- `<runtime_root>/output/segmentation_evaluation/<task_or_evaluation>/<run_id>/`
- `<runtime_root>/output/predictions/<task>/<model_run_id>/<input_set>/`
- `<runtime_root>/output/quantification_results/<cohort_id>/`

Use the runtime-root-relative paths above as the canonical path contract in current code and docs.

## Environment Overrides

The project supports a small set of path overrides when needed:

- `EQ_DATA_PATH`
- `EQ_OUTPUT_PATH`
- `EQ_CACHE_PATH`
- `EQ_MODEL_PATH`
- `EQ_RUNTIME_ROOT`
- `EQ_RUNTIME_OUTPUT_PATH`
- `EQ_RUNTIME_MODEL_PATH`

These should be the exception rather than the default. Site-specific cohort-source overrides are local data plumbing and are documented in the technical lab notebook or path-helper code when needed.
