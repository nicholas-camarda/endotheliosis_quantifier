## ADDED Requirements

### Requirement: Source-aware estimator uses the current quantification workflow
The source-aware endotheliosis estimator SHALL run inside the existing quantification YAML workflow and SHALL use the current scored MR TIFF/ROI evidence contract.

#### Scenario: Estimator runs from the main quantification config
- **WHEN** `eq run-config --config configs/endotheliosis_quantification.yaml` completes burden and learned ROI evaluation
- **THEN** the workflow SHALL call `evaluate_source_aware_endotheliosis_estimator`
- **AND** the estimator SHALL use `subject_id` as the validation grouping key
- **AND** the estimator SHALL use `cohort_id` as the primary source/context field
- **AND** the estimator SHALL preserve `subject_id`, `sample_id`, `image_id`, `subject_image_id`, `cohort_id`, `score`, and ROI/image path provenance in prediction artifacts where present

#### Scenario: Unsupported score values fail closed
- **WHEN** source-aware estimator inputs contain a score outside `[0.0, 0.5, 1.0, 1.5, 2.0, 3.0]`
- **THEN** the estimator SHALL fail before candidate fitting
- **AND** the error SHALL identify the unsupported score values and the supported rubric

### Requirement: Source-aware estimator artifacts are contained and indexed
The source-aware estimator SHALL write all artifacts under one indexed output subtree and SHALL avoid duplicate flat aliases.

#### Scenario: Indexed output subtree is written
- **WHEN** source-aware estimator evaluation completes
- **THEN** the workflow SHALL write `burden_model/source_aware_estimator/INDEX.md`
- **AND** all source-aware estimator artifacts SHALL live under `burden_model/source_aware_estimator/summary/`, `burden_model/source_aware_estimator/predictions/`, `burden_model/source_aware_estimator/diagnostics/`, `burden_model/source_aware_estimator/evidence/`, or `burden_model/source_aware_estimator/internal/`
- **AND** the workflow SHALL NOT write duplicate source-aware estimator artifacts to flat `burden_model/*` locations

#### Scenario: Index explains what to open first
- **WHEN** `burden_model/source_aware_estimator/INDEX.md` is opened
- **THEN** it SHALL identify the estimator verdict file, image predictions, subject predictions, source diagnostics, reliability diagnostics, evidence review, and internal candidate table
- **AND** it SHALL label each artifact as human-facing, diagnostic, prediction, evidence, or internal
- **AND** it SHALL state the claim boundary: grade-equivalent predictive burden for the current scored MR TIFF/ROI data only

#### Scenario: Artifact manifest caps first-pass outputs
- **WHEN** source-aware estimator evaluation completes
- **THEN** the workflow SHALL write `burden_model/source_aware_estimator/summary/artifact_manifest.json`
- **AND** the manifest SHALL list every source-aware estimator artifact by relative path, role, consumer, reportability, and whether the artifact is required
- **AND** first-pass source-aware estimator outputs SHALL be limited to `INDEX.md`, `summary/estimator_verdict.json`, `summary/estimator_verdict.md`, `summary/metrics_by_split.csv`, `summary/metrics_by_split.json`, `summary/artifact_manifest.json`, `summary/figures/metrics_by_split.png`, `summary/figures/predicted_vs_observed.png`, `summary/figures/calibration_by_score.png`, `summary/figures/source_performance.png`, `summary/figures/uncertainty_width_distribution.png`, `summary/figures/reliability_label_counts.png`, `predictions/image_predictions.csv`, `predictions/subject_predictions.csv`, `diagnostics/upstream_roi_adequacy.json`, `diagnostics/source_sensitivity.json`, `diagnostics/reliability_labels.json`, `evidence/source_aware_estimator_review.html`, `internal/candidate_metrics.csv`, and `internal/candidate_summary.json` unless a named consumer is recorded in the manifest

### Requirement: Source-aware candidates are explicit and bounded
The source-aware estimator SHALL evaluate a small fixed candidate set unless a later OpenSpec change expands it.

#### Scenario: Candidate set is capped
- **WHEN** source-aware candidates are fit
- **THEN** the only allowed candidate IDs SHALL be `pooled_roi_qc`, `pooled_learned_roi`, `pooled_hybrid`, `source_adjusted_roi_qc`, `source_adjusted_hybrid`, `within_source_calibrated_hybrid`, and `subject_source_adjusted_hybrid`
- **AND** every candidate row in `burden_model/source_aware_estimator/internal/candidate_metrics.csv` SHALL identify candidate ID, target level, feature family, source handling mode, row count, subject count, validation grouping, stage-index MAE, grade-scale MAE, coverage, average interval or prediction-set width, finite-output status, warning count, and intended use
- **AND** optional foundation/backbone providers SHALL NOT be added by this change

#### Scenario: Subject-level candidate uses subject aggregation
- **WHEN** `subject_source_adjusted_hybrid` is fit
- **THEN** it SHALL aggregate features to one row per `subject_id`
- **AND** validation SHALL hold out subjects
- **AND** outputs SHALL label the target level as `subject`

### Requirement: Source-aware validation reports source behavior
The estimator SHALL report pooled, within-source, source-adjusted, and leave-source-out behavior without presenting source-aware performance as external validation.

#### Scenario: Source diagnostics are written
- **WHEN** source-aware estimator evaluation completes
- **THEN** the workflow SHALL write `burden_model/source_aware_estimator/diagnostics/source_sensitivity.json`
- **AND** the artifact SHALL include row and subject counts by `cohort_id`, score support by `cohort_id`, pooled metrics, within-source metrics, source-adjusted metrics, and leave-source-out metrics where estimable
- **AND** non-estimable source/score cells SHALL be labeled with a reason rather than omitted silently

#### Scenario: Unknown source context is labeled
- **WHEN** a prediction row has missing `cohort_id` or a `cohort_id` not represented in the source-aware training/evaluation context
- **THEN** the estimator SHALL label the row with `unknown_source`
- **AND** the row SHALL NOT be counted as a standard known-source estimate in reportable known-source summaries
- **AND** the estimator verdict SHALL state whether unknown-source rows were predicted, excluded from a scope, or blocked

#### Scenario: Leave-source-out results are scoped
- **WHEN** leave-source-out evaluation is estimable
- **THEN** the estimator verdict SHALL report train-source, test-source, row count, subject count, stage-index MAE, grade-scale MAE, coverage, and qualitative degradation status
- **AND** the report SHALL state that leave-source-out evaluation is a current-data sensitivity check, not external validation

### Requirement: Source-aware estimator reports metrics by split
The estimator SHALL report training/apparent, validation, and testing metrics with explicit split labels.

#### Scenario: Metrics by split are written
- **WHEN** source-aware estimator evaluation completes
- **THEN** the workflow SHALL write `burden_model/source_aware_estimator/summary/metrics_by_split.csv`
- **AND** the workflow SHALL write `burden_model/source_aware_estimator/summary/metrics_by_split.json`
- **AND** each metrics row SHALL include candidate ID, target level, split label, split description, row count, subject count, source scope, stage-index MAE, grade-scale MAE, coverage where applicable, average interval or prediction-set width where applicable, finite-output status, warning count, and whether the split is eligible for model selection

#### Scenario: Training apparent metrics are labeled
- **WHEN** the selected estimator is fit on the full available scored dataset or on training folds
- **THEN** apparent or in-fold metrics SHALL use the split label `training_apparent`
- **AND** reports SHALL state that `training_apparent` metrics are optimistic diagnostics and not evidence of held-out performance

#### Scenario: Subject-heldout validation metrics are primary
- **WHEN** grouped out-of-fold validation is performed
- **THEN** subject-heldout validation metrics SHALL use the split label `validation_subject_heldout`
- **AND** no `subject_id` SHALL appear in both train and validation partitions for a validation fold
- **AND** `validation_subject_heldout` SHALL be the primary current-data model-selection split unless an explicit held-out test partition is predeclared

#### Scenario: Testing metrics require an explicit held-out test partition
- **WHEN** a predeclared held-out test partition exists and is not used for model fitting, candidate selection, calibration tuning, or reliability-threshold tuning
- **THEN** testing metrics SHALL use the split label `testing_explicit_heldout`
- **AND** the metrics artifact SHALL identify the test partition source and exclusion rule

#### Scenario: No explicit test set is labeled honestly
- **WHEN** no predeclared independent held-out test partition exists
- **THEN** the metrics artifact SHALL include `testing_not_available_current_data_sensitivity`
- **AND** leave-source-out, within-source, and full-dataset sensitivity checks SHALL NOT be labeled as independent testing

### Requirement: Source-aware estimator writes summary figures
The estimator SHALL write a capped set of human-facing summary figures for model behavior, calibration, source sensitivity, uncertainty, and reliability labels.

#### Scenario: Summary figures are written
- **WHEN** source-aware estimator evaluation completes
- **THEN** the workflow SHALL write `burden_model/source_aware_estimator/summary/figures/metrics_by_split.png`
- **AND** the workflow SHALL write `burden_model/source_aware_estimator/summary/figures/predicted_vs_observed.png`
- **AND** the workflow SHALL write `burden_model/source_aware_estimator/summary/figures/calibration_by_score.png`
- **AND** the workflow SHALL write `burden_model/source_aware_estimator/summary/figures/source_performance.png`
- **AND** the workflow SHALL write `burden_model/source_aware_estimator/summary/figures/uncertainty_width_distribution.png`
- **AND** the workflow SHALL write `burden_model/source_aware_estimator/summary/figures/reliability_label_counts.png`
- **AND** all six figures SHALL be listed in `summary/artifact_manifest.json`

#### Scenario: Metrics by split figure compares train validation and testing availability
- **WHEN** `summary/figures/metrics_by_split.png` is rendered
- **THEN** it SHALL show at least training/apparent, subject-heldout validation, and testing-explicit or testing-not-available/current-data-sensitivity status for selected image-level and subject-level candidates where available
- **AND** it SHALL not label current-data sensitivity as independent testing

#### Scenario: Calibration and uncertainty figures expose weak regions
- **WHEN** `summary/figures/calibration_by_score.png` and `summary/figures/uncertainty_width_distribution.png` are rendered
- **THEN** the figures SHALL make score strata, broad intervals, and score-2-like uncertainty visible where estimable
- **AND** non-estimable strata SHALL be represented as missing or explicitly labeled rather than silently removed from the figure metadata

#### Scenario: Source and reliability figures expose scope limiters
- **WHEN** `summary/figures/source_performance.png` and `summary/figures/reliability_label_counts.png` are rendered
- **THEN** the figures SHALL show source/cohort behavior and reliability-label counts for selected reportable scopes where estimable
- **AND** unknown-source rows SHALL be visually distinct or explicitly counted when present

### Requirement: Source-aware estimator emits reliability labels
The estimator SHALL separate hard blockers from scope limiters and per-prediction reliability labels.

#### Scenario: Reliability diagnostics are written
- **WHEN** source-aware estimator evaluation completes
- **THEN** the workflow SHALL write `burden_model/source_aware_estimator/diagnostics/reliability_labels.json`
- **AND** the artifact SHALL define all emitted reliability labels, including `standard_current_data`, `wide_uncertainty`, `transitional_score_region`, `source_sensitive`, `underpowered_stratum`, `leave_source_out_degraded`, `numerical_warning_scope_limiter`, and `unknown_source`
- **AND** the artifact SHALL define hard blockers, including broken joins, unsupported scores, nonfinite selected predictions, validation leakage across `subject_id`, untraceable provenance, missing required verdict/index artifacts, and claims outside grade-equivalent prediction

#### Scenario: Image predictions include reliability labels
- **WHEN** image-level predictions are written
- **THEN** `burden_model/source_aware_estimator/predictions/image_predictions.csv` SHALL contain predicted grade-equivalent burden, uncertainty interval or prediction-set fields, candidate ID, fold or validation source, `cohort_id`, observed `score` when available, and `reliability_label`
- **AND** rows with score-2-like ambiguity or broad uncertainty SHALL remain present in the predictions table with a reliability label rather than being dropped

#### Scenario: Subject predictions include reliability labels
- **WHEN** subject-level predictions are written
- **THEN** `burden_model/source_aware_estimator/predictions/subject_predictions.csv` SHALL contain one row per held-out or summarized `subject_id` per selected subject-level candidate
- **AND** each row SHALL include observed subject mean target where available, predicted subject burden, uncertainty interval, `cohort_id`, candidate ID, and `reliability_label`

### Requirement: Source-aware estimator verdict is reader-first
The estimator SHALL produce a top-level verdict that answers what happened, what can be trusted, and what remains limited.

#### Scenario: Verdict artifacts are written
- **WHEN** source-aware estimator evaluation completes
- **THEN** the workflow SHALL write `burden_model/source_aware_estimator/summary/estimator_verdict.json`
- **AND** the workflow SHALL write `burden_model/source_aware_estimator/summary/estimator_verdict.md`
- **AND** the verdict SHALL identify selected image candidate if any, selected subject candidate if any, upstream ROI adequacy status, hard blockers, scope limiters, reportable scopes, non-reportable scopes, next action, and whether README snippets may include the result
- **AND** reportable scopes SHALL be separated as `image_level`, `subject_level`, and `aggregate_current_data`

#### Scenario: Fixed-data limitations become scope limiters
- **WHEN** score-specific undercoverage, wide intervals, source sensitivity, small source/score strata, leave-source-out degradation, or nonfatal numerical warnings occur without broken joins, leakage, nonfinite selected predictions, or invalid claims
- **THEN** the estimator verdict SHALL classify those conditions as scope limiters
- **AND** the workflow SHALL still write predictions, uncertainty, and reliability labels
- **AND** the report SHALL avoid presenting limited scopes as solved

#### Scenario: Hard blockers prevent estimator claims
- **WHEN** a hard blocker is present
- **THEN** `estimator_verdict.json` SHALL set the relevant estimator claim status to blocked
- **AND** combined reports SHALL not present the blocked result as usable for analysis

### Requirement: Upstream MR TIFF to ROI adequacy is reported
The source-aware estimator SHALL report whether upstream MR TIFF, mask, and ROI inputs are adequate for the estimator scopes it presents.

#### Scenario: Upstream ROI adequacy diagnostics are written
- **WHEN** source-aware estimator evaluation completes
- **THEN** the workflow SHALL write `burden_model/source_aware_estimator/diagnostics/upstream_roi_adequacy.json`
- **AND** the artifact SHALL include total input rows, scored rows, usable ROI rows, failed ROI rows, `roi_status` counts, row and subject counts by `cohort_id`, manual-mask or model-derived mask context where present, segmentation model path or provenance where present, and ROI/image path availability counts
- **AND** non-estimable or missing upstream provenance fields SHALL be recorded with explicit reasons rather than omitted silently

#### Scenario: Upstream inadequacy limits reportable scope
- **WHEN** usable ROI rows, subject groups, source support, or provenance are insufficient for an image-level, subject-level, or aggregate-current-data scope
- **THEN** `estimator_verdict.json` SHALL mark that scope non-reportable or limited
- **AND** the workflow SHALL avoid presenting downstream model metrics as evidence that upstream MR TIFF-to-ROI processing was adequate

### Requirement: Source-aware evidence remains predictive
Source-aware estimator evidence SHALL support review of predictive behavior without claiming histologic mechanism.

#### Scenario: Evidence review is written
- **WHEN** source-aware estimator evaluation completes
- **THEN** the workflow SHALL write `burden_model/source_aware_estimator/evidence/source_aware_estimator_review.html`
- **AND** the review SHALL link representative correct, high-error, high-uncertainty, score-2-like, source-sensitive, and subject-summary examples where available
- **AND** the review SHALL label visual evidence as predictive support evidence rather than proof of closed-lumen mechanism
