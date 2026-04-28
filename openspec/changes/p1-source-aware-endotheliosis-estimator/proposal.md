## Why

The phase-0 learned ROI change proved that learned ROI estimation can run end-to-end, but it also showed that promotion-style readiness gates are too rigid for the practical goal of estimating endotheliosis from the fixed MR TIFF data we have. This change reframes the next step as a source-aware, grade-calibrated estimator that always reports uncertainty, source sensitivity, and reliability limits instead of treating every fixed-data limitation as a hard stop.

## What Changes

- Add a bounded source-aware estimator workflow for the existing `configs/endotheliosis_quantification.yaml` quantification run.
- Keep the estimator claim predictive and grade-calibrated: estimate endotheliosis burden from MR TIFF-derived ROI evidence, not true tissue-area percent, closed-capillary percent, or causal/mechanistic endotheliosis.
- Reclassify score-specific undercoverage, especially score-2 ambiguity, from an automatic global blocker into an uncertainty and reliability-label condition unless it exposes broken implementation, leakage, or misleading claims.
- Treat `cohort_id` and related source behavior as modeled nuisance/context variables that must be reported through pooled, within-source, source-adjusted, and leave-source-out diagnostics.
- Require the estimator verdict to audit upstream MR TIFF-to-ROI adequacy, including ROI status counts, usable ROI counts, mask/source context, segmentation artifact provenance, and whether estimates are based on manual masks or model-derived masks.
- Define behavior for unknown or missing source context so future MR TIFF applications do not silently receive known-source reliability labels.
- Distinguish reportable image-level, subject-level, and aggregate/cohort-level use rather than treating "reasonable result" as a single binary status.
- Require explicit training/apparent, subject-heldout validation, and held-out testing or current-data sensitivity metrics, with clear labels so full-dataset apparent metrics are never represented as independent testing.
- Add source-aware candidate families under a single implementation surface rather than scattering experimental logic through the stable quantification pipeline.
- Reduce artifact confusion by requiring one indexed estimator output subtree with human-facing summaries separated from diagnostics, predictions, evidence, and internal candidate artifacts.
- Cap the first-pass artifact manifest so the output tree cannot grow without the index naming each artifact and its consumer.
- Add a capped summary figure set for the estimator verdict so training/validation/testing behavior, calibration, source sensitivity, uncertainty width, and reliability labels are visually reviewable without creating plot sprawl.
- Preserve hard fail-closed behavior for broken joins, unsupported labels, nonfinite predictions, held-out leakage, missing provenance, and untraceable output schemas.
- Keep experimental outputs out of README/result snippets unless the estimator summary explicitly marks the relevant scope as reportable for the current data and claim boundary.

## Capabilities

### New Capabilities

- `source-aware-endotheliosis-estimator`: Source-aware, grade-calibrated practical estimator contract for MR TIFF/ROI-derived endotheliosis burden outputs, reliability labels, source sensitivity, and artifact containment.

### Modified Capabilities

- `learned-roi-quantification`: Reclassify phase-0 learned ROI hard-readiness failures into P1 hard blockers, scope limiters, reliability labels, and source-aware candidate inputs.
- `endotheliosis-burden-index`: Add estimator artifact index requirements and clarify how source-aware estimator outputs appear in combined quantification reports and README snippets.

## Impact

- Affected command: `eq run-config --config configs/endotheliosis_quantification.yaml`.
- Affected modules: `src/eq/quantification/pipeline.py`, `src/eq/quantification/learned_roi.py`, `src/eq/quantification/learned_roi_review.py`, and new bounded module `src/eq/quantification/source_aware_estimator.py`.
- Affected tests: add `tests/unit/test_quantification_source_aware_estimator.py` and extend focused quantification pipeline tests only where the combined report contract changes.
- Affected artifact root: `burden_model/source_aware_estimator/` under the quantification output root.
- Affected report surfaces: `quantification_review/quantification_review.html`, `quantification_review/results_summary.csv`, `quantification_review/results_summary.md`, and `quantification_review/readme_results_snippet.md`.
- Storage boundary: no raw data, models, logs, or generated estimator outputs are committed; runtime artifacts remain under the configured runtime output root.
- Compatibility: P0 `burden_model/learned_roi/` artifacts remain evidence and inputs, but P1 SHALL NOT create duplicate flat aliases or back-compat shim outputs.
- Scientific claim boundary: outputs are predictive grade-equivalent burden estimates calibrated to the current scored MR TIFF/ROI data; they are not external validation, causal evidence, or tissue-percent measurements.

## Explicit Decisions

- Change name: `p1-source-aware-endotheliosis-estimator`.
- New orchestrator module: `src/eq/quantification/source_aware_estimator.py`.
- Public orchestration function: `evaluate_source_aware_endotheliosis_estimator`.
- Output root: `burden_model/source_aware_estimator/`.
- Top-level human entrypoint artifact: `burden_model/source_aware_estimator/INDEX.md`.
- Human-facing summary artifact: `burden_model/source_aware_estimator/summary/estimator_verdict.json`.
- Human-facing markdown summary: `burden_model/source_aware_estimator/summary/estimator_verdict.md`.
- Image prediction artifact: `burden_model/source_aware_estimator/predictions/image_predictions.csv`.
- Subject prediction artifact: `burden_model/source_aware_estimator/predictions/subject_predictions.csv`.
- Source diagnostics artifact: `burden_model/source_aware_estimator/diagnostics/source_sensitivity.json`.
- Metrics split artifact: `burden_model/source_aware_estimator/summary/metrics_by_split.csv`.
- Metrics split JSON artifact: `burden_model/source_aware_estimator/summary/metrics_by_split.json`.
- Summary figures directory: `burden_model/source_aware_estimator/summary/figures/`.
- Reliability diagnostics artifact: `burden_model/source_aware_estimator/diagnostics/reliability_labels.json`.
- Candidate metrics artifact: `burden_model/source_aware_estimator/internal/candidate_metrics.csv`.
- Candidate summary artifact: `burden_model/source_aware_estimator/internal/candidate_summary.json`.
- Evidence review artifact: `burden_model/source_aware_estimator/evidence/source_aware_estimator_review.html`.
- Upstream adequacy artifact: `burden_model/source_aware_estimator/diagnostics/upstream_roi_adequacy.json`.
- Artifact manifest artifact: `burden_model/source_aware_estimator/summary/artifact_manifest.json`.
- Initial candidate IDs:
  - `pooled_roi_qc`
  - `pooled_learned_roi`
  - `pooled_hybrid`
  - `source_adjusted_roi_qc`
  - `source_adjusted_hybrid`
  - `within_source_calibrated_hybrid`
  - `subject_source_adjusted_hybrid`
- Source variables: use `cohort_id` as the required source/context field for P1; use `lane_assignment` only as a reported context field when present, not as a primary adjustment target.
- Unknown source behavior: rows with missing or non-training `cohort_id` SHALL receive `unknown_source` reliability labeling and SHALL NOT be presented as standard known-source estimates.
- Reportable scopes: verdicts SHALL separate `image_level`, `subject_level`, and `aggregate_current_data` scopes.
- Metrics split labels: use `training_apparent`, `validation_subject_heldout`, `testing_explicit_heldout`, and `testing_not_available_current_data_sensitivity` exactly.
- Required summary figures:
  - `summary/figures/metrics_by_split.png`
  - `summary/figures/predicted_vs_observed.png`
  - `summary/figures/calibration_by_score.png`
  - `summary/figures/source_performance.png`
  - `summary/figures/uncertainty_width_distribution.png`
  - `summary/figures/reliability_label_counts.png`
- Fixed hard blockers: broken joins, unsupported score values outside `[0.0, 0.5, 1.0, 1.5, 2.0, 3.0]`, nonfinite selected predictions, validation leakage across `subject_id`, untraceable provenance, missing required index/verdict artifacts, and claims that exceed grade-equivalent prediction.
- Fixed scope limiters: score-specific undercoverage, wide prediction sets, source sensitivity, small source/score strata, leave-source-out degradation, weak single-image reliability, and candidate numerical warnings that do not cause nonfinite predictions.

## Open Questions

- [audit_first_then_decide] Should source-adjusted candidates include a regularized explicit `cohort_id` indicator, cohort-specific calibration only, or both? Decide after auditing P0 selected features and score support by cohort in the implementation pass.
- [audit_first_then_decide] Should the README snippet ever include a P1 source-aware estimator result, or should it remain confined to runtime review artifacts until a later promotion change? Decide from the P1 verdict after the full YAML workflow rerun.
