## 1. P0 Evidence Audit

- [ ] 1.1 Inspect the archived P0 runtime artifacts under `burden_model/learned_roi/` and record available feature families, selected candidates, score support, source support, warnings, and artifact availability.
- [ ] 1.2 Decide from the audit whether P1 source-adjusted candidates use explicit `cohort_id` indicators, cohort-specific calibration, or both, and record the decision in `audit-results.md`.
- [ ] 1.3 Decide from the audit whether source-aware estimator outputs can ever enter `readme_results_snippet.md` in this change or must remain runtime-review-only, and record the decision in `audit-results.md`.
- [ ] 1.4 Audit current ROI extraction artifacts and record total rows, usable ROI rows, failed ROI rows, `roi_status` counts, source support, subject support, and available segmentation/mask provenance in `audit-results.md`.

## 2. Core Estimator Module

- [ ] 2.1 Create `src/eq/quantification/source_aware_estimator.py` with output path helpers for `burden_model/source_aware_estimator/`.
- [ ] 2.2 Implement input validation for required identity columns, `subject_id` grouping, `cohort_id` source context, supported score rubric, finite feature columns, and provenance fields.
- [ ] 2.3 Implement feature assembly from available ROI/QC, learned ROI, burden, and embedding-derived inputs without fitting new optional backbone or foundation providers.
- [ ] 2.4 Implement candidate configuration only after completing tasks 1.2 and 1.4 so source adjustment and ROI adequacy decisions are recorded before fitting starts.
- [ ] 2.5 Implement capped candidate evaluation for `pooled_roi_qc`, `pooled_learned_roi`, `pooled_hybrid`, `source_adjusted_roi_qc`, `source_adjusted_hybrid`, `within_source_calibrated_hybrid`, and `subject_source_adjusted_hybrid`.
- [ ] 2.6 Implement grouped subject-heldout validation and subject-level aggregation so no `subject_id` appears in both train and validation partitions for a candidate fold.

## 3. Source Sensitivity And Reliability

- [ ] 3.1 Write `diagnostics/source_sensitivity.json` with row/subject counts by `cohort_id`, score support by source, pooled metrics, within-source metrics, source-adjusted metrics, and leave-source-out metrics where estimable.
- [ ] 3.2 Write `diagnostics/reliability_labels.json` defining hard blockers, scope limiters, and per-prediction labels.
- [ ] 3.3 Classify score-2 ambiguity, broad intervals, source sensitivity, small strata, leave-source-out degradation, and nonfatal numerical warnings as scope limiters rather than automatic global blockers.
- [ ] 3.4 Preserve hard blockers for broken joins, unsupported scores, nonfinite selected predictions, validation leakage, untraceable provenance, missing verdict/index artifacts, and claims outside grade-equivalent prediction.
- [ ] 3.5 Label missing or non-training `cohort_id` rows as `unknown_source` and exclude them from standard known-source reportable summaries.
- [ ] 3.6 Separate reportability as `image_level`, `subject_level`, and `aggregate_current_data` in the verdict and combined summaries.
- [ ] 3.7 Compute metrics by split with explicit `training_apparent`, `validation_subject_heldout`, `testing_explicit_heldout`, and `testing_not_available_current_data_sensitivity` labels as applicable.
- [ ] 3.8 Ensure full-dataset apparent metrics are never labeled as independent testing.
- [ ] 3.9 Generate the capped six-figure summary set under `summary/figures/` for metrics by split, predicted vs observed, calibration by score, source performance, uncertainty width distribution, and reliability-label counts.

## 4. Contained Artifacts

- [ ] 4.1 Write `summary/estimator_verdict.json` and `summary/estimator_verdict.md` with selected candidates, hard blockers, scope limiters, reportable scopes, non-reportable scopes, README eligibility, and next action.
- [ ] 4.2 Write `summary/metrics_by_split.csv` and `summary/metrics_by_split.json` with training/apparent, subject-heldout validation, and explicit-heldout-testing or testing-not-available rows.
- [ ] 4.3 Write `predictions/image_predictions.csv` with burden estimates, uncertainty fields, candidate ID, fold or validation source, `cohort_id`, observed score when available, and reliability label.
- [ ] 4.4 Write `predictions/subject_predictions.csv` with one row per held-out or summarized `subject_id` per selected subject-level candidate and reliability label.
- [ ] 4.5 Write `diagnostics/upstream_roi_adequacy.json` and include its status in `summary/estimator_verdict.json`.
- [ ] 4.6 Write `internal/candidate_metrics.csv` and `internal/candidate_summary.json` for exhaustive candidate evidence, not as first-read human-facing artifacts.
- [ ] 4.7 Write `evidence/source_aware_estimator_review.html` with representative correct, high-error, high-uncertainty, score-2-like, source-sensitive, and subject-summary examples where available.
- [ ] 4.8 Write `summary/artifact_manifest.json` listing every source-aware estimator artifact by path, role, consumer, reportability, and required status.
- [ ] 4.9 Include all six summary figure paths in `summary/artifact_manifest.json` and link them from `summary/estimator_verdict.md`.
- [ ] 4.10 Write `INDEX.md` at `burden_model/source_aware_estimator/` that explains what to open first, artifact roles, trust status, claim boundary, figures, and every manifest-listed artifact.
- [ ] 4.11 Verify the workflow writes no duplicate source-aware estimator flat aliases under `burden_model/*` and no unmanifested source-aware estimator artifacts.

## 5. Pipeline And Report Integration

- [ ] 5.1 Export `evaluate_source_aware_endotheliosis_estimator` from `src/eq/quantification/__init__.py`.
- [ ] 5.2 Add one bounded call from `src/eq/quantification/pipeline.py` after burden and learned ROI artifacts are available.
- [ ] 5.3 Update `quantification_review/quantification_review.html` generation to link `burden_model/source_aware_estimator/INDEX.md` and show verdict-level source-aware status.
- [ ] 5.4 Link or embed the capped source-aware summary figures from `quantification_review/quantification_review.html`.
- [ ] 5.5 Update `results_summary.csv` and `results_summary.md` generation with source-aware hard-blocker, scope-limiter, upstream ROI adequacy, training/apparent metrics, validation metrics, testing availability, image-reportable, subject-reportable, aggregate-current-data-reportable, and README-eligibility rows.
- [ ] 5.6 Update `readme_results_snippet.md` generation so source-aware results appear only when `estimator_verdict.json` marks the relevant scope eligible.

## 6. Tests

- [ ] 6.1 Add `tests/unit/test_quantification_source_aware_estimator.py` covering input validation, capped candidate IDs, grouped validation, metrics-by-split labeling, summary figure generation, upstream ROI adequacy, unknown-source labeling, scoped reportability, source diagnostics, reliability labels, artifact containment, manifest completeness, and no flat aliases.
- [ ] 6.2 Extend existing quantification pipeline tests only for the combined report contract and README-snippet eligibility behavior.
- [ ] 6.3 Add regression coverage proving score-2 undercoverage produces reliability/scope labels rather than suppressing prediction rows.
- [ ] 6.4 Add regression coverage proving hard blockers prevent estimator claims when selected predictions are nonfinite or `subject_id` leakage is detected.
- [ ] 6.5 Add regression coverage proving unknown-source rows are labeled `unknown_source` and excluded from standard known-source reportable summaries.
- [ ] 6.6 Add regression coverage proving apparent full-dataset metrics use `training_apparent` and are not labeled as independent testing when no explicit test partition exists.
- [ ] 6.7 Add regression coverage proving only the six required first-pass summary figures are generated unless a named manifest consumer is recorded.

## 7. Documentation

- [ ] 7.1 Update `docs/OUTPUT_STRUCTURE.md` to document `burden_model/source_aware_estimator/` as an indexed experimental estimator subtree.
- [ ] 7.2 Update current-state user documentation only if the estimator verdict supports a reportable scope; otherwise document where runtime reviewers can inspect the experimental source-aware index.
- [ ] 7.3 Keep public-facing wording predictive and grade-equivalent; do not describe source-aware outputs as true tissue percent, closed-capillary percent, causal evidence, or external validation.

## 8. Validation

- [ ] 8.1 Run `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest -q tests/unit/test_quantification_source_aware_estimator.py`.
- [ ] 8.2 Run focused quantification tests covering learned ROI, burden, pipeline report integration, and source-aware estimator behavior.
- [ ] 8.3 Run `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest -q`.
- [ ] 8.4 Run `ruff check` and `ruff format --check` on changed source and test files.
- [ ] 8.5 Run `openspec validate p1-source-aware-endotheliosis-estimator --strict`.
- [ ] 8.6 Run `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python scripts/check_openspec_explicitness.py openspec/changes/p1-source-aware-endotheliosis-estimator`.
- [ ] 8.7 Run the full workflow with `PYTORCH_ENABLE_MPS_FALLBACK=1 MPLCONFIGDIR=/tmp/mpl_eq PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m eq run-config --config configs/endotheliosis_quantification.yaml`.
- [ ] 8.8 Record the full runtime verdict, artifact root, upstream ROI adequacy, training/apparent metrics, validation metrics, testing availability, summary figure paths, hard blockers, scope limiters, reportable scopes, README eligibility, manifest completeness, and next action in `audit-results.md`.
