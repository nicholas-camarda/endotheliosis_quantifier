## 1. Evidence Audit And Decisions

- [x] 1.1 Inspect `src/eq/quantification/burden.py`, `src/eq/quantification/source_aware_estimator.py`, and current runtime artifacts to decide whether P2 can reuse existing cumulative-threshold helpers or should keep a contained implementation in `src/eq/quantification/severe_aware_ordinal_estimator.py`.
- [x] 1.2 Record the helper-reuse decision in `audit-results.md`, including exact functions inspected, warning behavior observed, and the reason for reuse or containment.
- [x] 1.3 Inspect current P1 runtime outputs under `burden_model/source_aware_estimator/` and record selected candidates, high-score behavior, source support, scope limiters, and testing status in `audit-results.md`.
- [x] 1.4 Compute threshold support by row, subject, and source for `score >= 1.5`, `score >= 2`, and `score >= 3`; decide whether `score >= 3` is estimable, exploratory, or non-estimable.
- [x] 1.5 Audit available feature families and record whether ROI/QC, morphology, learned ROI, embeddings, or reduced embedding summaries are eligible for first-pass severe-aware candidate selection.
- [x] 1.6 Inspect severe false negatives and high-uncertainty severe cases for whether failures appear driven by bad ROI extraction, missed glomerulus segmentation, mask/ROI geometry error, feature/model limitation, or visually ambiguous grading signal.
- [x] 1.7 Decide whether current evidence justifies a future segmentation-backbone comparison or targeted Label Studio patch/mask annotation pilot, and record the decision in `audit-results.md`.
- [x] 1.8 Decide whether a MedSAM or promptable-SAM audit is warranted as a separate upstream comparison, and record oracle-prompt versus automatic-prompt feasibility in `audit-results.md`.
- [x] 1.9 Inventory feasibility for current FastAI U-Net, torchvision DeepLabV3, nnU-Net v2, DeepLabV3+ via external dependency, Mask2Former-style models, SAM, and MedSAM, including installed status, install path, weights/code source, hardware target, dataset conversion burden, and whether existing masks are sufficient.
- [x] 1.10 Decide whether alternative segmentation could improve severity-correlated feature extraction rather than only ROI geometry, and record candidate features affected by mask quality in `audit-results.md`.
- [x] 1.11 Record all audit-first decisions from `proposal.md` and `design.md` before final candidate fitting starts.
- [x] 1.12 Confirm the repo-wide execution logging contract from `p1-repo-wide-execution-logging-contract` is implemented before wiring P2 runtime execution, including availability of `src/eq/utils/execution_logging.py` and durable capture for `eq run-config --config configs/endotheliosis_quantification.yaml`.
- [x] 1.13 Record P2 logging participation in `audit-results.md`: `evaluate_severe_aware_ordinal_endotheliosis_estimator` is `high_level_function_events_only`, durable capture is owned by `endotheliosis_quantification`, and P2 adds no independent log root or file-handler system.

## 2. Core Estimator Module

- [x] 2.1 Create `src/eq/quantification/severe_aware_ordinal_estimator.py` with output path helpers for `burden_model/severe_aware_ordinal_estimator/`.
- [x] 2.2 Implement input validation for required identity columns, supported score rubric, `subject_id` grouping, `cohort_id` source context, finite feature columns, and required provenance fields.
- [x] 2.3 Implement source-aware handoff loading from `burden_model/source_aware_estimator/summary/estimator_verdict.json`, `summary/metrics_by_split.csv`, and selected prediction artifacts when present.
- [x] 2.4 Implement severe threshold support computation for `score >= 1.5`, `score >= 2`, and `score >= 3`.
- [x] 2.5 Implement feature-family assembly for ROI/QC, morphology, learned ROI, embedding-derived inputs, and any audit-approved reduced embedding summaries without adding new learned ROI providers.
- [x] 2.6 Implement candidate configuration only after tasks 1.2, 1.4, and 1.5 have recorded audit decisions.
- [x] 2.7 Add function-level logger events in `evaluate_severe_aware_ordinal_endotheliosis_estimator` for start, resolved inputs, output root, row/subject/source counts, threshold support, feature-family decisions, candidate IDs, hard blockers, scope limiters, verdict path, artifact manifest path, completion, and failure context.
- [x] 2.8 Ensure the evaluator does not call `setup_logging(...)`, attach durable file handlers, create `$EQ_RUNTIME_ROOT/logs/...`, create repo-root `logs/`, or implement custom subprocess teeing.

## 3. Severe Separability And Threshold Modeling

- [x] 3.1 Write `diagnostics/severe_separability_audit.json` with threshold support, feature-family support, feature diagnostics, warning diagnostics, and severe/non-severe descriptive summaries.
- [x] 3.2 Write `diagnostics/threshold_support.json` with positive/negative row and subject support by threshold and source.
- [x] 3.3 Implement subject-heldout severe-threshold candidates for audit-approved feature families.
- [x] 3.4 Implement ordinal/cumulative-threshold candidates that preserve the ordered score rubric `[0, 0.5, 1, 1.5, 2, 3]`.
- [x] 3.5 Implement two-stage candidates where a severe-risk gate informs or calibrates downstream burden or ordinal output.
- [x] 3.6 Implement subject-level severe-aware aggregation candidate `subject_severe_aware_ordinal`.
- [x] 3.7 Ensure candidate fitting fails closed on nonfinite selected predictions, validation leakage, unsupported scores, or missing required identity fields.

## 4. Metrics And Reliability

- [x] 4.1 Write `summary/metrics_by_split.csv` and `summary/metrics_by_split.json` with `training_apparent`, `validation_subject_heldout`, and testing-availability rows.
- [x] 4.2 Ensure apparent metrics are marked ineligible for model selection.
- [x] 4.3 Ensure no `subject_id` appears in both train and validation partitions for any candidate fold.
- [x] 4.4 Write `summary/severe_threshold_metrics.csv` with severe recall, severe precision, severe false-negative count, severe false-negative rate, threshold support, and source-stratified metrics where estimable.
- [x] 4.5 Write `diagnostics/source_severe_sensitivity.json` with severe-threshold behavior by `cohort_id`, leave-source-out sensitivity where estimable, and non-estimable source/threshold reasons.
- [x] 4.6 Write `diagnostics/reliability_labels.json` defining hard blockers, scope limiters, and per-prediction labels for severe-risk, ordinal uncertainty, source sensitivity, and underpowered thresholds.
- [x] 4.7 Classify score-2/3 underprediction, source-confounded severe support, broad ordinal prediction sets, underpowered thresholds, and nonfatal numerical warnings as scope limiters unless they expose a hard blocker.

## 5. Contained Artifacts

- [x] 5.1 Write `summary/estimator_verdict.json` and `summary/estimator_verdict.md` with selected candidates, selected threshold, selected output type, hard blockers, scope limiters, reportable scopes, non-reportable scopes, testing status, README eligibility, and next action.
- [x] 5.2 Write `predictions/image_predictions.csv` with observed score, predicted burden where available, severe-risk fields, ordinal threshold fields or prediction set where available, candidate ID, fold, `cohort_id`, reliability label, and provenance columns.
- [x] 5.3 Write `predictions/subject_predictions.csv` with subject-level observed target, predicted subject burden or severe risk, uncertainty fields, `cohort_id`, candidate ID, fold, and reliability label.
- [x] 5.4 Write `internal/candidate_metrics.csv` and `internal/candidate_summary.json` for exhaustive candidate evidence.
- [x] 5.5 Write `evidence/severe_false_negative_review.html` with severe true positives, severe false negatives, low/mid false positives, high-uncertainty severe cases, and source-stratified examples where available.
- [x] 5.6 Include failure-localization annotations in `evidence/severe_false_negative_review.html` so reviewers can distinguish upstream ROI/segmentation failures from non-separable severe grading signal.
- [x] 5.7 Write a capped first-pass figure set under `summary/figures/` covering severe threshold metrics, predicted versus observed severity, severe false-negative summary, calibration by score or threshold, source severe performance, uncertainty or ordinal prediction-set width, and reliability-label counts where estimable.
- [x] 5.8 Write `summary/artifact_manifest.json` listing every severe-aware estimator artifact by path, role, consumer, reportability, and required status.
- [x] 5.9 Write `INDEX.md` at `burden_model/severe_aware_ordinal_estimator/` explaining what to open first, artifact roles, trust status, claim boundary, figures, and manifest-listed artifacts.
- [x] 5.10 Verify the workflow writes no duplicate severe-aware estimator flat aliases under `burden_model/*` and no unmanifested severe-aware estimator artifacts.

## 6. Pipeline And Report Integration

- [x] 6.1 Export `evaluate_severe_aware_ordinal_endotheliosis_estimator` from `src/eq/quantification/__init__.py`.
- [x] 6.2 Add one bounded call from `src/eq/quantification/pipeline.py` after burden, learned ROI, and source-aware artifacts are available.
- [x] 6.3 Update `quantification_review/quantification_review.html` generation to link `burden_model/severe_aware_ordinal_estimator/INDEX.md` and show verdict-level severe-aware status.
- [x] 6.4 Link the capped severe-aware summary figures and severe false-negative review from the combined HTML review when present.
- [x] 6.5 Update `results_summary.csv` and `results_summary.md` generation with severe-aware hard-blocker, scope-limiter, output-type, severe-threshold, training/apparent, validation, testing-availability, reportability, and README-eligibility rows.
- [x] 6.6 Update `readme_results_snippet.md` generation so severe-aware results appear only when `estimator_verdict.json` marks the relevant scope eligible.
- [x] 6.7 Confirm `src/eq/quantification/pipeline.py` calls the severe-aware evaluator inside the existing `endotheliosis_quantification` execution surface so its logger events are captured by the repo-wide durable run log after P1 logging is implemented.

## 7. Tests

- [x] 7.1 Add `tests/unit/test_quantification_severe_aware_ordinal_estimator.py` covering input validation, threshold support, severe separability audit, capped candidate IDs, grouped validation, metrics-by-split labeling, severe false-negative metrics, source sensitivity, reliability labels, manifest completeness, and no flat aliases.
- [x] 7.2 Add regression coverage proving `score >= 3` is non-estimable or exploratory when independent support is insufficient.
- [x] 7.3 Add regression coverage proving severe false negatives remain present in prediction and review artifacts rather than being hidden by overall MAE.
- [x] 7.4 Add regression coverage proving high-dimensional learned or embedding features are not selected from apparent training metrics when subject-heldout validation degrades.
- [x] 7.5 Add regression coverage proving hard blockers prevent severe-aware claims when selected predictions are nonfinite or `subject_id` leakage is detected.
- [x] 7.6 Extend existing quantification pipeline tests for combined review contract and README-snippet eligibility behavior.
- [x] 7.7 Add regression coverage proving all generated severe-aware artifacts are manifest-listed and no more than the capped first-pass figure set is generated.
- [x] 7.8 Add `caplog` coverage proving the severe-aware evaluator emits function-level events for start, input/output roots, threshold support, selected candidates, verdict path, completion, and failure context.
- [x] 7.9 Add regression coverage proving the severe-aware evaluator does not create independent durable log files, attach leaked file handlers, erase an active execution-log handler, or create repo-root runtime directories.
- [x] 7.10 Extend the quantification run-config logging test so a dry-run or monkeypatched severe-aware run proves P2 events are captured through the existing `endotheliosis_quantification` durable log path rather than a P2-specific log root.

## 8. Documentation

- [x] 8.1 Update `docs/OUTPUT_STRUCTURE.md` to document `burden_model/severe_aware_ordinal_estimator/` as an indexed experimental estimator subtree.
- [x] 8.2 Update current-state user documentation only if the final estimator verdict supports a reportable scope; otherwise document where runtime reviewers can inspect the severe-aware index.
- [x] 8.3 Keep all public-facing wording predictive and grade-equivalent or severe-risk scoped; do not describe severe-aware outputs as tissue percent, closed-capillary percent, causal evidence, or external validation.
- [x] 8.4 Record final runtime findings in `audit-results.md`, including severe support, selected output type, severe false-negative behavior, source sensitivity, manifest completeness, reportable scopes, README eligibility, and next action.
- [x] 8.5 Record whether future manual annotation, a targeted Label Studio pilot, or segmentation-backbone comparison is recommended, deferred, or not supported by current evidence.
- [x] 8.6 If MedSAM or another promptable segmenter is recommended for future work, record the exact proposed comparison protocol: oracle prompts, automatic prompts, segmentation metrics, ROI-feature stability metrics, downstream severe false-negative metrics, artifact root, and success criteria.
- [x] 8.7 If nnU-Net, DeepLab, Mask2Former-style, or other upstream segmentation baselines are recommended for future work, record the exact proposed comparison protocol, dependency/environment plan, existing-mask dataset conversion plan, expected runtime, artifact root, and promotion criteria.
- [x] 8.8 If alternative segmentation appears promising, record which severity-relevant features changed, whether severe separability improved, whether `score >= 2` false negatives improved, and whether the improvement justifies a separate segmentation comparison change.

## 9. Validation

- [x] 9.1 Run `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest -q tests/unit/test_quantification_severe_aware_ordinal_estimator.py`.
- [x] 9.2 Run focused quantification tests covering burden, learned ROI, source-aware estimator, severe-aware estimator, and pipeline report integration.
- [x] 9.3 Run focused logging-contract tests affected by P2, including severe-aware `caplog` coverage and quantification run-config durable-capture coverage.
- [x] 9.4 Run `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest -q`.
- [x] 9.5 Run `ruff check` and `ruff format --check` on changed source and test files.
- [x] 9.6 Run `openspec validate p2-severe-aware-ordinal-endotheliosis-estimator --strict`.
- [x] 9.7 Run `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python scripts/check_openspec_explicitness.py openspec/changes/p2-severe-aware-ordinal-endotheliosis-estimator`.
- [x] 9.8 Run the full workflow with `PYTORCH_ENABLE_MPS_FALLBACK=1 MPLCONFIGDIR=/tmp/mpl_eq PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m eq run-config --config configs/endotheliosis_quantification.yaml`.
- [x] 9.9 Inspect the final runtime severe-aware verdict, artifact root, severe support, severe-threshold metrics, source sensitivity, summary figures, hard blockers, scope limiters, reportable scopes, README eligibility, manifest completeness, durable run log, captured P2 logger events, and next action before marking the change complete.
