## 1. Target Contract

- [x] 1.1 Replace the seven-bin quantification score contract with the six-bin rubric `[0.0, 0.5, 1.0, 1.5, 2.0, 3.0]` in the shared quantification path used by `src/eq/quantification/pipeline.py` and `src/eq/quantification/ordinal.py`.
- [x] 1.2 Add explicit score-validation helpers that fail before model fitting when a score outside the six-bin rubric is present.
- [x] 1.3 Update ordinal missing-support logic so absence of `2.5` is not a blocker and actual missing support is evaluated only against the six supported rubric values.

## 2. Burden Model Implementation

- [x] 2.1 Create `src/eq/quantification/burden.py` with constants for the six allowed scores, five cumulative thresholds, public probability column names, and the `endotheliosis_burden_0_100` formula.
- [x] 2.2 Implement biological-grouping audit output in `burden_model/grouping_audit.json`, certifying `subject_prefix` as biological biological-unit ID or deriving/using a stronger `biological_unit_id` field before cross-validation, conformal calibration, nearest-neighbor exclusion, and grouped intervals.
- [x] 2.3 Implement grouped cumulative threshold model fitting with standardized embeddings, deterministic regularized logistic threshold models, constant-threshold handling for single-class folds, threshold-support status, and captured numerical warnings.
- [x] 2.4 Write `burden_model/threshold_support.csv` with row-level and biological-group-level positive/negative support for every public threshold overall and by cohort where available.
- [x] 2.5 Implement deterministic monotonic probability correction and assert that public threshold probabilities are finite, within `[0, 1]`, and non-increasing by threshold.
- [x] 2.6 Implement normalized stage-index targets using the exact mapping `0 -> 0`, `0.5 -> 20`, `1.0 -> 40`, `1.5 -> 60`, `2.0 -> 80`, and `3.0 -> 100`; use stage-index MAE as the primary absolute-error metric.
- [x] 2.7 Implement grouped conformal score prediction sets as the primary per-image uncertainty output, including coverage and average set size overall, by cohort, and by observed score stratum where estimable.
- [x] 2.8 Implement calibrated per-image burden interval outputs as secondary derived uncertainty fields, including `burden_interval_low_0_100`, `burden_interval_high_0_100`, `burden_interval_coverage`, and `burden_interval_method`.
- [x] 2.9 Implement aggregate confidence-interval summaries with biological-unit resampling, biological-unit-level mean estimands, weighting rules, cluster counts, and non-estimable or unstable flags.
- [x] 2.10 Implement prediction evidence artifacts with threshold probability profiles and nearest scored examples in the frozen embedding space; for out-of-fold predictions, restrict neighbors to the training fold and exclude the same biological unit.
- [x] 2.11 Implement `evaluate_burden_index_table(embedding_df: pd.DataFrame, output_dir: Path, n_splits: int = 3) -> dict[str, Path]`.
- [x] 2.12 Write `burden_model/burden_predictions.csv`, `burden_model/burden_metrics.json`, `burden_model/threshold_metrics.csv`, `burden_model/threshold_support.csv`, `burden_model/calibration_bins.csv`, `burden_model/uncertainty_calibration.json`, `burden_model/grouping_audit.json`, `burden_model/prediction_explanations.csv`, `burden_model/nearest_examples.csv`, `burden_model/cohort_metrics.csv`, `burden_model/group_summary_intervals.csv`, and `burden_model/burden_model.joblib`.

## 3. Pipeline Integration

- [x] 3.1 Update `src/eq/quantification/pipeline.py::evaluate_embedding_table()` to call the burden evaluator and return burden artifacts in addition to existing ordinal artifacts.
- [x] 3.2 Preserve `ordinal_model/ordinal_predictions.csv`, `ordinal_model/ordinal_metrics.json`, `ordinal_model/ordinal_confusion_matrix.csv`, `ordinal_model/ordinal_embedding_model.pkl`, and review report outputs as comparator artifacts.
- [x] 3.3 Add direct regularized regression comparator metrics to the burden or comparison output so the report can compare cumulative threshold burden, direct stage-index regression, and ordinal/multiclass behavior.
- [x] 3.4 Update quantification review report generation so `Endotheliosis burden index (0-100)` is the leading operational output and ordinal/multiclass metrics are explicitly labeled comparator or diagnostic results.
- [x] 3.5 Update the review report to show score prediction sets, secondary per-image burden intervals, grouped aggregate confidence intervals, threshold probability profiles, nearest-example evidence, threshold-support status, grouping-audit status, and explicit caveats that these are predictive support artifacts rather than causal explanations.
- [x] 3.6 Add or refactor combined report generation so the run writes `quantification_review/quantification_review.html`, `quantification_review/review_examples.csv`, `quantification_review/results_summary.md`, `quantification_review/results_summary.csv`, and `quantification_review/readme_results_snippet.md`.
- [x] 3.7 Ensure the combined report includes an operational verdict, claim-boundary section, overall metrics, cohort/biological-unit-level summaries, comparator summaries, reviewer example gallery, artifact links, and provenance.
- [x] 3.8 Ensure review examples include representative correct predictions, high-error cases, high-uncertainty cases, cohort-stratified cases where available, and high-burden cases where support permits.
- [x] 3.9 Ensure `configs/endotheliosis_quantification.yaml` continues to run the same `workflow: endotheliosis_quantification` full-cohort command without adding a separate user-facing utility.
- [x] 3.10 Update `eq quant-endo` help text, terminal copy, README/docs references, and report language so they describe the burden-index primary model plus comparator workflow rather than an ordinal-only baseline.
- [x] 3.11 Preserve `eq prepare-quant-contract` as the pre-model contract/scored-example preparation CLI and ensure it still stops before ROI extraction, embeddings, and model fitting.

## 4. Tests

- [x] 4.1 Add unit tests for six-bin score validation, including rejection of unsupported `2.5` input and no missing-support blocker when `2.5` is absent.
- [x] 4.2 Add unit tests for cumulative threshold target construction and exact `endotheliosis_burden_0_100` calculation from threshold probabilities.
- [x] 4.3 Add unit tests proving public threshold probabilities are monotonic after correction and remain finite within `[0, 1]`.
- [x] 4.4 Add grouped-evaluation tests proving subject groups are not split across train and validation folds.
- [x] 4.5 Add unit tests for grouping-audit behavior, including date-suffixed or repeated-acquisition identifiers that must not split the same biological unit if a stronger `biological_unit_id` can be derived.
- [x] 4.6 Add unit tests for threshold-support gates by biological group and cohort.
- [x] 4.7 Add unit tests for calibrated interval schema, interval bounds, prediction-set supported values, empirical coverage metadata, and average prediction-set size metadata.
- [x] 4.8 Add unit tests for grouped aggregate confidence-interval summaries that prove repeated image rows are not resampled as independent biological units and that biological-unit-level mean burden is the primary summary.
- [x] 4.9 Add unit tests for nearest-example evidence and prediction-explanation artifact schemas, including fold-pure neighbor selection and same-biological-unit exclusion.
- [x] 4.10 Add focused pipeline tests proving full-cohort manifest quantification writes `burden_model/*` artifacts and preserves `ordinal_model/*` comparator artifacts.
- [x] 4.11 Add report-schema tests proving `quantification_review/quantification_review.html`, `review_examples.csv`, `results_summary.md`, `results_summary.csv`, and `readme_results_snippet.md` are written and contain the required burden, comparator, uncertainty, cohort-summary, example, and claim-boundary sections.
- [x] 4.12 Add CLI smoke or focused tests proving `eq quant-endo` and `eq prepare-quant-contract` still route through the same contract-first engine with the correct stop-after behavior.

## 5. Runtime Validation And Decision Record

- [x] 5.1 Run changed-file lint for `src/eq/quantification/pipeline.py`, `src/eq/quantification/ordinal.py`, `src/eq/quantification/burden.py`, and touched tests.
- [x] 5.2 Run targeted quantification tests and then the full project test suite with `/Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest -q`.
- [x] 5.3 Run the full cohort workflow with `PYTORCH_ENABLE_MPS_FALLBACK=1 MPLCONFIGDIR=/tmp/mpl_eq PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m eq run-config --config configs/endotheliosis_quantification.yaml`.
- [x] 5.4 Inspect `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/quantification_results/endotheliosis_quantification_full_cohort_transfer_p0_adjudicated/burden_model/grouping_audit.json`, `burden_model/threshold_support.csv`, `burden_model/burden_metrics.json`, `burden_model/cohort_metrics.csv`, `burden_model/uncertainty_calibration.json`, `burden_model/group_summary_intervals.csv`, `burden_model/prediction_explanations.csv`, `burden_model/nearest_examples.csv`, `ordinal_model/ordinal_metrics.json`, and the updated review report for grouping-key validity, threshold support, stability, calibration, prediction-set behavior, uncertainty interval behavior, stage-index absolute error, cohort-stratified behavior, interpretability usefulness, and comparator ranking.
- [x] 5.5 Inspect `quantification_review/quantification_review.html`, `quantification_review/review_examples.csv`, `quantification_review/results_summary.md`, `quantification_review/results_summary.csv`, and `quantification_review/readme_results_snippet.md` for reviewer usability and README/docs suitability.
- [x] 5.6 Record the refreshed operational verdict in `openspec/changes/p0-model-endotheliosis-burden-index/audit-results.md`, including whether burden-index modeling, direct regression, or ordinal/multiclass should be used for the next quantification step.
- [x] 5.7 Record concrete quantification results in `audit-results.md`, including overall metrics, cohort/biological-unit-level burden summaries, comparator metrics, uncertainty calibration, support/blocker status, and a paste-ready README/docs summary recommendation.

## 6. Final Specialist Review

- [x] 6.1 Run or delegate a stats review of the refreshed implementation and runtime artifacts, focused on estimands, grouped validation, calibration, intervals, support gates, and model-selection logic.
- [x] 6.2 Run or delegate an implementation-fidelity audit that checks code, schemas, artifact paths, and generated outputs against this OpenSpec contract.
- [x] 6.3 Run or delegate a documentation/reporting review that checks `quantification_review/*`, README/docs wording, and result summaries against the live artifact schema and claim boundary.
- [x] 6.4 Run or delegate a robustness-test review that checks leakage tests, unsupported-score tests, underpowered-threshold tests, report-schema tests, and artifact-join tests.
- [x] 6.5 Record all review findings in `audit-results.md` and fix or explicitly mark blockers before completing this change.

## 7. OpenSpec Validation

- [x] 7.1 Run `openspec validate p0-model-endotheliosis-burden-index --strict`.
- [x] 7.2 Run `python scripts/check_openspec_explicitness.py openspec/changes/p0-model-endotheliosis-burden-index` if the checker is present.
- [x] 7.3 Reopen proposal, design, specs, tasks, and audit results to confirm no unresolved `[resolve_before_apply]` questions or vague placeholder implementation surfaces remain.

## 8. Identity And Precision Readiness Follow-Through

- [x] 8.1 Replace remaining quantification identity terminology with the plain contract `subject_id`, `sample_id`, and `image_id`; remove `animal_id`, `biological_unit_id`, `biological_subject_id`, and `unified_image_id` from active quantification artifacts.
- [x] 8.2 Regenerate the runtime manifest with Dox identity and treatment group joined from `Rand_Assign.xlsx`; verify admitted rows report 60 subjects, 707 scored image replicates, and 707 image IDs.
- [x] 8.3 Add the Dox score-workbook agreement audit for `2023-11-16_all-labeled-glom-data_score-table-filtered.xlsx`; record that current admitted Dox scores match the dated long workbook.
- [x] 8.4 Make subject-heldout validation grouped by `subject_id` the primary burden validation design and write `burden_model/validation_design.json`.
- [x] 8.5 Add cohort-level stability artifacts comparing subject-heldout validation and final full-cohort fitted cohort burden summaries.
- [x] 8.6 Define report readiness using explicit cohort-level stability gates: subject-heldout versus final full-cohort fitted cohort means differ by no more than 5 burden-index points unless explicitly marked exploratory.
- [x] 8.7 Test stronger quantification signals to improve precision: ROI-derived morphology/intensity features and frozen-embedding plus ROI-feature grouped ridge screens.
- [x] 8.8 Record the precision-screen results in `audit-results.md`, including model family, feature set, validation grouping, stage-index MAE, prediction-set coverage, average prediction-set size, and cohort-level stability.
- [x] 8.9 Rerun focused tests, full tests, OpenSpec validation, and the full-cohort quantification workflow after the identity and precision-readiness updates.

## 9. Precision Candidate Expansion

- [x] 9.1 Expand `burden_model/signal_comparator_metrics.csv` into the canonical precision-candidate screen covering image-level frozen embeddings, image-level ROI scalar features, image-level embedding-plus-ROI features, subject-level global-mean baseline, subject-level ROI scalar features, subject-level frozen embeddings, subject-level embedding-plus-ROI features, and explicit calibrated target definitions.
- [x] 9.2 Write `burden_model/subject_level_candidate_predictions.csv` with one row per `subject_id` per subject-level candidate, including observed subject mean stage-index target, out-of-fold predicted subject burden, fold, cohort, feature set, model family, absolute error, and finite-output status.
- [x] 9.3 Write `burden_model/precision_candidate_summary.json` with the current primary burden metrics, best image-level candidate, best subject-level candidate, numerical-warning status, and an explicit recommendation on whether any candidate should replace the current burden model.
- [x] 9.4 Update `quantification_review/quantification_review.html`, `quantification_review/results_summary.md`, and `quantification_review/results_summary.csv` so reviewer-facing results include the precision candidate screen and do not require opening raw CSVs to see the candidate verdict.
- [x] 9.5 Add or update focused tests proving the expanded signal screen writes image-level and subject-level candidates, preserves subject-heldout validation for image-level candidates, and writes subject-level predictions and summary artifacts.
- [x] 9.6 Rerun focused quantification tests, the full test suite, OpenSpec validation, and the full-cohort quantification workflow; inspect the refreshed precision artifacts.
- [x] 9.7 Record the overnight precision-candidate plan, commands, results, what worked, what did not, and next decision in `audit-results.md`.

## 10. Follow-Up Readiness Plan

- [ ] 10.1 Add explicit `subject_burden` and `per_image_burden` track labels to the quantification review outputs, result summaries, and README snippet so subject/cohort summaries cannot be confused with individual image prediction readiness.
- [ ] 10.2 Promote the subject-level ROI aggregation path into a first-class cohort-summary candidate with subject-heldout validation, subject-level prediction artifacts, cohort/treatment summaries, grouped bootstrap confidence intervals, and track-specific readiness gates.
- [ ] 10.3 Add per-image candidate families that directly address broad prediction sets: ROI-feature cumulative-threshold modeling, embedding-plus-ROI modeling after feature diagnostics, and calibrated direct stage-index modeling with conformal residual intervals.
- [ ] 10.4 Add calibration comparisons for per-image prediction sets: global subject-heldout residual calibration, fold-specific subject-heldout calibration, score-stratified calibration where support permits, and conservative finite-sample quantiles.
- [ ] 10.5 Add feature and numerical diagnostics before candidate fitting, including nonfinite counts, zero-variance and near-zero-variance counts, feature rank or singular-value diagnostics where feasible, and warning attribution to embeddings, ROI features, or both.
- [ ] 10.6 Update readiness logic so README/docs-ready status is track-specific and cannot be true for per-image predictions while average prediction-set size remains broad or empirical coverage remains below nominal.
- [ ] 10.7 Rerun the full-cohort workflow and record in `audit-results.md` what improved, what did not improve, which track is selected for sharing, and the exact next action if either track remains exploratory.
