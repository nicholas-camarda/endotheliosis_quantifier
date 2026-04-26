## ADDED Requirements

### Requirement: Quantification exposes a grade-equivalent burden index
The endotheliosis quantification workflow SHALL produce a continuous `endotheliosis_burden_0_100` output derived from ordered image-level score-threshold probabilities. The output SHALL be documented and reported as a grade-equivalent burden index, not as a true pixel-area percentage of endotheliosis.

#### Scenario: Burden predictions are written for the full scored cohort
- **WHEN** `eq run-config --config configs/endotheliosis_quantification.yaml` completes the full admitted scored mask-paired cohort workflow
- **THEN** the run SHALL write `burden_model/burden_predictions.csv` under the quantification output root
- **AND** the table SHALL contain one row per evaluated embedding row
- **AND** the table SHALL contain `endotheliosis_burden_0_100`
- **AND** the table SHALL contain `burden_interval_low_0_100`, `burden_interval_high_0_100`, `burden_interval_coverage`, `burden_interval_method`, and `prediction_set_scores`
- **AND** the table SHALL preserve row-identifying fields needed to join back to the scored example, including `subject_image_id`, `subject_prefix`, `cohort_id` when available, `lane_assignment` when available, `score`, and image or mask path provenance columns present in the embedding table

#### Scenario: Burden output is not represented as true tissue percent
- **WHEN** the quantification review report describes `endotheliosis_burden_0_100`
- **THEN** the report SHALL call it `Endotheliosis burden index (0-100)`
- **AND** the report SHALL state that it is derived from manually scored image-level grade thresholds
- **AND** the report SHALL NOT describe it as pixel-level percent endotheliosis, tissue-area percent, or externally validated clinical percentage

#### Scenario: Burden scale is a normalized ordinal stage index
- **WHEN** the burden-index formula is documented or validated
- **THEN** the workflow SHALL define the estimand as normalized expected crossed rubric stages
- **AND** the observed-score-to-index mapping SHALL be `0 -> 0`, `0.5 -> 20`, `1.0 -> 40`, `1.5 -> 60`, `2.0 -> 80`, and `3.0 -> 100`
- **AND** stage-index MAE against this mapping SHALL be the primary absolute-error metric
- **AND** raw grade-scale MAE SHALL be secondary

### Requirement: Burden predictions expose calibrated per-image uncertainty
The burden-index workflow SHALL provide calibrated uncertainty for each image-level burden prediction. Score prediction sets SHALL be the primary uncertainty object for the six-level ordered rubric; continuous burden bounds SHALL be secondary derived uncertainty intervals. These outputs SHALL be reported as prediction sets, prediction intervals, or uncertainty intervals for individual predictions, not as classical confidence intervals around a population mean.

#### Scenario: Per-image prediction intervals are emitted
- **WHEN** burden predictions are written
- **THEN** `burden_model/burden_predictions.csv` SHALL contain `burden_interval_low_0_100`, `burden_interval_high_0_100`, `burden_interval_coverage`, and `burden_interval_method`
- **AND** `0 <= burden_interval_low_0_100 <= endotheliosis_burden_0_100 <= burden_interval_high_0_100 <= 100` SHALL hold within floating-point tolerance
- **AND** `burden_interval_coverage` SHALL identify the nominal coverage target used for the interval
- **AND** `burden_interval_method` SHALL identify the calibration method

#### Scenario: Prediction set scores are emitted
- **WHEN** burden predictions are written
- **THEN** `burden_model/burden_predictions.csv` SHALL contain `prediction_set_scores`
- **AND** each prediction set SHALL contain only supported rubric values from `[0.0, 0.5, 1.0, 1.5, 2.0, 3.0]`
- **AND** the prediction set construction method SHALL be recorded in `burden_model/uncertainty_calibration.json`
- **AND** prediction sets SHALL be calibrated with grouped subject-level conformal calibration or another grouped out-of-fold method that does not use same-animal rows as independent calibration examples

#### Scenario: Uncertainty calibration is auditable
- **WHEN** burden evaluation completes
- **THEN** the run SHALL write `burden_model/uncertainty_calibration.json`
- **AND** the artifact SHALL record the calibration method, grouping column, nominal coverage, empirical coverage where estimable, residual or conformity score summary, and any folds or thresholds where calibration was underpowered
- **AND** the artifact SHALL report prediction-set empirical coverage and average set size overall, by cohort where estimable, and by observed score stratum where estimable

### Requirement: Burden model uses the current six-bin score rubric
The burden-index model SHALL use the current image-level score rubric `[0.0, 0.5, 1.0, 1.5, 2.0, 3.0]` and SHALL reject unsupported target scores rather than silently coercing them.

#### Scenario: Six-bin rubric is accepted
- **WHEN** the evaluated embedding table contains scores drawn from `0`, `0.5`, `1`, `1.5`, `2`, and `3`
- **THEN** burden modeling SHALL treat those as the complete supported rubric for the current workflow
- **AND** absence of `2.5` SHALL NOT create a missing target-class blocker

#### Scenario: Unsupported score is present
- **WHEN** the evaluated embedding table contains a score outside `[0.0, 0.5, 1.0, 1.5, 2.0, 3.0]`
- **THEN** the quantification workflow SHALL fail before model fitting
- **AND** the failure message SHALL identify the unsupported score value and the supported rubric

### Requirement: Burden model predicts cumulative threshold probabilities
The burden-index model SHALL predict ordered threshold probabilities for the explicit thresholds `score > 0`, `score > 0.5`, `score > 1`, `score > 1.5`, and `score > 2`.

#### Scenario: Threshold probability columns are emitted
- **WHEN** burden predictions are written
- **THEN** `burden_model/burden_predictions.csv` SHALL contain `prob_score_gt_0`, `prob_score_gt_0p5`, `prob_score_gt_1`, `prob_score_gt_1p5`, and `prob_score_gt_2`
- **AND** each probability SHALL be finite and within `[0, 1]`

#### Scenario: Public threshold probabilities are monotonic
- **WHEN** a prediction row is inspected
- **THEN** `prob_score_gt_0 >= prob_score_gt_0p5 >= prob_score_gt_1 >= prob_score_gt_1p5 >= prob_score_gt_2` SHALL hold after any deterministic monotonic correction
- **AND** the correction method SHALL be recorded in `burden_model/burden_metrics.json`

#### Scenario: Burden index formula is stable
- **WHEN** threshold probabilities are available for a prediction row
- **THEN** `endotheliosis_burden_0_100` SHALL equal `100 * mean([prob_score_gt_0, prob_score_gt_0p5, prob_score_gt_1, prob_score_gt_1p5, prob_score_gt_2])` within floating-point tolerance
- **AND** the value SHALL be finite and within `[0, 100]`

### Requirement: Burden validation is grouped and cohort-stratified
The burden-index model SHALL be evaluated with grouped subject-level cross-validation and SHALL report overall and cohort-stratified performance on the current full scored cohort.

#### Scenario: Biological grouping key is audited before validation
- **WHEN** burden modeling begins
- **THEN** the run SHALL write `burden_model/grouping_audit.json`
- **AND** the artifact SHALL identify the grouping key used for cross-validation, conformal calibration, nearest-neighbor exclusion, and grouped resampling
- **AND** the artifact SHALL state whether `subject_prefix` is certified as the biological animal identifier or whether a stronger `animal_id` grouping field is used
- **AND** if the biological grouping key cannot be certified, the report SHALL mark the burden model as exploratory rather than operationally ready

#### Scenario: Grouped burden evaluation runs
- **WHEN** burden modeling evaluates the full scored cohort
- **THEN** cross-validation SHALL group rows by the certified biological animal grouping key
- **AND** no biological animal group SHALL appear in both train and validation partitions for a fold
- **AND** the number of folds SHALL be recorded in `burden_model/burden_metrics.json`

#### Scenario: Overall burden metrics are written
- **WHEN** burden evaluation completes
- **THEN** `burden_model/burden_metrics.json` SHALL include `n_examples`, `n_subject_groups`, `n_splits`, score counts, threshold positive counts, overall stage-index MAE, overall grade-scale MAE, threshold Brier scores, and numerical-stability status

#### Scenario: Threshold support gates are written
- **WHEN** burden evaluation completes
- **THEN** the run SHALL write `burden_model/threshold_support.csv`
- **AND** the artifact SHALL report positive and negative support by row count and biological-group count for each public threshold overall and by cohort where available
- **AND** thresholds or cohorts with inadequate biological-group support SHALL be marked underpowered or non-estimable
- **AND** an underpowered public threshold SHALL prevent the report from calling the model operationally ready for that unsupported threshold or stratum

#### Scenario: Cohort-stratified burden metrics are written
- **WHEN** burden evaluation completes and `cohort_id` is available
- **THEN** `burden_model/cohort_metrics.csv` SHALL include separate rows for `lauren_preeclampsia` and `vegfri_dox` when those cohorts are present
- **AND** each row SHALL report at least the row count, biological-group count, stage-index MAE, grade-scale MAE, image-row mean predicted burden, and animal-weighted mean predicted burden

#### Scenario: Group summary confidence intervals are written
- **WHEN** burden evaluation completes and groupable strata are available
- **THEN** `burden_model/group_summary_intervals.csv` SHALL summarize burden estimates for available strata such as `subject_prefix`, `cohort_id`, treatment group, or other validated grouping columns
- **AND** interval calculations SHALL use grouped resampling or another method that does not treat repeated image rows from the same subject as independent biological units
- **AND** the artifact SHALL label the interval type as an aggregate confidence interval, distinct from per-image prediction intervals
- **AND** the artifact SHALL record the estimand, resampling unit, weighting rule, number of clusters, and non-estimable or unstable flags
- **AND** the primary biological summary SHALL be the mean of animal-level mean burdens rather than the unweighted mean over image rows

#### Scenario: Calibration artifacts are written
- **WHEN** burden evaluation completes
- **THEN** `burden_model/calibration_bins.csv` SHALL summarize predicted burden bins against observed threshold-derived burden targets
- **AND** empty bins SHALL be represented explicitly or omitted with the binning method recorded in `burden_model/burden_metrics.json`

### Requirement: Burden model comparison is explicit
The quantification workflow SHALL compare the cumulative threshold burden model against direct regression and ordinal/multiclass comparator outputs before presenting an operational verdict.

#### Scenario: Comparator metrics are available
- **WHEN** the full-cohort quantification report is generated
- **THEN** it SHALL include burden-index metrics, direct regression comparator metrics, and ordinal/multiclass comparator metrics
- **AND** it SHALL label which model family is the primary operational candidate for downstream quantification

#### Scenario: Operational selection is evidence-based
- **WHEN** the report states a model verdict
- **THEN** the verdict SHALL be based on refreshed full-cohort validation artifacts from the current run
- **AND** the verdict SHALL distinguish numerical stability, discrimination or ranking, calibration, absolute error, and cohort-stratified behavior
- **AND** the report SHALL NOT promote a model solely because the pipeline completed

### Requirement: Quantification report supports reviewer inspection and result sharing
The quantification workflow SHALL write a combined reviewer-facing report that makes burden, ordinal comparator, direct-regression comparator, cohort summaries, and example-level evidence inspectable from one place. The existing ordinal review report may remain as a comparator artifact, but it SHALL NOT be the only human-readable report after this change.

#### Scenario: Combined quantification review report is written
- **WHEN** the full-cohort quantification workflow completes with `stop_after: model`
- **THEN** the run SHALL write `quantification_review/quantification_review.html`
- **AND** it SHALL write `quantification_review/review_examples.csv`
- **AND** it SHALL include links or references to the relevant `burden_model/*`, `ordinal_model/*`, ROI, embedding, and scored-example artifacts
- **AND** it SHALL identify whether the rows shown are held-out fold predictions, final fitted predictions, or exploratory examples

#### Scenario: Report leads with operational verdict and claim boundary
- **WHEN** `quantification_review/quantification_review.html` is opened
- **THEN** the first report section SHALL state the selected operational model status, blocker status, and whether the quantification results are ready for README/docs sharing
- **AND** it SHALL state that the burden output is a predictive ordinal stage-burden index from image-level grades, not pixel-level percent tissue involvement
- **AND** it SHALL distinguish descriptive, predictive, and non-causal interpretation boundaries

#### Scenario: Report includes summary tables for readers
- **WHEN** the combined report is generated
- **THEN** it SHALL include overall row count, biological-group count, score distribution, threshold-support status, stage-index MAE, grade-scale MAE, prediction-set empirical coverage, average prediction-set size, and numerical-stability status
- **AND** it SHALL include cohort and animal-level burden summaries using animal-weighted means and uncertainty intervals where estimable
- **AND** it SHALL include comparator summaries for burden, direct stage-index regression, and ordinal/multiclass outputs

#### Scenario: Report includes reviewer example gallery
- **WHEN** review examples are selected
- **THEN** the selection SHALL include representative correct predictions, high-error examples, high-uncertainty examples, cohort-stratified examples where available, and clinically or scientifically relevant high-burden examples where support permits
- **AND** each example SHALL show raw image with ROI box, mask overlay, ROI crop, observed score, burden prediction with uncertainty, prediction set, ordinal comparator prediction, threshold probability profile, nearest scored examples, cohort, biological group, fold, and provenance paths
- **AND** the report SHALL label nearest examples and threshold profiles as predictive support evidence rather than mechanistic explanation

#### Scenario: README/docs-ready result summaries are written
- **WHEN** the refreshed full-cohort run finishes
- **THEN** the run SHALL write `quantification_review/results_summary.md`
- **AND** it SHALL write `quantification_review/results_summary.csv`
- **AND** it SHALL write `quantification_review/readme_results_snippet.md`
- **AND** these summaries SHALL use values from the refreshed runtime artifacts generated by the current run
- **AND** the README snippet SHALL include only claims allowed by the model verdict and claim-boundary status

### Requirement: Burden predictions include model-evidence artifacts
The burden-index workflow SHALL produce evidence artifacts that help reviewers understand why a row received a particular burden estimate without claiming causal or mechanistic explanation.

#### Scenario: Threshold-profile explanations are written
- **WHEN** burden predictions are written
- **THEN** the run SHALL write `burden_model/prediction_explanations.csv`
- **AND** each row SHALL include the row identifier, threshold probabilities, burden estimate, prediction interval fields, prediction set scores, and source image or ROI provenance
- **AND** the artifact SHALL label threshold probabilities as model evidence rather than causal explanation

#### Scenario: Nearest scored examples are written
- **WHEN** burden evaluation completes
- **THEN** the run SHALL write `burden_model/nearest_examples.csv`
- **AND** each evaluated row SHALL include at least the nearest scored example identifiers, nearest example scores, distances in the frozen embedding space, cohort or source identifiers when available, and path provenance needed to inspect the examples
- **AND** nearest-neighbor search SHALL be computed within the same frozen-embedding feature space used by the burden model
- **AND** nearest examples for out-of-fold predictions SHALL come only from the corresponding training fold and SHALL exclude the same biological animal
- **AND** nearest-example rows SHALL include `cohort_id` and `lane_assignment` when available

#### Scenario: Threshold-level calibration is reported
- **WHEN** burden evaluation completes
- **THEN** threshold-level calibration summaries SHALL be written to `burden_model/threshold_metrics.csv` or another documented burden artifact
- **AND** pooled burden calibration SHALL NOT be the only calibration evidence used for model selection

#### Scenario: Visual attribution is optional and caveated
- **WHEN** the report includes visual attribution or saliency panels
- **THEN** the panels SHALL be generated from the same ROI/embedding path used for quantification
- **AND** the report SHALL label them as heuristic model-support visualizations
- **AND** the report SHALL NOT claim that attribution proves the biological mechanism of endotheliosis

### Requirement: Burden artifacts carry model provenance
Burden-index artifacts SHALL record enough provenance to identify the model family, score rubric, thresholds, grouped split configuration, feature columns, calibration or monotonic correction method, and source embedding table shape.

#### Scenario: Final burden model is serialized
- **WHEN** burden modeling completes
- **THEN** the run SHALL write `burden_model/burden_model.joblib`
- **AND** the serialized artifact SHALL include the allowed score values, threshold values, embedding column names, scaler or preprocessing state, final fitted threshold models, and model metadata

#### Scenario: Burden metrics identify the implementation
- **WHEN** `burden_model/burden_metrics.json` is inspected
- **THEN** it SHALL identify `eq.quantification.burden` as the canonical module
- **AND** it SHALL identify the estimator family, regularization settings, random state, monotonic correction method, and any numerical warning messages captured during fitting or prediction

### Requirement: Quantification CLI roles remain clear
The burden-index implementation SHALL integrate into the existing maintained quantification workflow and direct CLI surface without introducing a duplicate user-facing command or compatibility alias.

#### Scenario: YAML workflow remains the reproducible front door
- **WHEN** `eq run-config --config configs/endotheliosis_quantification.yaml` completes with `stop_after: model`
- **THEN** it SHALL run the contract-first quantification engine
- **AND** it SHALL write the new `burden_model/*` artifacts and the retained `ordinal_model/*` comparator artifacts under the configured quantification output root

#### Scenario: Direct quantification CLI remains one-shot
- **WHEN** `eq quant-endo` is run with explicit `--data-dir`, `--segmentation-model`, and `--output-dir`
- **THEN** it SHALL run the same quantification engine used by the YAML workflow
- **AND** its help text and terminal copy SHALL describe the current burden-index plus comparator workflow rather than an ordinal-only baseline

#### Scenario: Contract preparation CLI remains pre-model only
- **WHEN** `eq prepare-quant-contract` is run
- **THEN** it SHALL stop after contract/scored-example artifact generation
- **AND** it SHALL NOT fit burden, direct-regression, or ordinal comparator models
- **AND** its help text SHALL make clear that it prepares score-linked image/mask contract artifacts for later quantification

### Requirement: Final implementation review is independent and recorded
The change SHALL end with a specialist review packet that checks statistical validity, implementation fidelity, reporting clarity, documentation consistency, and robustness before the OpenSpec change is marked complete.

#### Scenario: Specialist review packet is recorded
- **WHEN** implementation, tests, and the full-cohort runtime run are complete
- **THEN** `openspec/changes/p0-model-endotheliosis-burden-index/audit-results.md` SHALL include findings from a stats review, implementation audit, documentation/reporting review, and robustness-test review
- **AND** the review SHALL explicitly inspect the combined quantification report structure and the README/docs-ready result summaries
- **AND** the review SHALL identify any high-severity blockers or state that none were found

#### Scenario: Review findings affect completion
- **WHEN** a high-severity review finding affects statistical validity, artifact fidelity, report interpretation, or test coverage
- **THEN** the implementation SHALL fix the issue or mark the p0 change blocked
- **AND** the change SHALL NOT be marked complete only because tests or the workflow command finished
