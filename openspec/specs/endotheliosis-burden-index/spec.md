# endotheliosis-burden-index Specification

## Purpose
Define the endotheliosis burden-index model, uncertainty artifacts, subject/cohort summary contract, and readiness gates for quantification outputs.
## Requirements
### Requirement: Burden artifacts are organized by role
Future burden-index runs SHALL organize burden artifacts into role-specific subfolders so deployed exploratory models, validation outputs, calibration outputs, summaries, evidence, candidates, diagnostics, and feature sets are unambiguous.

#### Scenario: Grouped burden output folders are written
- **WHEN** `eq run-config --config configs/endotheliosis_quantification.yaml` completes quantification
- **THEN** the run SHALL write burden artifacts under `burden_model/primary_burden_index/model/`, `burden_model/primary_burden_index/validation/`, `burden_model/primary_burden_index/calibration/`, `burden_model/primary_burden_index/summaries/`, `burden_model/primary_burden_index/evidence/`, `burden_model/primary_burden_index/candidates/`, `burden_model/primary_burden_index/diagnostics/`, and `burden_model/primary_burden_index/feature_sets/`
- **AND** candidate-screen artifacts SHALL live under `burden_model/primary_burden_index/candidates/`
- **AND** feature tables SHALL live under `burden_model/primary_burden_index/feature_sets/`
- **AND** review evidence SHALL live under `burden_model/primary_burden_index/evidence/`

#### Scenario: Candidate artifacts are not presented as deployed models
- **WHEN** quantification review reports or summaries list model artifacts
- **THEN** `burden_model/primary_burden_index/candidates/*` artifacts SHALL be labeled as candidate screens or review evidence
- **AND** only serialized selected model artifacts under `burden_model/primary_burden_index/model/` SHALL be described as model artifacts
- **AND** the report SHALL state that candidate screens do not establish deployment readiness

#### Scenario: Historical flat outputs are not silently shimmed
- **WHEN** the grouped output layout is active
- **THEN** the workflow SHALL NOT write duplicate compatibility aliases to the old flat `burden_model/*` locations
- **AND** documentation SHALL state that old flat outputs are historical runtime artifacts

### Requirement: Quantification exposes a grade-equivalent burden index
The endotheliosis quantification workflow SHALL produce a continuous `endotheliosis_burden_0_100` output derived from ordered image-level score-threshold probabilities. The output SHALL be documented and reported as a grade-equivalent burden index, not as a true pixel-area percentage of endotheliosis.

#### Scenario: Burden predictions are written for the full scored cohort
- **WHEN** `eq run-config --config configs/endotheliosis_quantification.yaml` completes the full admitted scored mask-paired cohort workflow
- **THEN** the run SHALL write `burden_model/primary_burden_index/model/burden_predictions.csv` under the quantification output root
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
- **THEN** `burden_model/primary_burden_index/model/burden_predictions.csv` SHALL contain `burden_interval_low_0_100`, `burden_interval_high_0_100`, `burden_interval_coverage`, and `burden_interval_method`
- **AND** `0 <= burden_interval_low_0_100 <= endotheliosis_burden_0_100 <= burden_interval_high_0_100 <= 100` SHALL hold within floating-point tolerance
- **AND** `burden_interval_coverage` SHALL identify the nominal coverage target used for the interval
- **AND** `burden_interval_method` SHALL identify the calibration method

#### Scenario: Prediction set scores are emitted
- **WHEN** burden predictions are written
- **THEN** `burden_model/primary_burden_index/model/burden_predictions.csv` SHALL contain `prediction_set_scores`
- **AND** each prediction set SHALL contain only supported rubric values from `[0.0, 0.5, 1.0, 1.5, 2.0, 3.0]`
- **AND** the prediction set construction method SHALL be recorded in `burden_model/primary_burden_index/calibration/uncertainty_calibration.json`
- **AND** prediction sets SHALL be calibrated with grouped subject-level conformal calibration or another grouped out-of-fold method that does not use same-biological-unit rows as independent calibration examples

#### Scenario: Uncertainty calibration is auditable
- **WHEN** burden evaluation completes
- **THEN** the run SHALL write `burden_model/primary_burden_index/calibration/uncertainty_calibration.json`
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
- **THEN** `burden_model/primary_burden_index/model/burden_predictions.csv` SHALL contain `prob_score_gt_0`, `prob_score_gt_0p5`, `prob_score_gt_1`, `prob_score_gt_1p5`, and `prob_score_gt_2`
- **AND** each probability SHALL be finite and within `[0, 1]`

#### Scenario: Public threshold probabilities are monotonic
- **WHEN** a prediction row is inspected
- **THEN** `prob_score_gt_0 >= prob_score_gt_0p5 >= prob_score_gt_1 >= prob_score_gt_1p5 >= prob_score_gt_2` SHALL hold after any deterministic monotonic correction
- **AND** the correction method SHALL be recorded in `burden_model/primary_burden_index/model/burden_metrics.json`

#### Scenario: Burden index formula is stable
- **WHEN** threshold probabilities are available for a prediction row
- **THEN** `endotheliosis_burden_0_100` SHALL equal `100 * mean([prob_score_gt_0, prob_score_gt_0p5, prob_score_gt_1, prob_score_gt_1p5, prob_score_gt_2])` within floating-point tolerance
- **AND** the value SHALL be finite and within `[0, 100]`

### Requirement: Burden validation is grouped and cohort-stratified
The burden-index model SHALL be evaluated with grouped subject-level cross-validation and SHALL report overall and cohort-stratified performance on the current full scored cohort.

#### Scenario: Manifest identity uses subject, sample, and image IDs
- **WHEN** the manifest scored-example table is prepared
- **THEN** every admitted scored mask-paired row SHALL include `subject_id`, `sample_id`, and `image_id`
- **AND** `subject_id` SHALL identify the source animal/subject without stripping experiment-date information that distinguishes different subjects
- **AND** `sample_id` SHALL identify the scored image replicate under that subject
- **AND** `image_id` SHALL identify the exact scored image/file row
- **AND** the active quantification artifacts SHALL NOT expose `animal_id`, `biological_unit_id`, `biological_subject_id`, or `unified_image_id` as supported identity columns

#### Scenario: VEGFRi/Dox identity uses the assignment workbook
- **WHEN** VEGFRi/Dox rows are imported into the manifest
- **THEN** `Rand_Assign.xlsx` SHALL provide `source_date`, `source_alt_sample_id`, `source_assignment_group`, `source_identity_workbook`, and `source_identity_sheet`
- **AND** `subject_id` SHALL be namespaced as `vegfri_dox__<experiment_date>__<source_sample_id>`
- **AND** rows such as `M1` and `M1--2023-06-12` SHALL remain different `subject_id` values
- **AND** treatment group SHALL come from the assignment workbook, not from sample-prefix inference

#### Scenario: Dox score workbook agreement is audited
- **WHEN** the Dox manifest is regenerated
- **THEN** the run SHALL write `raw_data/cohorts/vegfri_dox/metadata/score_workbook_agreement.csv`
- **AND** the run SHALL write `raw_data/cohorts/vegfri_dox/metadata/score_workbook_agreement_summary.json`
- **AND** present score mismatches against `2023-11-16_all-labeled-glom-data_score-table-filtered.xlsx` SHALL fail the manifest build
- **AND** the audit SHALL record the source workbook path so future automatic grading/inference work can trace scores to the dated long score table

#### Scenario: Subject-heldout validation is primary
- **WHEN** the scored cohort contains multiple image replicates per subject
- **THEN** burden validation SHALL group by `subject_id`
- **AND** all image replicates from the same subject SHALL stay in the same train or validation partition
- **AND** the run SHALL write `burden_model/primary_burden_index/validation/validation_design.json`

#### Scenario: Cohort-level burden stability is assessed
- **WHEN** cohort-level burden summaries are considered for README/docs or downstream analysis
- **THEN** the run SHALL compare subject-heldout cohort mean burden estimates with final full-cohort fitted cohort mean burden estimates
- **AND** the run SHALL write `burden_model/primary_burden_index/summaries/cohort_stability.csv`
- **AND** cohort mean estimates SHOULD differ by no more than 5 burden-index points unless the cohort summary is explicitly marked exploratory
- **AND** final full-cohort fitted predictions SHALL be reported separately from held-out validation predictions

#### Scenario: Biological grouping key is audited before validation
- **WHEN** burden modeling begins
- **THEN** the run SHALL write `burden_model/primary_burden_index/diagnostics/grouping_audit.json`
- **AND** the artifact SHALL identify the grouping key used for cross-validation, conformal calibration, nearest-neighbor exclusion, and grouped resampling
- **AND** the artifact SHALL state whether `subject_id` is present and used as the certified grouped-validation key
- **AND** if `subject_id` cannot be certified, the report SHALL mark the burden model as exploratory rather than operationally ready

#### Scenario: Grouped burden evaluation runs
- **WHEN** burden modeling evaluates the full scored cohort
- **THEN** cross-validation SHALL group rows by the certified biological unit grouping key
- **AND** no `subject_id` group SHALL appear in both train and validation partitions for a fold
- **AND** the number of folds SHALL be recorded in `burden_model/primary_burden_index/model/burden_metrics.json`

#### Scenario: Overall burden metrics are written
- **WHEN** burden evaluation completes
- **THEN** `burden_model/primary_burden_index/model/burden_metrics.json` SHALL include `n_examples`, `n_subject_groups`, `n_splits`, score counts, threshold positive counts, overall stage-index MAE, overall grade-scale MAE, threshold Brier scores, and numerical-stability status

#### Scenario: Threshold support gates are written
- **WHEN** burden evaluation completes
- **THEN** the run SHALL write `burden_model/primary_burden_index/validation/threshold_support.csv`
- **AND** the artifact SHALL report positive and negative support by row count and biological-group count for each public threshold overall and by cohort where available
- **AND** thresholds or cohorts with inadequate biological-group support SHALL be marked underpowered or non-estimable
- **AND** an underpowered public threshold SHALL prevent the report from calling the model operationally ready for that unsupported threshold or stratum

#### Scenario: Cohort-stratified burden metrics are written
- **WHEN** burden evaluation completes and `cohort_id` is available
- **THEN** `burden_model/primary_burden_index/summaries/cohort_metrics.csv` SHALL include separate rows for `lauren_preeclampsia` and `vegfri_dox` when those cohorts are present
- **AND** each row SHALL report at least the row count, biological-group count, stage-index MAE, grade-scale MAE, image-row mean predicted burden, and biological-unit-weighted mean predicted burden

#### Scenario: Group summary confidence intervals are written
- **WHEN** burden evaluation completes and groupable strata are available
- **THEN** `burden_model/primary_burden_index/summaries/group_summary_intervals.csv` SHALL summarize burden estimates for available strata such as `subject_prefix`, `cohort_id`, treatment group, or other validated grouping columns
- **AND** interval calculations SHALL use grouped resampling or another method that does not treat repeated image rows from the same subject as independent biological units
- **AND** the artifact SHALL label the interval type as an aggregate confidence interval, distinct from per-image prediction intervals
- **AND** the artifact SHALL record the estimand, resampling unit, weighting rule, number of clusters, and non-estimable or unstable flags
- **AND** the primary biological summary SHALL be the mean of biological-unit-level mean burdens rather than the unweighted mean over image rows

#### Scenario: Calibration artifacts are written
- **WHEN** burden evaluation completes
- **THEN** `burden_model/primary_burden_index/calibration/calibration_bins.csv` SHALL summarize predicted burden bins against observed threshold-derived burden targets
- **AND** empty bins SHALL be represented explicitly or omitted with the binning method recorded in `burden_model/primary_burden_index/model/burden_metrics.json`

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

#### Scenario: Precision candidate screen is expanded
- **WHEN** burden evaluation completes
- **THEN** `burden_model/primary_burden_index/candidates/signal_comparator_metrics.csv` SHALL include image-level frozen-embedding, ROI scalar, and embedding-plus-ROI ridge candidates validated with subject-heldout folds
- **AND** it SHALL include subject-level global-mean baseline, ROI scalar, frozen-embedding, and embedding-plus-ROI candidates validated across held-out subjects
- **AND** every candidate row SHALL identify the target level, target definition, model family, feature set, validation grouping, row count, subject count, feature count, stage-index MAE, finite-output status, backend warning count, candidate status, and intended use
- **AND** image-level candidates SHALL NOT split a `subject_id` across train and validation folds
- **AND** subject-level candidates SHALL use one aggregated row per `subject_id`

#### Scenario: Subject-level precision predictions are written
- **WHEN** subject-level precision candidates are evaluated
- **THEN** `burden_model/primary_burden_index/candidates/subject_level_candidate_predictions.csv` SHALL contain one row per held-out subject per subject-level candidate
- **AND** each row SHALL include `subject_id`, `cohort_id`, observed subject mean stage-index target, predicted subject burden, absolute error, fold, candidate identifier, feature set, model family, and prediction source

#### Scenario: Precision candidate recommendation is recorded
- **WHEN** the precision candidate screen is written
- **THEN** `burden_model/primary_burden_index/candidates/precision_candidate_summary.json` SHALL identify the current primary burden model metrics, best image-level candidate, best subject-level candidate, numerical-warning status, and recommendation
- **AND** the recommendation SHALL distinguish candidates suitable for per-image operational prediction from candidates suitable only for subject/cohort-level burden summaries
- **AND** the recommendation SHALL NOT select a candidate with nonfinite outputs

### Requirement: Subject and image burden tracks are distinct
The quantification workflow SHALL distinguish the subject/cohort burden-summary target from the per-image prediction target. A model that improves subject-level burden summaries SHALL NOT be described as solving per-image prediction uncertainty unless it is evaluated on the per-image target.

#### Scenario: Subject burden track is explicitly defined
- **WHEN** the workflow evaluates subject-level candidates
- **THEN** the target SHALL be the mean observed stage-index target per `subject_id`
- **AND** validation SHALL hold out subjects rather than image rows
- **AND** output artifacts SHALL label the prediction level as `subject`
- **AND** cohort or treatment summaries SHALL be based on subject-level values, not raw pooled image rows

#### Scenario: Per-image burden track is explicitly defined
- **WHEN** the workflow evaluates per-image candidates
- **THEN** the target SHALL remain the individual image or ROI row's stage-index target
- **AND** validation SHALL use subject-heldout folds
- **AND** the candidate SHALL report prediction-set coverage, average prediction-set size, stage-index MAE, grade-scale MAE, numerical-stability status, and cohort-stratified behavior
- **AND** the candidate SHALL NOT be called operational unless prediction sets are materially narrower than the current baseline while meeting or exceeding nominal coverage

#### Scenario: Effective sample size is reported honestly
- **WHEN** a report discusses model readiness
- **THEN** it SHALL state row count and independent subject count separately
- **AND** it SHALL explain that repeated images from the same subject increase within-subject information but do not create independent validation subjects
- **AND** it SHALL avoid claiming that the model has 707 independent examples when subject-heldout validation has 60 independent subject groups

### Requirement: Next readiness pass addresses current blockers
The next quantification readiness pass SHALL address broad per-image uncertainty, slight undercoverage, numerical warnings, and README/docs readiness as separate measurable blockers.

#### Scenario: Broad per-image uncertainty is tested directly
- **WHEN** a new per-image candidate is evaluated
- **THEN** it SHALL report average prediction-set size against the current baseline value `5.308 / 6`
- **AND** it SHALL report whether the set size narrowed without dropping empirical coverage below nominal
- **AND** candidate families SHALL include ROI-feature cumulative-threshold modeling, embedding-plus-ROI modeling after feature diagnostics, and calibrated direct stage-index modeling with conformal residual intervals unless a recorded audit explains why a family is invalid

#### Scenario: Prediction-set coverage is calibrated
- **WHEN** prediction-set coverage is below nominal
- **THEN** the workflow SHALL evaluate conformal calibration choices, including global subject-heldout residual calibration, fold-specific subject-heldout calibration, score-stratified calibration where support permits, and conservative finite-sample quantiles
- **AND** the report SHALL identify which calibration choice was selected and why
- **AND** the report SHALL include overall, cohort-level, and observed-score-stratum coverage where estimable

#### Scenario: Backend matrix warnings are diagnosed
- **WHEN** a candidate emits backend matrix-operation warnings
- **THEN** the workflow SHALL write feature diagnostics that include nonfinite feature counts, zero-variance feature counts, near-zero-variance feature counts, feature count, row count, subject count, and rank or singular-value diagnostics where feasible
- **AND** the workflow SHALL record whether warnings are associated with high-dimensional embeddings, ROI scalar features, or both
- **AND** a candidate with repeated numerical warnings SHALL remain exploratory unless a documented preprocessing or model-family fix removes or explains the warnings

#### Scenario: README/docs readiness is track-specific
- **WHEN** `quantification_review/readme_results_snippet.md` is generated
- **THEN** it SHALL state whether the shareable result, if any, is a subject/cohort burden-summary result or a per-image prediction result
- **AND** it SHALL NOT present subject-level summary performance as evidence that individual image scores are operationally precise
- **AND** it SHALL NOT mark results README/docs-ready unless the selected track passes its own coverage, stability, numerical, and claim-boundary gates

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
- **AND** it SHALL include cohort and biological-unit-level burden summaries using biological-unit-weighted means and uncertainty intervals where estimable
- **AND** it SHALL include comparator summaries for burden, direct stage-index regression, and ordinal/multiclass outputs
- **AND** it SHALL include the expanded precision candidate screen with image-level and subject-level candidate metrics

#### Scenario: Morphology feature summary is included when available
- **WHEN** morphology features are generated
- **THEN** `quantification_review/quantification_review.html` SHALL include a morphology feature summary section
- **AND** it SHALL link to `burden_model/primary_burden_index/evidence/morphology_feature_review/feature_review.html`
- **AND** it SHALL show whether operator adjudication was provided

#### Scenario: Subject/cohort and image-level tracks are separated
- **WHEN** morphology candidates are reported
- **THEN** the review SHALL separate `subject_burden` readiness from `per_image_burden` readiness
- **AND** subject/cohort readiness SHALL NOT be used as evidence that individual image predictions are operationally precise

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
- **THEN** the run SHALL write `burden_model/primary_burden_index/evidence/prediction_explanations.csv`
- **AND** each row SHALL include the row identifier, threshold probabilities, burden estimate, prediction interval fields, prediction set scores, and source image or ROI provenance
- **AND** the artifact SHALL label threshold probabilities as model evidence rather than causal explanation

#### Scenario: Nearest scored examples are written
- **WHEN** burden evaluation completes
- **THEN** the run SHALL write `burden_model/primary_burden_index/evidence/nearest_examples.csv`
- **AND** each evaluated row SHALL include at least the nearest scored example identifiers, nearest example scores, distances in the frozen embedding space, cohort or source identifiers when available, and path provenance needed to inspect the examples
- **AND** nearest-neighbor search SHALL be computed within the same frozen-embedding feature space used by the burden model
- **AND** nearest examples for out-of-fold predictions SHALL come only from the corresponding training fold and SHALL exclude the same biological unit
- **AND** nearest-example rows SHALL include `cohort_id` and `lane_assignment` when available

#### Scenario: Threshold-level calibration is reported
- **WHEN** burden evaluation completes
- **THEN** threshold-level calibration summaries SHALL be written to `burden_model/primary_burden_index/validation/threshold_metrics.csv` or another documented burden artifact
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
- **THEN** the run SHALL write `burden_model/primary_burden_index/model/burden_model.joblib`
- **AND** the serialized artifact SHALL include the allowed score values, threshold values, embedding column names, scaler or preprocessing state, final fitted threshold models, and model metadata

#### Scenario: Burden metrics identify the implementation
- **WHEN** `burden_model/primary_burden_index/model/burden_metrics.json` is inspected
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

### Requirement: Burden report includes learned ROI evidence
The combined quantification report SHALL include learned ROI candidate evidence when learned ROI artifacts are generated.

#### Scenario: Learned ROI section is included
- **WHEN** `burden_model/learned_roi/candidates/learned_roi_candidate_summary.json` exists
- **THEN** `quantification_review/quantification_review.html` SHALL include a learned ROI quantification section
- **AND** it SHALL link to `burden_model/learned_roi/evidence/learned_roi_review.html`
- **AND** it SHALL show per-image readiness, subject/cohort readiness, selected track if any, blockers, and claim boundary
- **AND** it SHALL show cohort-confounding diagnostic status and whether cohort diagnostics blocked any track

#### Scenario: README snippet uses selected learned track only when ready
- **WHEN** `quantification_review/readme_results_snippet.md` is generated
- **THEN** it SHALL include learned ROI results only if `learned_roi_candidate_summary.json` marks a track README/docs-ready
- **AND** it SHALL identify whether the selected result is a per-image prediction result or a subject/cohort burden-summary result
- **AND** it SHALL NOT present a subject/cohort result as evidence that individual image predictions are operationally precise
- **AND** it SHALL NOT include learned ROI results when readiness passed by stage-index MAE alone without passing uncertainty, numerical-stability, ordinal/grade-scale, and cohort-diagnostic gates

### Requirement: Learned ROI outputs use contained burden subtree layout
Learned ROI artifacts SHALL live under the learned ROI subtree of the burden output contract.

#### Scenario: Learned ROI output folders are written
- **WHEN** learned ROI evaluation runs
- **THEN** first-read learned ROI artifacts SHALL be written under `burden_model/learned_roi/summary/`
- **AND** typed learned ROI artifacts SHALL be written under `burden_model/learned_roi/validation/`, `burden_model/learned_roi/calibration/`, `burden_model/learned_roi/summaries/`, `burden_model/learned_roi/evidence/`, `burden_model/learned_roi/candidates/`, `burden_model/learned_roi/diagnostics/`, and `burden_model/learned_roi/feature_sets/`
- **AND** the workflow SHALL NOT write duplicate learned ROI compatibility aliases to flat `burden_model/*` locations

### Requirement: Learned ROI burden claims remain predictive
Learned ROI burden outputs SHALL preserve the current claim boundary for endotheliosis burden.

#### Scenario: Learned ROI report avoids mechanistic overclaiming
- **WHEN** learned ROI results are reported
- **THEN** the report SHALL describe predictions as grade-equivalent endotheliosis burden estimates from learned ROI features
- **AND** it SHALL NOT describe them as true tissue-area percent, closed-capillary percent, causal explanation, or validated mechanistic endotheliosis measurement
- **AND** visual evidence SHALL be labeled as predictive support evidence rather than proof of histologic mechanism
- **AND** the report SHALL state row count and independent `subject_id` count separately when reporting learned ROI validation or burden summaries

### Requirement: Combined burden reports include source-aware estimator verdicts
The combined quantification report SHALL include source-aware estimator verdict information when source-aware estimator artifacts are generated.

#### Scenario: Source-aware verdict appears in combined review
- **WHEN** `burden_model/source_aware_estimator/summary/estimator_verdict.json` exists
- **THEN** `quantification_review/quantification_review.html` SHALL include a source-aware estimator section
- **AND** it SHALL link to `burden_model/source_aware_estimator/INDEX.md`
- **AND** it SHALL show selected image candidate, selected subject candidate, upstream ROI adequacy status, hard blockers, scope limiters, reportable scopes, non-reportable scopes, and claim boundary
- **AND** it SHALL link to `burden_model/source_aware_estimator/summary/metrics_by_split.csv` when present
- **AND** it SHALL link to or embed the capped source-aware summary figures when present
- **AND** it SHALL prioritize the estimator verdict over internal candidate metrics

#### Scenario: Results summaries include verdict-level rows
- **WHEN** source-aware estimator artifacts exist
- **THEN** `quantification_review/results_summary.csv` SHALL include verdict-level source-aware rows for hard-blocker status, scope-limiter status, image-reportable status, subject-reportable status, and README-snippet eligibility
- **AND** `quantification_review/results_summary.csv` SHALL include an upstream ROI adequacy row when `diagnostics/upstream_roi_adequacy.json` exists
- **AND** `quantification_review/results_summary.csv` SHALL include source-aware training/apparent, validation, and testing-availability summary rows when `summary/metrics_by_split.csv` exists
- **AND** `quantification_review/results_summary.md` SHALL summarize source-aware behavior in human-readable prose
- **AND** neither summary SHALL require the reader to inspect `internal/candidate_metrics.csv` to know whether the estimator is usable for the current claim

### Requirement: README snippets only include source-aware results when explicitly reportable
Source-aware estimator outputs SHALL enter README snippets only when the estimator verdict marks the relevant scope reportable.

#### Scenario: README snippet excludes non-reportable estimator scopes
- **WHEN** `estimator_verdict.json` marks `readme_snippet_eligible` as false
- **THEN** `quantification_review/readme_results_snippet.md` SHALL NOT include source-aware estimator results
- **AND** the combined review SHALL still link to the source-aware estimator index for runtime review

#### Scenario: README snippet includes claim boundary when eligible
- **WHEN** `estimator_verdict.json` marks `readme_snippet_eligible` as true
- **THEN** `quantification_review/readme_results_snippet.md` SHALL state whether the reportable result is image-level, subject-level, or aggregate-only
- **AND** it SHALL state that the estimate is a grade-equivalent burden estimate calibrated to the current scored MR TIFF/ROI data
- **AND** it SHALL NOT describe the result as external validation, causal evidence, closed-capillary percent, or true tissue-area percent

### Requirement: Burden artifact ergonomics are enforced for experimental estimators
Experimental burden estimators SHALL be organized so the first artifact a reader opens explains the output tree and trust status.

#### Scenario: Experimental estimator index is required
- **WHEN** any experimental burden estimator writes more than one artifact subtree under `burden_model/`
- **THEN** it SHALL include an `INDEX.md` at that estimator subtree root
- **AND** the index SHALL identify human-facing, diagnostic, prediction, evidence, and internal artifacts
- **AND** the index SHALL state what can be trusted, what is limited, and what should not be reported

#### Scenario: Experimental internals are separated from summaries
- **WHEN** experimental estimator candidate metrics, diagnostic feature tables, or exhaustive candidate outputs are written
- **THEN** they SHALL live under an `internal/` or `diagnostics/` role folder
- **AND** human-facing verdict files SHALL live under `summary/`
- **AND** the workflow SHALL NOT duplicate the same experimental table into multiple role folders without a named consumer
