## ADDED Requirements

### Requirement: Learned ROI provider availability is audited
The quantification workflow SHALL audit learned ROI feature providers before fitting learned ROI candidates.

#### Scenario: Provider audit is written
- **WHEN** `eq run-config --config configs/endotheliosis_quantification.yaml` reaches learned ROI feature extraction
- **THEN** the run SHALL write `burden_model/learned_roi/diagnostics/provider_audit.json`
- **AND** the artifact SHALL include provider IDs, availability status, package/module provenance, model/weight provenance where available, failure messages where unavailable, and whether network access or downloads were required
- **AND** candidates SHALL use only providers marked `available`

#### Scenario: Required baseline providers are evaluated
- **WHEN** provider audit runs
- **THEN** it SHALL evaluate `current_glomeruli_encoder` and `simple_roi_qc`
- **AND** it SHALL evaluate optional local providers `torchvision_resnet18_imagenet` and `timm_dino_or_vit` when their packages and weights are usable without network downloads
- **AND** unavailable optional providers SHALL be recorded as unavailable rather than silently substituted

### Requirement: Learned ROI feature tables are written
The quantification workflow SHALL write learned ROI feature tables from the existing ROI image/mask/scored-example contract.

#### Scenario: Learned feature table is written
- **WHEN** at least one learned ROI provider is available
- **THEN** the run SHALL write `burden_model/learned_roi/feature_sets/learned_roi_features.csv`
- **AND** the table SHALL contain one row per valid ROI row and provider-feature columns prefixed with `learned_<provider_id>_`
- **AND** the table SHALL preserve `subject_id`, `sample_id`, `image_id`, `subject_image_id`, `cohort_id`, `score`, `roi_image_path`, `roi_mask_path`, `raw_image_path`, and `raw_mask_path` when present
- **AND** all provider feature columns SHALL be finite or the provider SHALL be excluded with a recorded reason

#### Scenario: Feature metadata is written
- **WHEN** learned ROI features are written
- **THEN** the run SHALL write `burden_model/learned_roi/feature_sets/learned_roi_feature_metadata.json`
- **AND** the metadata SHALL record provider IDs, feature counts, preprocessing transforms, image-size policy, mask-use policy, package versions where available, and feature table shape

#### Scenario: Feature diagnostics are written
- **WHEN** learned ROI features are written
- **THEN** the run SHALL write `burden_model/learned_roi/diagnostics/learned_roi_feature_diagnostics.json`
- **AND** the diagnostics SHALL include row count, subject count, provider count, feature count by provider, nonfinite counts, zero-variance counts, near-zero-variance counts, missingness counts, feature ranges, and rank or singular-value diagnostics where feasible

### Requirement: Learned ROI candidates use grouped validation
Learned ROI candidates SHALL be evaluated with validation that respects subject identity.

#### Scenario: Image-level learned candidates use subject-heldout folds
- **WHEN** image-level learned ROI candidates are fit
- **THEN** validation SHALL group by `subject_id`
- **AND** no `subject_id` SHALL appear in both train and validation partitions for a fold
- **AND** candidate metrics SHALL report row count, subject count, provider ID, feature count, stage-index MAE, grade-scale MAE, prediction-set coverage, average prediction-set size, numerical-stability status, and cohort-stratified behavior where estimable

#### Scenario: Subject-level learned candidates use one row per subject
- **WHEN** subject/cohort learned ROI candidates are fit
- **THEN** features SHALL be aggregated to one row per `subject_id`
- **AND** validation SHALL hold out subjects
- **AND** outputs SHALL report subject-level MAE, cohort summaries, grouped bootstrap confidence intervals, cohort stability, and whether the result supports subject/cohort summaries only

### Requirement: Learned ROI uncertainty is calibrated
Learned ROI candidates SHALL expose calibrated uncertainty before any operational claim.

#### Scenario: Prediction sets are written for image-level candidates
- **WHEN** image-level learned ROI predictions are written
- **THEN** the run SHALL write `burden_model/learned_roi/validation/learned_roi_predictions.csv`
- **AND** each row SHALL include predicted score or burden, prediction set scores, burden interval fields when applicable, fold, provider ID, candidate ID, and prediction source
- **AND** prediction sets SHALL be calibrated with grouped subject-heldout conformal calibration or another grouped out-of-fold method

#### Scenario: Calibration summary is written
- **WHEN** learned ROI candidate evaluation completes
- **THEN** the run SHALL write `burden_model/learned_roi/calibration/learned_roi_calibration.json`
- **AND** the artifact SHALL report nominal coverage, empirical coverage, average prediction-set size, residual or conformity score summaries, and coverage by cohort and observed score where estimable

### Requirement: Learned ROI review evidence is generated
The workflow SHALL generate reviewer-facing learned ROI evidence without claiming mechanistic proof.

#### Scenario: Learned ROI review report is written
- **WHEN** learned ROI candidate evaluation completes
- **THEN** the run SHALL write `burden_model/learned_roi/evidence/learned_roi_review.html`
- **AND** it SHALL link selected high-error, high-uncertainty, representative correct, high-burden, and cohort-stratified examples where available
- **AND** each example SHALL show ROI crop, mask or mask outline, observed score, prediction, uncertainty, provider ID, fold, nearest scored examples, and provenance paths

#### Scenario: Nearest examples are written
- **WHEN** learned ROI review evidence is generated
- **THEN** the run SHALL write `burden_model/learned_roi/evidence/learned_roi_nearest_examples.csv`
- **AND** nearest examples for held-out predictions SHALL exclude the same `subject_id`
- **AND** nearest-example distances SHALL be computed in the same learned feature space used by the candidate

#### Scenario: Attribution is caveated
- **WHEN** saliency, attention, or other visual attribution artifacts are generated
- **THEN** they SHALL be written under `burden_model/learned_roi/evidence/assets/`
- **AND** the report SHALL label them as heuristic model-support visualizations
- **AND** the report SHALL NOT claim that attribution proves closed-lumen or causal endotheliosis mechanism

### Requirement: Learned ROI candidate summary selects only ready tracks
Learned ROI candidate summaries SHALL separate exploratory candidates from selected operational tracks.

#### Scenario: Candidate summary is written
- **WHEN** learned ROI candidate evaluation completes
- **THEN** the run SHALL write `burden_model/learned_roi/candidates/learned_roi_candidate_summary.json`
- **AND** the summary SHALL identify candidate count, provider availability, best image-level candidate, best subject-level candidate, per-image readiness, subject/cohort readiness, blockers, and next action
- **AND** it SHALL explicitly state whether any result is suitable for README/docs sharing

#### Scenario: Readiness gates block weak candidates
- **WHEN** a learned ROI candidate has broad prediction sets, undercoverage, nonfinite outputs, unexplained numerical warnings, subject/cohort instability, or cohort-specific failure
- **THEN** the summary SHALL mark the relevant track exploratory or blocked
- **AND** the report SHALL NOT promote that candidate solely because it has the lowest MAE
