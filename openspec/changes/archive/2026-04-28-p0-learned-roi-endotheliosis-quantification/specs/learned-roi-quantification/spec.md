## ADDED Requirements

### Requirement: Learned ROI provider availability is audited
The quantification workflow SHALL audit learned ROI feature providers before fitting learned ROI candidates.

#### Scenario: Provider audit is written
- **WHEN** `eq run-config --config configs/endotheliosis_quantification.yaml` reaches learned ROI feature extraction
- **THEN** the run SHALL write `burden_model/learned_roi/diagnostics/provider_audit.json`
- **AND** the artifact SHALL include provider IDs, availability status, package/module provenance, model/weight provenance where available, failure messages where unavailable, and whether network access or downloads were required
- **AND** provider statuses SHALL be one of `available_fit_allowed`, `available_audit_only`, `unavailable`, or `failed`
- **AND** candidates SHALL use only phase-1 providers marked `available_fit_allowed`

#### Scenario: Required baseline providers are evaluated
- **WHEN** provider audit runs
- **THEN** it SHALL evaluate `current_glomeruli_encoder` and `simple_roi_qc`
- **AND** it SHALL evaluate optional local providers `torchvision_resnet18_imagenet` and `timm_dino_or_vit` when their packages and weights are usable without network downloads
- **AND** `current_glomeruli_encoder` and `simple_roi_qc` SHALL be the only providers marked `available_fit_allowed`
- **AND** optional backbone or foundation providers SHALL be recorded as `available_audit_only`, `unavailable`, or `failed`, not silently substituted and not fitted in phase 1

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

#### Scenario: Phase-1 candidate set is capped
- **WHEN** learned ROI candidates are fit
- **THEN** the only image-level candidate IDs SHALL be `image_current_glomeruli_encoder`, `image_simple_roi_qc`, and `image_current_glomeruli_encoder_plus_simple_roi_qc`
- **AND** the only subject-level candidate IDs SHALL be `subject_current_glomeruli_encoder`, `subject_simple_roi_qc`, and `subject_current_glomeruli_encoder_plus_simple_roi_qc`
- **AND** optional `torchvision`, `timm`, DINO/ViT, UNI, CONCH, or other foundation/backbone providers SHALL NOT produce fitted candidate rows in `learned_roi_candidate_metrics.csv`

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

### Requirement: Learned ROI cohort confounding is diagnosed
Learned ROI candidates SHALL include diagnostics that can block readiness when learned features mainly separate cohort, stain, acquisition, or mask-source differences.

#### Scenario: Cohort confounding diagnostics are written
- **WHEN** learned ROI candidate evaluation completes
- **THEN** the run SHALL write `burden_model/learned_roi/diagnostics/cohort_confounding_diagnostics.json`
- **AND** the artifact SHALL include row and subject counts by cohort, score distribution by cohort, candidate residual summaries by cohort, image-level prediction-set coverage by cohort, average prediction-set size by cohort, and a cohort-predictability screen from the selected feature set
- **AND** it SHALL include a leave-one-cohort-out or train-one-cohort/test-other diagnostic when cohort score support is sufficient to estimate it

#### Scenario: Cohort confounding blocks readiness
- **WHEN** cohort-specific prediction-set coverage is below `0.80` for a cohort with at least 30 rows, one cohort's grade-scale MAE exceeds another cohort's grade-scale MAE by at least `0.35`, selected features predict cohort with cross-validated balanced accuracy at least `0.80` while score prediction remains weak or unstable, or leave-one-cohort-out diagnostics show finite-prediction failure or qualitative collapse
- **THEN** the learned ROI candidate summary SHALL mark the relevant track blocked by cohort diagnostics
- **AND** README/docs-ready status SHALL be false for that track

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
- **AND** it SHALL report stage-index metrics and ordinal/grade-scale metrics side by side so candidates are not selected solely from the 0-100 stage-index recoding

#### Scenario: Readiness gates block weak candidates
- **WHEN** a learned ROI candidate has broad prediction sets, undercoverage, nonfinite outputs, unexplained numerical warnings, subject/cohort instability, or cohort-specific failure
- **THEN** the summary SHALL mark the relevant track exploratory or blocked
- **AND** the report SHALL NOT promote that candidate solely because it has the lowest MAE

#### Scenario: Image-track README readiness uses explicit gates
- **WHEN** image-level learned ROI readiness is evaluated
- **THEN** README/docs-ready status SHALL require empirical prediction-set coverage at least `0.88` overall at nominal `0.90` coverage
- **AND** it SHALL require empirical prediction-set coverage at least `0.80` for each observed-score stratum with at least 30 rows
- **AND** it SHALL require empirical prediction-set coverage at least `0.80` for each cohort with at least 30 rows
- **AND** it SHALL require average prediction-set size at most `4.0`
- **AND** it SHALL require no nonfinite predictions, no nonfinite feature columns used by the selected candidate, no unresolved numerical-instability warnings, and no cohort-confounding blocker

#### Scenario: Subject/cohort-track README readiness uses explicit gates
- **WHEN** subject/cohort learned ROI readiness is evaluated
- **THEN** README/docs-ready status SHALL require one aggregated row per `subject_id`, subject-heldout validation, grouped bootstrap confidence intervals, no nonfinite predictions, no unresolved numerical-instability warnings, and no cohort-confounding blocker
- **AND** the summary SHALL state that subject/cohort readiness does not imply operationally precise per-image predictions

### Requirement: Learned ROI implementation iterates to a bounded verdict
The learned ROI implementation SHALL continue correcting in-scope defects until the workflow either passes readiness gates or produces complete failure evidence.

#### Scenario: Iteration remains within fixed phase-1 scope
- **WHEN** implementation changes are made after an initial learned ROI run fails a gate
- **THEN** iteration SHALL be limited to implementation defects, calibration or reporting corrections, and required artifact completeness within the approved phase-1 candidate set
- **AND** iteration SHALL NOT expand fitted providers, weaken validation gates, add compatibility paths, introduce new biological claims, or tune against held-out results without a new explicit OpenSpec decision

#### Scenario: Failure evidence is complete when gates remain unmet
- **WHEN** no learned ROI track passes readiness gates after in-scope corrections
- **THEN** `learned_roi_candidate_summary.json`, `learned_roi_calibration.json`, `cohort_confounding_diagnostics.json`, review evidence, and `audit-results.md` SHALL identify which gates failed and the concrete next action
- **AND** README/docs-ready status SHALL remain false for failed tracks
