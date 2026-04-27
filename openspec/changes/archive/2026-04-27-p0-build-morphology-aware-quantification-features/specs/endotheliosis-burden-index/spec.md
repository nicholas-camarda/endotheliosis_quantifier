## ADDED Requirements

### Requirement: Burden artifacts are organized by role
Future burden-index runs SHALL organize burden artifacts into role-specific subfolders so deployed exploratory models, validation outputs, calibration outputs, summaries, evidence, candidates, diagnostics, and feature sets are unambiguous.

#### Scenario: Grouped burden output folders are written
- **WHEN** `eq run-config --config configs/endotheliosis_quantification.yaml` completes quantification
- **THEN** the run SHALL write burden artifacts under `burden_model/primary_model/`, `burden_model/validation/`, `burden_model/calibration/`, `burden_model/summaries/`, `burden_model/evidence/`, `burden_model/candidates/`, `burden_model/diagnostics/`, and `burden_model/feature_sets/`
- **AND** candidate-screen artifacts SHALL live under `burden_model/candidates/`
- **AND** feature tables SHALL live under `burden_model/feature_sets/`
- **AND** review evidence SHALL live under `burden_model/evidence/`

#### Scenario: Candidate artifacts are not presented as deployed models
- **WHEN** quantification review reports or summaries list model artifacts
- **THEN** `burden_model/candidates/*` artifacts SHALL be labeled as candidate screens or review evidence
- **AND** only serialized selected model artifacts under `burden_model/primary_model/` SHALL be described as model artifacts
- **AND** the report SHALL state that candidate screens do not establish deployment readiness

#### Scenario: Historical flat outputs are not silently shimmed
- **WHEN** the new grouped output layout is active
- **THEN** the workflow SHALL NOT write duplicate compatibility aliases to the old flat `burden_model/*` locations
- **AND** documentation SHALL state that old flat outputs are historical runtime artifacts

### Requirement: Burden report includes morphology-aware evidence
The combined quantification report SHALL include morphology-aware feature evidence when those features are generated.

#### Scenario: Morphology feature summary is included
- **WHEN** morphology features are available
- **THEN** `quantification_review/quantification_review.html` SHALL include a morphology feature summary section
- **AND** it SHALL link to `burden_model/evidence/morphology_feature_review/feature_review.html`
- **AND** it SHALL show whether operator adjudication was provided

#### Scenario: Subject/cohort and image-level tracks are separated
- **WHEN** morphology candidates are reported
- **THEN** the review SHALL separate `subject_burden` readiness from `per_image_burden` readiness
- **AND** subject/cohort readiness SHALL NOT be used as evidence that individual image predictions are operationally precise

#### Scenario: README snippet is gated by selected track
- **WHEN** `quantification_review/readme_results_snippet.md` is generated
- **THEN** it SHALL state whether the selected shareable result is a subject/cohort burden summary or a per-image prediction result
- **AND** it SHALL remain not shareable if neither track passes readiness gates
