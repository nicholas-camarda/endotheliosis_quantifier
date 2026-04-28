## ADDED Requirements

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

### Requirement: Learned ROI outputs use grouped burden layout
Learned ROI artifacts SHALL live under the grouped burden output contract.

#### Scenario: Learned ROI output folders are written
- **WHEN** learned ROI evaluation runs
- **THEN** artifacts SHALL be written under `burden_model/learned_roi/primary_model/`, `burden_model/learned_roi/validation/`, `burden_model/learned_roi/calibration/`, `burden_model/learned_roi/summaries/`, `burden_model/learned_roi/evidence/`, `burden_model/learned_roi/candidates/`, `burden_model/learned_roi/diagnostics/`, and `burden_model/learned_roi/feature_sets/`
- **AND** the workflow SHALL NOT write duplicate learned ROI compatibility aliases to flat `burden_model/*` locations

### Requirement: Learned ROI burden claims remain predictive
Learned ROI burden outputs SHALL preserve the current claim boundary for endotheliosis burden.

#### Scenario: Learned ROI report avoids mechanistic overclaiming
- **WHEN** learned ROI results are reported
- **THEN** the report SHALL describe predictions as grade-equivalent endotheliosis burden estimates from learned ROI features
- **AND** it SHALL NOT describe them as true tissue-area percent, closed-capillary percent, causal explanation, or validated mechanistic endotheliosis measurement
- **AND** visual evidence SHALL be labeled as predictive support evidence rather than proof of histologic mechanism
- **AND** the report SHALL state row count and independent `subject_id` count separately when reporting learned ROI validation or burden summaries
