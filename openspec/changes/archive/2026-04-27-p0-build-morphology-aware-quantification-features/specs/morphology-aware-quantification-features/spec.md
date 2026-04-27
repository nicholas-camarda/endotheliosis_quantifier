## ADDED Requirements

### Requirement: Morphology features are extracted from quantification ROIs
The quantification workflow SHALL extract morphology-aware features from the existing union ROI image and mask crops before candidate fitting.

#### Scenario: Morphology feature table is written
- **WHEN** `eq run-config --config configs/endotheliosis_quantification.yaml` reaches quantification feature extraction
- **THEN** the run SHALL write `burden_model/feature_sets/morphology_features.csv`
- **AND** the table SHALL contain one row per valid ROI embedding row
- **AND** the table SHALL preserve `subject_id`, `sample_id`, `image_id`, `subject_image_id`, `score`, `cohort_id`, `lane_assignment`, `roi_image_path`, and `roi_mask_path` when present
- **AND** all numeric feature columns SHALL be finite or explicitly marked non-estimable with a reason column

#### Scenario: Open lumen features are present
- **WHEN** morphology features are written
- **THEN** the table SHALL include open-lumen features for pale lumen area fraction, lumen candidate count, lumen area summary, lumen circularity, lumen eccentricity, and open-space density
- **AND** feature metadata SHALL define the image-processing method used for each feature

#### Scenario: Collapsed slit features are present
- **WHEN** morphology features are written
- **THEN** the table SHALL include collapsed/slit/ridge features for ridge response, line density, skeleton length per mask area, slit-like object count, and ridge-to-lumen ratio
- **AND** these features SHALL be computed within the glomerulus ROI mask, not from unmasked background
- **AND** slit-like candidates touching the inner glomerular boundary band SHALL be excluded from accepted slit-like area/count features
- **AND** boundary-adjacent slit-like candidates SHALL be written as separate border false-slit features

#### Scenario: RBC confounder features are present
- **WHEN** morphology features are written
- **THEN** the table SHALL include erythrocyte-confounder features for RBC-like color burden, RBC-filled round-lumen candidates, RBC-filled lumen area fraction, and dark filled-lumen shape evidence
- **AND** the feature metadata SHALL state that RBC-filled patent lumina are not automatically collapsed lumina

#### Scenario: Nuclear and mesangial confounder features are present
- **WHEN** morphology features are written
- **THEN** the table SHALL include nuclear/mesangial-confounder features for compact dark cellular burden, compact dark object count, and slit-overlap exclusion burden
- **AND** collapsed/slit-like feature computation SHALL exclude nuclear/mesangial-like masks before writing slit area and slit count
- **AND** the feature metadata SHALL state that mesangial cells and compact nuclei are not automatically collapsed lumina

#### Scenario: Border false-slit features are present
- **WHEN** morphology features are written
- **THEN** the table SHALL include border false-slit area fraction, border false-slit object count, and slit boundary-overlap fraction
- **AND** the feature metadata SHALL state that glomerular capsule, outer-boundary, and crop-edge slit-like signals are rejection evidence rather than accepted collapsed-lumen proof

#### Scenario: Quality and orientation features are present
- **WHEN** morphology features are written
- **THEN** the table SHALL include blur/focus, stain/intensity range, orientation ambiguity, and lumen-detectability features
- **AND** rows with low-quality or ambiguous orientation signals SHALL be surfaced in feature QA selection

### Requirement: Morphology feature QA is reviewer-visible
The workflow SHALL generate a visual QA bundle so the operator can inspect whether feature extraction matches the histologic concept.

#### Scenario: Feature review HTML is written
- **WHEN** morphology features are extracted
- **THEN** the run SHALL write `burden_model/evidence/morphology_feature_review/feature_review.html`
- **AND** it SHALL show raw ROI, glomerulus mask, pale/open lumen overlay, RBC-filled lumen candidate overlay, border false-slit overlay, nuclear/mesangial false-slit overlay, accepted collapsed/slit-like overlay, manual score, and key feature values for selected cases

#### Scenario: Review cases cover important failure modes
- **WHEN** feature review cases are selected
- **THEN** the run SHALL write `burden_model/evidence/morphology_feature_review/feature_review_cases.csv`
- **AND** selected cases SHALL include high-score, low-score, high-uncertainty, high-RBC-confounder, high-collapsed-line, high-open-lumen, and poor-quality/orientation examples where available

#### Scenario: Feature review assets are path-stable
- **WHEN** feature review assets are generated
- **THEN** all images used by `feature_review.html` SHALL live under `burden_model/evidence/morphology_feature_review/assets/`
- **AND** the HTML SHALL use relative links within that review bundle

### Requirement: Operator adjudication is plug-and-play
The workflow SHALL create an operator review template that can be filled without editing code.

#### Scenario: Adjudication template is written
- **WHEN** feature review cases are selected
- **THEN** the run SHALL write `burden_model/evidence/morphology_feature_review/operator_adjudication_template.csv`
- **AND** the table SHALL include `case_id`, `subject_id`, `sample_id`, `image_id`, `score`, `open_empty_lumen_present`, `open_rbc_filled_lumen_present`, `collapsed_slit_like_lumen_present`, `mesangial_or_nuclear_false_slit_present`, `border_false_slit_present`, `poor_orientation_or_quality`, `feature_detection_problem`, `preferred_label_if_detection_wrong`, and `notes`

#### Scenario: Completed adjudication can be read on rerun
- **WHEN** a completed operator adjudication CSV is present at the configured template path
- **THEN** the workflow SHALL read it during the same YAML quantification workflow
- **AND** it SHALL write an agreement summary comparing feature flags against operator labels
- **AND** it SHALL not require a new CLI command or manual code edit

#### Scenario: Missing adjudication does not block first run
- **WHEN** no completed adjudication CSV is present
- **THEN** the workflow SHALL still write morphology features, QA panels, and candidate screens
- **AND** it SHALL report adjudication status as `not_provided`

### Requirement: Morphology candidate screens are track-specific
The workflow SHALL evaluate morphology features separately for image-level prediction and subject/cohort burden summaries.

#### Scenario: Image-level morphology candidates are evaluated
- **WHEN** image-level morphology candidates are fit
- **THEN** validation SHALL use subject-heldout folds
- **AND** candidate metrics SHALL include stage-index MAE, grade-scale MAE, prediction-set coverage, average prediction-set size, cohort-stratified metrics, finite-output status, and numerical-warning status

#### Scenario: Subject-level morphology candidates are evaluated
- **WHEN** subject-level morphology candidates are fit
- **THEN** features SHALL be aggregated by `subject_id`
- **AND** validation SHALL hold out subjects
- **AND** outputs SHALL include subject-level predictions, subject-level MAE, cohort summaries, grouped bootstrap confidence intervals, and cohort stability

#### Scenario: Candidate summary is written
- **WHEN** morphology candidate evaluation completes
- **THEN** the run SHALL write `burden_model/candidates/morphology_candidate_summary.json`
- **AND** the summary SHALL identify the best image-level candidate, best subject-level candidate, selected track if any, readiness status, and next action
- **AND** it SHALL explicitly state whether the result is suitable for README/docs sharing
- **AND** it SHALL block morphology-candidate promotion when accepted slit signal is common in score-zero images, accepted slit signal is nearly ubiquitous, slit signal has high boundary overlap, or nuclear/mesangial confounder burden is high

### Requirement: Morphology feature diagnostics are written
The workflow SHALL diagnose feature quality and numerical stability before candidate fitting.

#### Scenario: Feature diagnostics are written
- **WHEN** morphology features are extracted
- **THEN** the run SHALL write `burden_model/diagnostics/morphology_feature_diagnostics.json`
- **AND** it SHALL include row count, subject count, feature count, nonfinite counts, zero-variance counts, near-zero-variance counts, missingness counts, and feature ranges

#### Scenario: Numerical warnings are attributed
- **WHEN** a morphology, embedding, or combined candidate emits matrix-operation warnings
- **THEN** the candidate diagnostics SHALL record whether warnings are associated with morphology features, frozen embeddings, or combined feature matrices
- **AND** a candidate with repeated unexplained numerical warnings SHALL remain exploratory
