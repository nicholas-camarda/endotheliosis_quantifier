## ADDED Requirements

### Requirement: P3 evaluates binary no-low versus moderate-severe review triage
The P3 grade-model workflow SHALL evaluate a binary review-triage product that groups glomeruli into no/low endotheliosis versus moderate/severe endotheliosis when the current data support that narrower claim better than multi-ordinal grading.

#### Scenario: Primary binary target excludes borderline score one
- **WHEN** P3 builds the primary binary triage target
- **THEN** rows with score `0` or `0.5` SHALL be labeled `no_low`
- **AND** rows with score `1.5`, `2`, or `3` SHALL be labeled `moderate_severe`
- **AND** rows with score `1.0` SHALL be labeled `borderline_review` and excluded from primary binary training and primary grouped-development metric calculation
- **AND** the workflow SHALL report the number of rows and subjects excluded as borderline review

#### Scenario: Inclusive sensitivity target is separate
- **WHEN** P3 evaluates a score-one-inclusive binary sensitivity target
- **THEN** rows with score `0`, `0.5`, or `1.0` MAY be labeled `no_low_inclusive`
- **AND** rows with score `1.5`, `2`, or `3` SHALL be labeled `moderate_severe`
- **AND** sensitivity metrics SHALL be reported separately from primary binary metrics
- **AND** sensitivity results SHALL NOT replace the primary target verdict

#### Scenario: Binary triage is selected before multi-ordinal deployment
- **WHEN** binary triage passes review-prioritization gates but three-band, four-band, or six-bin ordinal candidates do not pass their deployment gates
- **THEN** P3 MAY select binary review triage as the best current-data product
- **AND** the final verdict SHALL state that the product is designed to speed human grading review by prioritizing likely no/low, likely moderate/severe, and uncertain/borderline cases
- **AND** it SHALL NOT describe the product as a full ordinal grader

### Requirement: Binary triage candidates may use atlas PCA GMM and anchor-derived evidence
Binary triage candidate models SHALL evaluate atlas-derived feature families alongside existing ROI/QC, morphology, learned ROI, and embedding candidates.

#### Scenario: Atlas-derived candidate features are generated
- **WHEN** adjudicated atlas evidence and cluster assignments are available
- **THEN** P3 SHALL make candidate features available for selected atlas view cluster ID, Gaussian-mixture posterior or distance fields where available, reduced embedding PCA coordinates, nearest reviewed anchor distance, nearest reviewed anchor class, blocked-cluster indicator, and recovered-anchor proximity where computable
- **AND** these fields SHALL preserve row identity and shall not use reviewer target labels as leakage features

#### Scenario: Pure atlas cluster mapping is a baseline only
- **WHEN** P3 evaluates a direct mapping from reviewed atlas clusters to no/low or moderate/severe groups
- **THEN** it SHALL report that mapping as a simple baseline
- **AND** it SHALL compare the mapping against supervised grouped-development candidates before any product selection
- **AND** blocked clusters SHALL route to uncertain or review-needed status rather than a forced binary class

#### Scenario: Hybrid candidates are evaluated leakage-safely
- **WHEN** P3 evaluates hybrid candidates using ROI/QC, morphology, learned ROI, embedding PCA, GMM, or anchor-distance features
- **THEN** preprocessing, feature selection, dimensionality reduction, calibration, and threshold selection SHALL be learned inside training folds only
- **AND** validation-fold labels SHALL NOT be used to choose principal components, GMM-to-class mapping, anchor thresholds, feature subsets, calibration, or operating thresholds

### Requirement: Binary triage reports uncertainty confidence intervals and operating thresholds
Binary triage outputs SHALL include uncertainty and confidence evidence sufficient for review prioritization and honest current-data interpretation.

#### Scenario: Grouped-development metrics include uncertainty
- **WHEN** binary triage candidates are evaluated
- **THEN** metrics SHALL include recall, precision, specificity, balanced accuracy, AUROC, average precision, false-negative count, false-positive count, threshold, row count, subject count, source support, and finite-output status
- **AND** grouped-resampling or bootstrap confidence intervals SHALL be written for key metrics when support permits
- **AND** non-estimable confidence intervals SHALL be explicitly marked with the reason

#### Scenario: Predictions include confidence and review routing
- **WHEN** binary triage predictions are written
- **THEN** each row SHALL include predicted probability or score, operating-threshold decision, uncertainty or reliability label, near-threshold flag, source/cohort warning flag, nearest-anchor evidence, and final review route
- **AND** possible review routes SHALL include `likely_no_low`, `likely_moderate_severe`, `borderline_review`, and `uncertain_review`

#### Scenario: Thresholds are selected for review workflow needs
- **WHEN** a binary triage candidate requires an operating threshold
- **THEN** threshold selection SHALL happen inside grouped development data
- **AND** the selected threshold SHALL record whether it prioritizes sensitivity for moderate/severe cases, precision for review workload reduction, or a balanced review-triage operating point
- **AND** the verdict SHALL state the selected operating objective

### Requirement: Binary triage explanations are reviewer-facing decision evidence
Binary triage SHALL write feature explanation artifacts that help reviewers understand why a case was routed without presenting explanations as causal biology.

#### Scenario: Feature explanations are written for model candidates
- **WHEN** a binary triage candidate produces row-level predictions
- **THEN** the workflow SHALL write row-level explanation fields or artifacts that include feature contribution summaries for interpretable models or permutation-style summaries for less interpretable models
- **AND** explanations SHALL identify the contribution of feature families such as ROI/QC, morphology, learned ROI, embedding PCA, GMM/cluster evidence, and anchor distance where present
- **AND** explanations SHALL be labeled as model-decision evidence, not causal or mechanistic proof

#### Scenario: Reviewer evidence includes visual and anchor context
- **WHEN** P3 writes a binary triage review HTML or queue
- **THEN** each selected review case SHALL show ROI image and ROI mask, original score, adjudication evidence when present, predicted route, confidence label, top explanation fields, nearest reviewed anchor, and cluster/source warnings
- **AND** high-confidence errors, near-threshold cases, source-sensitive cases, and blocked-cluster cases SHALL be eligible for review queues

### Requirement: Binary triage promotion is gated and claim-bounded
Binary triage SHALL only become a README-facing or handoff product when it passes explicit current-data gates and remains claim-bounded as review prioritization.

#### Scenario: Binary triage gate can pass review-prioritization only
- **WHEN** a binary triage candidate has finite outputs, no leakage, acceptable grouped-development moderate/severe recall, acceptable review-workload precision or specificity, bounded source sensitivity, and no confirmed Dox-style non-severe overcall blocker
- **THEN** P3 MAY mark it as `review_triage_current_data_candidate`
- **AND** the final verdict SHALL list the exact metrics and gates that passed

#### Scenario: Binary triage gate fails honestly
- **WHEN** binary triage fails due to severe/moderate false negatives, unusable precision, source sensitivity, nonfinite output, adjudication conflict, or reviewed non-severe overcalls
- **THEN** P3 SHALL mark it as diagnostic only or blocked
- **AND** the final verdict SHALL state the concrete blocker and next review or data action

#### Scenario: Binary triage does not imply external validation
- **WHEN** binary triage artifacts are reported
- **THEN** reports SHALL state that grouped-development and source-sensitivity metrics are current-data evidence
- **AND** they SHALL NOT call the result external validation, clinical deployment, automated grading replacement, or calibrated multi-ordinal classification
