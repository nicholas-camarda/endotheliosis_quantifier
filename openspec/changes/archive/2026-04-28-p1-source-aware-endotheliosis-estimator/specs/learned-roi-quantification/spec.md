## ADDED Requirements

### Requirement: Learned ROI outputs feed source-aware estimation without expanding providers
Learned ROI outputs SHALL be reusable inputs for the source-aware estimator, but P1 SHALL NOT expand the learned ROI provider set.

#### Scenario: Source-aware estimator reads learned ROI artifacts
- **WHEN** `burden_model/learned_roi/feature_sets/learned_roi_features.csv` and `burden_model/learned_roi/candidates/learned_roi_candidate_summary.json` exist
- **THEN** the source-aware estimator SHALL be allowed to use learned ROI feature columns and selected learned ROI diagnostics as candidate inputs
- **AND** the source-aware estimator SHALL preserve learned ROI provider provenance in its internal candidate metadata
- **AND** missing learned ROI artifacts SHALL be reported as an input availability condition rather than silently replaced by new provider extraction

#### Scenario: Provider expansion is prohibited in P1
- **WHEN** source-aware estimator candidates are fit
- **THEN** the workflow SHALL NOT fit new `torchvision`, `timm`, DINO/ViT, UNI, CONCH, or other foundation/backbone learned ROI providers
- **AND** any optional provider status from P0 SHALL remain audit evidence only unless a later OpenSpec change explicitly changes provider scope

### Requirement: Learned ROI readiness failures are reclassified for practical estimation
P1 SHALL distinguish learned ROI failures that invalidate estimation from learned ROI limitations that constrain reliability.

#### Scenario: P0 hard-readiness blockers become source-aware inputs
- **WHEN** P0 learned ROI artifacts report score-specific undercoverage, broad prediction sets, cohort/source sensitivity, or nonfatal numerical warnings
- **THEN** the source-aware estimator SHALL treat those conditions as candidate input diagnostics and possible scope limiters
- **AND** those conditions SHALL NOT automatically prevent source-aware estimator prediction artifacts from being written

#### Scenario: Learned ROI invalidity remains a hard blocker
- **WHEN** learned ROI feature inputs contain nonfinite selected feature columns, unsupported labels, unverifiable row provenance, subject leakage, or missing required identity columns
- **THEN** the source-aware estimator SHALL exclude the affected feature family or block the affected estimator claim with an explicit reason
- **AND** it SHALL NOT silently substitute another feature family under the same candidate ID

### Requirement: Score-2 ambiguity is reported rather than hidden
Score-2-like transitional behavior SHALL be represented as uncertainty and reliability metadata when it is not caused by broken data or leakage.

#### Scenario: Score-2 undercoverage becomes a reliability label
- **WHEN** observed score-2 coverage is below a promotion threshold but predictions remain finite and validation grouping is valid
- **THEN** source-aware image predictions for ambiguous or score-2-like cases SHALL include a reliability label such as `transitional_score_region` or `wide_uncertainty`
- **AND** the estimator verdict SHALL list score-2 behavior as a scope limiter rather than a global hard blocker

#### Scenario: Score-2 behavior remains auditable
- **WHEN** calibration diagnostics are written
- **THEN** score-2 coverage, interval width, row count, subject count, and source distribution SHALL remain visible in source-aware diagnostics
- **AND** reports SHALL not imply that score-2-like cases have the same single-image reliability as better-calibrated strata
