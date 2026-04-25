# glomeruli-candidate-comparison Delta Specification

## MODIFIED Requirements

### Requirement: Candidate comparison produces a deterministic promotion report
The glomeruli candidate-comparison workflow SHALL write a promotion report artifact that makes the comparison and decision auditable.

#### Scenario: Promotion report is generated
- **WHEN** candidate comparison completes
- **THEN** the report records each candidate's provenance, deterministic-manifest metrics, trivial-baseline comparisons, and prediction-review results
- **AND** the report records the seed used for each candidate
- **AND** the report states the final decision outcome as `promoted`, `blocked`, or `insufficient_evidence`
- **AND** the report includes a manifest-coverage summary that records total crops plus unique-image and unique-subject coverage
- **AND** the HTML review surface labels each panel with category, crop provenance, per-panel metrics, threshold, and explicit panel-order semantics
- **AND** the report structure preserves a clean path for future repeated-seed candidate rows without breaking the initial artifact contract

#### Scenario: Shared-manifest prediction semantics are applied
- **WHEN** candidate probabilities are converted into binary segmentation masks for promotion comparison
- **THEN** the comparison workflow SHALL use learner-consistent preprocessing or a deterministic preprocessing path whose equivalence is recorded
- **AND** it SHALL record the threshold policy used to convert probabilities into masks
- **AND** it SHALL NOT treat the hardcoded `0.01` threshold as promotion-ready unless the overcoverage audit provides threshold-sweep and background false-positive evidence supporting that policy
- **AND** it SHALL report whether the threshold was fixed a priori, selected from validation evidence, or exploratory

#### Scenario: Compatibility artifact is available
- **WHEN** a current compatibility-era glomeruli artifact is available for comparison
- **THEN** the promotion report includes it as a non-promoted comparison artifact alongside the transfer and scratch candidates

#### Scenario: Candidate family failures are reported
- **WHEN** a candidate family is unavailable and the workflow still emits a structured promotion report
- **THEN** that report SHALL record the family failure explicitly
- **AND** report generation alone SHALL NOT be treated as evidence that the supported transfer-versus-scratch comparison executed successfully end to end

## ADDED Requirements

### Requirement: Candidate comparison records threshold calibration evidence
Candidate comparison SHALL include threshold calibration provenance whenever it presents binary mask metrics for glomeruli promotion or README-facing performance claims.

#### Scenario: Candidate comparison uses overcoverage audit output
- **WHEN** candidate comparison is run with an overcoverage audit artifact directory
- **THEN** the promotion report records the audit directory, audit summary path, selected threshold policy, threshold grid, and background false-positive foreground fraction at the selected threshold
- **AND** it records positive and boundary recall at the selected threshold

#### Scenario: Candidate comparison lacks threshold sweep evidence
- **WHEN** candidate comparison has no threshold sweep evidence for the candidate artifacts
- **THEN** the report may mark artifacts as runtime-compatible research candidates
- **AND** it SHALL NOT mark a candidate as scientifically promoted
- **AND** it SHALL include root cause `threshold_policy_unverified`

### Requirement: Candidate comparison distinguishes overcoverage from missing negatives
Candidate comparison SHALL distinguish present-but-insufficient negative supervision from missing negative supervision.

#### Scenario: Negative supervision is present but background overcoverage remains
- **WHEN** `negative_crop_supervision_status=present` and background crops still exceed the configured background false-positive foreground limit
- **THEN** the report records that negative supervision is present
- **AND** it classifies the remaining failure as `threshold_policy_artifact`, `training_signal_insufficient`, `resize_policy_artifact`, `augmentation_policy_artifact`, or `overcoverage_unclassified`
- **AND** it does not report `negative_background_supervision_missing` as the primary remaining root cause

#### Scenario: Negative supervision is absent
- **WHEN** candidate provenance records `negative_crop_supervision_status=absent`
- **THEN** the report may record `negative_background_supervision_missing`
- **AND** it SHALL NOT infer negative supervision from unreviewed or untracked background-looking crops
