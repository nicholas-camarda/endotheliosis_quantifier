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
- **AND** only `audit_backed_fixed_threshold` and `validation_derived_threshold` SHALL be eligible to clear the threshold-policy promotion gate

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

#### Scenario: Explicit threshold is supplied without audit evidence
- **WHEN** candidate comparison is run with `--prediction-threshold` and without an overcoverage audit artifact directory
- **THEN** the report records `threshold_policy_status=fixed_review_threshold`
- **AND** the thresholded masks may be used for review only
- **AND** the threshold-policy promotion gate remains blocked

#### Scenario: Explicit threshold is supplied with audit evidence
- **WHEN** candidate comparison is run with both `--prediction-threshold` and an overcoverage audit artifact directory
- **THEN** the report records `threshold_policy_status=audit_backed_fixed_threshold`
- **AND** it records the attached audit path, selected threshold, threshold grid, background false-positive foreground fraction, and positive/boundary recall

#### Scenario: Threshold is derived from audit evidence
- **WHEN** candidate comparison is run with an overcoverage audit artifact directory and without `--prediction-threshold`
- **THEN** the report records `threshold_policy_status=validation_derived_threshold`
- **AND** it selects one shared threshold from `threshold_sweep.csv` by requiring every candidate family's mean background false-positive foreground fraction to be `<= 0.02`, maximizing mean positive/boundary recall, then breaking ties by lower background foreground fraction and lower threshold
- **AND** it records the threshold-selection rule and selected-threshold evidence in `promotion_report.json`, `candidate_summary.csv`, `promotion_report.md`, and `promotion_report.html`

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

### Requirement: Candidate comparison audits category gates with category-appropriate metrics
Candidate comparison SHALL explain `category_metric_failure` with category-specific pass/fail evidence before it is used to justify retraining.

#### Scenario: Category-gate audit is written
- **WHEN** candidate comparison computes deterministic category metrics
- **THEN** it writes `category_gate_audit.csv`
- **AND** each row records candidate family, category, gate name, metric name, observed value, required comparator, required value, pass/fail status, threshold, threshold-policy status, and rationale
- **AND** `promotion_report.json`, `candidate_summary.csv`, `promotion_report.md`, and `promotion_report.html` summarize the category-gate status

#### Scenario: Background crop has empty truth
- **WHEN** a deterministic crop category is `background`
- **THEN** the category gate SHALL use background false-positive foreground fraction and pixel-level background correctness
- **AND** it SHALL NOT fail solely because empty-mask Dice or Jaccard is `0`
- **AND** the report SHALL still expose background Dice/Jaccard as descriptive metrics, not the primary background gate

#### Scenario: Positive or boundary crop is evaluated
- **WHEN** a deterministic crop category is `positive` or `boundary`
- **THEN** the category gate SHALL evaluate Dice, Jaccard, precision, recall, and prediction-to-truth foreground ratio against explicit thresholds
- **AND** the report SHALL identify whether failure is primarily low overlap, low recall, low precision, or foreground-size mismatch

#### Scenario: Category metric failure disappears under category-appropriate gates
- **WHEN** the legacy category metric failure was driven only by empty-background Dice/Jaccard but background false-positive foreground fraction is within the accepted limit
- **THEN** the report SHALL reclassify the blocker as a gate-semantics correction
- **AND** the implementation SHALL NOT use that finding alone to justify another training run

#### Scenario: Category metric failure persists under category-appropriate gates
- **WHEN** positive or boundary categories fail their explicit gates, or background false-positive foreground fraction exceeds the accepted limit
- **THEN** the report SHALL keep `category_metric_failure`
- **AND** it SHALL identify the next likely lever as threshold, resize, training signal, augmentation, or insufficient evidence

### Requirement: Candidate comparison treats resize benefit as an evidence gate
Candidate comparison SHALL require explicit resize evidence before `resize_benefit_unproven` can be cleared or used to justify production retraining.

#### Scenario: Current resize policy remains downsampled
- **WHEN** candidate provenance records `crop_size` larger than `output_size`
- **THEN** the report records `resize_benefit_unproven` unless a less-downsampled or no-downsample comparator is attached
- **AND** it records the current crop size, output size, resize ratio, interpolation assumptions, and threshold/resize order

#### Scenario: Resize comparator evidence is attached
- **WHEN** a less-downsampled or no-downsample comparator run exists
- **THEN** the report writes or consumes `resize_policy_screening_summary.csv`
- **AND** it compares background false-positive fraction, positive/boundary Dice, positive/boundary recall, precision, prediction-to-truth foreground ratio, runtime, device, batch size, and memory failures against the current policy

#### Scenario: Controlled resize screen is run
- **WHEN** `resize_benefit_unproven` remains after threshold-policy and category-gate correction
- **THEN** the comparison workflow SHALL support a controlled resize screen comparing `p0_resize_screen_current_512to256` against `p0_resize_screen_512to512`
- **AND** it SHALL use `p0_resize_screen_512to384` only after the `512to512` attempt records an MPS memory, unsupported-operation, or runtime failure
- **AND** every comparator SHALL use the same deterministic split, seed, negative-crop manifest, candidate families, threshold-selection rule, loss, sampler policy, `positive_focus_p`, and augmentation variant unless the run is labeled `combined_non_diagnostic`

#### Scenario: Resize-screening summary is written
- **WHEN** a resize screen is attempted
- **THEN** `resize_policy_screening_summary.csv` records run ID, candidate family, crop size, image size, crop-to-output ratio, batch size, device, threshold-policy status, selected threshold, category-gate status, background false-positive foreground fraction, positive/boundary Dice, positive/boundary recall, precision, prediction-to-truth foreground ratio, runtime status, failure reason, log path, and decision interpretation
- **AND** `audit-results.md` records the same run IDs, commands, output paths, and final resize decision

#### Scenario: Resize comparator materially improves boundaries
- **WHEN** a less-downsampled or no-downsample comparator materially improves positive or boundary Dice, recall, or prediction-to-truth foreground ratio without exceeding the accepted background false-positive gate
- **THEN** the report SHALL classify the current resize policy as a likely `resize_policy_artifact`
- **AND** the next production recipe SHALL use the selected less-downsampled policy rather than clearing the current policy by assumption

#### Scenario: Resize comparator does not improve the result
- **WHEN** the less-downsampled or no-downsample comparator is similar or worse and the current `512to256` policy passes category gates
- **THEN** the report MAY clear `resize_benefit_unproven` for the current policy
- **AND** it SHALL record that resize was tested and not selected as the next production lever

#### Scenario: Resize screen cannot run locally
- **WHEN** MPS memory, unsupported operations, or runtime constraints prevent a less-downsampled screen
- **THEN** the failed command and error are recorded in `resize_policy_screening_summary.csv` and `audit-results.md`
- **AND** the implementation SHALL NOT silently clear `resize_benefit_unproven`
