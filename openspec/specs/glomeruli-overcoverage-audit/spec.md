# glomeruli-overcoverage-audit Specification

## Purpose
TBD - created by archiving change p0-calibrate-glomeruli-overcoverage-controls. Update Purpose after archive.
## Requirements
### Requirement: Overcoverage audit has a CLI-first control surface
The repository SHALL expose a supported CLI command named `eq glomeruli-overcoverage-audit` for deterministic glomeruli overcoverage analysis.

#### Scenario: Audit command help is displayed
- **WHEN** `eq glomeruli-overcoverage-audit --help` is executed
- **THEN** the help lists arguments for `--run-id`, `--transfer-model-path`, `--scratch-model-path`, `--data-dir`, `--output-dir`, `--thresholds`, `--image-size`, `--crop-size`, `--examples-per-category`, `--device`, and `--negative-crop-manifest`
- **AND** the help states that the command audits existing artifacts before production retraining

#### Scenario: Audit output directory is not supplied
- **WHEN** `eq glomeruli-overcoverage-audit` runs without `--output-dir`
- **THEN** it writes results under the active runtime output root's `segmentation_evaluation/glomeruli_overcoverage_audit/<run_id>/` subtree
- **AND** it does not write generated audit outputs into the Git repository tree

#### Scenario: Required candidate path is missing
- **WHEN** a supplied `--transfer-model-path` or `--scratch-model-path` does not exist
- **THEN** the command fails with an error that identifies the missing candidate path
- **AND** it does not silently substitute another candidate artifact

### Requirement: Overcoverage audit records candidate input provenance
The overcoverage audit SHALL record every candidate artifact and evaluation input needed to reproduce the audit.

#### Scenario: Candidate inputs are recorded
- **WHEN** the audit starts
- **THEN** it writes `candidate_inputs.json`
- **AND** the file records `run_id`, candidate family, model path, model file hash when readable, data root, deterministic manifest source, device, image size, crop size, threshold list, negative crop manifest path when supplied, and active package/code provenance when available

#### Scenario: Candidate artifact cannot load
- **WHEN** a supplied candidate artifact fails to load in the current certified environment
- **THEN** `audit_summary.json` records the candidate as unavailable with root cause `insufficient_current_namespace_artifacts`
- **AND** the audit does not add legacy namespace shims or compatibility imports to make the artifact load

### Requirement: Overcoverage audit uses deterministic category-stratified examples
The overcoverage audit SHALL evaluate transfer and scratch candidates on deterministic `background`, `boundary`, and `positive` crop categories.

#### Scenario: Deterministic audit manifest is created
- **WHEN** no explicit deterministic validation manifest is supplied
- **THEN** the audit creates or reuses deterministic examples for `background`, `boundary`, and `positive`
- **AND** each crop is assigned to exactly one category
- **AND** the manifest records source image path, source mask path when available, crop box, category, source image identifier, and crop provenance

#### Scenario: Background examples are absent
- **WHEN** the audit cannot select at least one true background example
- **THEN** it records root cause `training_signal_insufficient` as unassessable for background overcoverage
- **AND** it marks the audit result as incomplete rather than passing threshold policy

### Requirement: Overcoverage audit records probability distributions before thresholded masks
The overcoverage audit SHALL compute foreground-probability summaries before converting predictions to binary masks.

#### Scenario: Probability summaries are written
- **WHEN** candidate predictions are computed for deterministic crops
- **THEN** the audit writes `probability_quantiles.csv`
- **AND** the file includes candidate family, category, crop identifier, source image identifier, foreground probability minimum, p01, p05, p10, p25, p50, p75, p90, p95, p99, maximum, mean, and foreground probability area summaries

#### Scenario: Probability review panels are written
- **WHEN** review panels are generated
- **THEN** the audit writes `review_panels/index.html`
- **AND** each panel shows raw crop, ground truth when available, foreground probability heatmap, binary masks at the required thresholds, and overlay labels for candidate family, category, crop identifier, and threshold

### Requirement: Overcoverage audit evaluates a fixed threshold grid
The overcoverage audit SHALL evaluate binary segmentation metrics across a fixed threshold grid.

#### Scenario: Default threshold grid is used
- **WHEN** no threshold list is supplied
- **THEN** the audit evaluates thresholds `0.01`, `0.05`, `0.10`, `0.25`, and `0.50`
- **AND** `threshold_sweep.csv` records metrics for every available candidate, category, and threshold

#### Scenario: Threshold sweep metrics are reported
- **WHEN** `threshold_sweep.csv` is written
- **THEN** it includes candidate family, category, threshold, crop count, Dice, Jaccard, precision, recall, predicted foreground fraction, truth foreground fraction, false-positive foreground fraction for background crops, and prediction-to-truth foreground ratio for positive and boundary crops

#### Scenario: Background false-positive curve is reported
- **WHEN** background examples are available
- **THEN** the audit writes `background_false_positive_curve.csv`
- **AND** the file records background predicted foreground fraction summaries by candidate and threshold

### Requirement: Overcoverage audit classifies root cause from evidence
The overcoverage audit SHALL classify overcoverage using explicit root-cause labels derived from probability and threshold evidence.

#### Scenario: Low threshold causes broad binary masks
- **WHEN** background foreground-probability quantiles are low or separable from positive/boundary crops but `0.01` produces unacceptable background foreground fraction
- **THEN** `audit_summary.json` records root cause `threshold_policy_artifact`
- **AND** `audit-results.md` states that another full training run is not the next required step until threshold policy is fixed

#### Scenario: Background probabilities are genuinely high
- **WHEN** background foreground-probability distributions substantially overlap with positive or boundary distributions across candidate families
- **THEN** `audit_summary.json` records root cause `training_signal_insufficient`
- **AND** `audit-results.md` identifies training-signal ablations required before full production retraining

#### Scenario: Resize evidence changes the conclusion
- **WHEN** less-downsampled or no-downsample evidence materially reduces background overcoverage while preserving positive and boundary recall
- **THEN** `audit_summary.json` records root cause `resize_policy_artifact`
- **AND** `audit-results.md` identifies the resize policy that should be screened in the next candidate run

#### Scenario: Resize evidence is the remaining gate
- **WHEN** threshold-policy and category-gate evidence pass but `resize_benefit_unproven` remains
- **THEN** `audit-results.md` identifies resize screening as the next P0 evidence target
- **AND** it names `quicktest_resize_screening.yaml`, `p0_resize_screen_current_512to256`, `p0_resize_screen_512to512`, `p0_resize_screen_512to384`, and `resize_policy_screening_summary.csv`
- **AND** it does not mark P0 complete until the resize screen has produced a selected policy, a cleared current-policy decision, or a recorded infeasibility state

#### Scenario: Augmentation evidence changes the conclusion
- **WHEN** a named augmentation variant materially worsens overcoverage compared with a controlled alternative
- **THEN** `audit_summary.json` records root cause `augmentation_policy_artifact`
- **AND** the report names the exact augmentation variants compared

### Requirement: Overcoverage audit records OpenSpec-local conclusions
The implementation SHALL maintain `openspec/changes/p0-calibrate-glomeruli-overcoverage-controls/audit-results.md` as the durable decision record for this change.

#### Scenario: Audit command is executed during implementation
- **WHEN** the audit is run on real or quick candidate artifacts
- **THEN** `audit-results.md` records the exact command, execution date, model paths, runtime output directory, generated artifact paths, threshold findings, probability findings, and current decision
- **AND** it distinguishes no-training evidence, short screening evidence, and full-run evidence

#### Scenario: Audit has not been run
- **WHEN** implementation tasks are reviewed for completion
- **THEN** the change is not marked complete unless `audit-results.md` states that the audit was run or states the exact hard failure that prevented execution

### Requirement: Overcoverage audit gates full retraining decisions
The repository SHALL not require another full glomeruli production training run before the no-training overcoverage audit has been attempted.

#### Scenario: No-training audit identifies threshold policy artifact
- **WHEN** the no-training audit identifies `threshold_policy_artifact`
- **THEN** the next implementation step is threshold/report contract correction
- **AND** full retraining is deferred until the corrected threshold policy is evaluated

#### Scenario: No-training audit identifies training signal failure
- **WHEN** the no-training audit identifies `training_signal_insufficient`
- **THEN** short screening ablations must identify at least one concrete sampler, loss, resize, or augmentation change before full production retraining

#### Scenario: Audit evidence is inconclusive
- **WHEN** the audit only has short-run artifacts or insufficient loadable current-namespace models
- **THEN** `audit_summary.json` records `inconclusive_short_run_only` or `insufficient_current_namespace_artifacts`
- **AND** `audit-results.md` states the minimum next evidence needed

