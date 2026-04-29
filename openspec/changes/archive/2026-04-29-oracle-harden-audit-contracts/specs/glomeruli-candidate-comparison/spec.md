## ADDED Requirements

### Requirement: Requested candidate training failures stop supported comparison runs
Glomeruli candidate comparison SHALL fail closed when requested transfer or no-base candidate training fails during a supported comparison run.

#### Scenario: Transfer training subprocess fails
- **WHEN** the candidate-comparison workflow requests transfer candidate training and the training subprocess returns nonzero
- **THEN** the workflow SHALL fail
- **AND** it SHALL NOT emit a supported comparison report that treats the transfer candidate as merely unavailable

#### Scenario: No-base training subprocess fails
- **WHEN** the candidate-comparison workflow requests no-base candidate training and the training subprocess returns nonzero
- **THEN** the workflow SHALL fail
- **AND** it SHALL NOT silently substitute a different candidate family

### Requirement: Candidate artifacts require current provenance
Supported candidate evaluation SHALL reject transfer and no-base artifacts that lack current-namespace model metadata and supported provenance.

#### Scenario: Candidate metadata is missing
- **WHEN** a transfer or no-base candidate artifact lacks current training metadata
- **THEN** candidate comparison SHALL reject the artifact before evaluation
- **AND** fallback provenance SHALL NOT be fabricated for that candidate

#### Scenario: Compatibility model is requested for audit
- **WHEN** an explicit compatibility-model audit path is requested
- **THEN** the comparison report SHALL label that artifact as compatibility or audit-only
- **AND** it SHALL NOT mark the artifact as supported or promotion eligible

### Requirement: Candidate comparison uses explicit threshold and preprocessing evidence
Candidate comparison SHALL record learner-consistent preprocessing and the threshold policy used for probability-to-mask conversion.

#### Scenario: Candidate predictions are thresholded
- **WHEN** candidate probabilities are converted to masks
- **THEN** the comparison artifact SHALL record the threshold value and threshold policy
- **AND** a hard-coded `0.01` threshold SHALL NOT be eligible for promotion unless validation evidence records it as audit-backed or validation-derived

#### Scenario: Candidate preprocessing is applied
- **WHEN** candidate predictions are generated
- **THEN** the comparison artifact SHALL record the preprocessing contract
- **AND** the preprocessing SHALL be learner-consistent or explicitly proven equivalent

### Requirement: Resize-screening rescue branches are unsupported
Resize-screening workflow configs SHALL NOT trigger alternate runs conditionally after a primary run fails.

#### Scenario: Fallback resize config is present
- **WHEN** a resize-screening config includes `fallback_run_id` or `run_if: primary_failed`
- **THEN** config validation SHALL reject the config
- **AND** alternate resize evaluation SHALL require a separate explicit workflow config
