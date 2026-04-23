## ADDED Requirements

### Requirement: Local-runtime quantification certification requires a numerically stable ordinal stage
Mac local-development certification SHALL NOT treat the contract-first quantification pipeline as fully healthy while the ordinal modeling step still emits unresolved numerical-instability warnings on the supported local runtime cohort.

#### Scenario: Local runtime quantification regression is executed
- **WHEN** the supported local-runtime quantification regression runs through the ordinal modeling step
- **THEN** it completes without unresolved overflow, divide-by-zero, or invalid-value warnings from the canonical ordinal estimator path

#### Scenario: Pipeline still completes with ordinal instability
- **WHEN** the local-runtime quantification regression writes all expected artifacts but the ordinal stage still emits unresolved numerical-instability warnings
- **THEN** Mac local-development certification remains incomplete for ordinal quantification health
- **AND** the result is treated as a pipeline-execution success with an unresolved quantification-model defect
