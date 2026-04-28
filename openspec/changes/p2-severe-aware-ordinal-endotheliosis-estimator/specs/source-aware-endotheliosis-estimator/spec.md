## ADDED Requirements

### Requirement: Source-aware verdicts hand off to severe-aware estimation
The source-aware estimator verdict SHALL be consumable as upstream evidence for severe-aware ordinal estimation without being reinterpreted as external validation.

#### Scenario: Severe-aware estimator reads source-aware verdict
- **WHEN** `burden_model/source_aware_estimator/summary/estimator_verdict.json` exists and severe-aware ordinal estimation runs
- **THEN** the severe-aware estimator SHALL read the source-aware verdict
- **AND** it SHALL preserve selected image candidate, selected subject candidate, hard blockers, scope limiters, reportable scopes, testing status, and claim boundary as upstream context
- **AND** it SHALL record that context in the severe-aware estimator verdict or diagnostics

#### Scenario: Source-aware limitations carry forward
- **WHEN** the source-aware verdict includes scope limiters such as broad uncertainty, source sensitivity, current-data sensitivity-only testing, or numerical-warning scope limits
- **THEN** the severe-aware estimator SHALL carry those limitations forward as context
- **AND** it SHALL not treat source-aware results as promoted truth, external validation, or proof that severe scores are solved

### Requirement: Source-aware severe failure evidence remains inspectable
The severe-aware workflow SHALL preserve the P1 finding that source-aware scalar estimates underpredict severe scores unless P2 evidence demonstrates a scoped improvement.

#### Scenario: P1 severe compression is recorded as baseline context
- **WHEN** severe-aware estimation starts and P1 source-aware prediction artifacts are present
- **THEN** the severe-aware diagnostics SHALL summarize P1 high-score behavior for score `2` and score `3` where estimable
- **AND** it SHALL compare new severe-aware selected candidates against that baseline
- **AND** it SHALL not hide P1 severe underprediction when P2 reports overall improvements

#### Scenario: P2 improvement is scoped
- **WHEN** severe-aware candidates improve severe threshold behavior but leave scalar burden or source sensitivity limited
- **THEN** the severe-aware verdict SHALL state the improved output type and the remaining source-aware or scalar-burden limitations separately
- **AND** the source-aware estimator's original claim boundary SHALL remain valid for P1 outputs
