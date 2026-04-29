## ADDED Requirements

### Requirement: Quantification model evaluators use shared estimability contracts
Quantification model evaluators SHALL use shared modeling-contract helpers for candidate/fold estimability, hard-blocker recording, insufficient-data verdicts, and supported sklearn model serialization.

#### Scenario: No candidate is estimable
- **WHEN** a quantification model evaluator has no candidate with usable feature columns or no candidate can produce predictions
- **THEN** the evaluator SHALL write the expected summary, diagnostics, and artifact-manifest outputs
- **AND** it SHALL set the verdict to `current_data_insufficient` or a stricter non-deployable diagnostic state
- **AND** it SHALL record a hard blocker such as `no_estimable_candidate_features`
- **AND** it SHALL NOT crash because an empty prediction or metric collection is concatenated

#### Scenario: Grouped fold is not target-estimable
- **WHEN** a severe, ordinal, or grade candidate is evaluated with grouped folds
- **AND** a training fold lacks the minimum target-class support required by the estimator
- **THEN** the evaluator SHALL record the unestimable fold or candidate in diagnostics
- **AND** it SHALL not call sklearn fitting for that unestimable fold
- **AND** it SHALL continue to a bounded diagnostic or insufficient-data verdict instead of raising an uncaught sklearn one-class error

#### Scenario: Hard blockers propagate to product verdicts
- **WHEN** candidate or fold estimability blockers are recorded
- **THEN** `summary/final_product_verdict.json` SHALL include the blocker or a summarized blocker category
- **AND** first-class family diagnostics SHALL include the relevant blocker details
- **AND** README-facing deployment statuses SHALL remain blocked while any required model-family blocker is unresolved

#### Scenario: Supported sklearn artifact serialization matches the filename contract
- **WHEN** a supported sklearn model artifact is written with a `.joblib` filename
- **THEN** the evaluator SHALL write it with `joblib.dump`
- **AND** tests SHALL prove it can be loaded with `joblib.load`
- **AND** any pickle-based supported artifact SHALL use a `.pkl` filename and explicit pickle metadata instead of a `.joblib` contract

#### Scenario: Shared helpers are reused rather than reimplemented
- **WHEN** P3 or a later quantification evaluator performs candidate/fold estimability checks, insufficient-data verdict construction, hard-blocker JSON payload construction, warning capture, or supported sklearn serialization
- **THEN** it SHALL call helpers in `src/eq/quantification/modeling_contracts.py`
- **AND** it SHALL NOT add a second local implementation of the same gate in the evaluator module
