## ADDED Requirements

### Requirement: Combined burden reports include severe-aware ordinal verdicts
The combined quantification report SHALL include severe-aware ordinal estimator verdict information when severe-aware estimator artifacts are generated.

#### Scenario: Severe-aware verdict appears in combined review
- **WHEN** `burden_model/severe_aware_ordinal_estimator/summary/estimator_verdict.json` exists
- **THEN** `quantification_review/quantification_review.html` SHALL include a severe-aware ordinal estimator section
- **AND** it SHALL link to `burden_model/severe_aware_ordinal_estimator/INDEX.md`
- **AND** it SHALL show selected image candidate, selected subject candidate, selected severe threshold, selected ordinal candidate, hard blockers, scope limiters, reportable scopes, testing status, README eligibility, and claim boundary
- **AND** it SHALL link to `burden_model/severe_aware_ordinal_estimator/summary/metrics_by_split.csv` when present
- **AND** it SHALL link to severe-threshold metrics and severe false-negative review artifacts when present
- **AND** it SHALL prioritize the estimator verdict over internal candidate metrics

#### Scenario: Severe-aware results summaries include verdict-level rows
- **WHEN** severe-aware estimator artifacts exist
- **THEN** `quantification_review/results_summary.csv` SHALL include verdict-level severe-aware rows for hard-blocker status, scope-limiter status, severe-threshold reportability, ordinal-set reportability, scalar-burden reportability, subject-level reportability, aggregate-current-data reportability, testing availability, and README-snippet eligibility
- **AND** `quantification_review/results_summary.csv` SHALL include severe false-negative summary rows when severe-threshold metrics exist
- **AND** `quantification_review/results_summary.md` SHALL summarize severe-aware behavior in human-readable prose
- **AND** neither summary SHALL require the reader to inspect `internal/candidate_metrics.csv` to know whether severe-end prediction is usable for the current claim

### Requirement: Combined reports distinguish scalar burden from severe-risk and ordinal-set outputs
The combined report SHALL avoid implying that severe-risk or ordinal-set reliability is the same as scalar burden reliability.

#### Scenario: Output type is explicit
- **WHEN** the severe-aware verdict identifies a reportable output type
- **THEN** the combined review SHALL state whether the reportable output is scalar burden, severe-risk label, ordinal prediction set, subject-level aggregate, aggregate-current-data summary, or limited/non-reportable evidence
- **AND** it SHALL not present severe-risk detection as equivalent to continuous burden estimation
- **AND** it SHALL not present subject-level aggregate behavior as proof of single-image precision

#### Scenario: Severe-end failure remains visible
- **WHEN** severe-aware artifacts report severe false negatives or score-2/3 underprediction
- **THEN** the combined review SHALL include that limitation in the severe-aware section
- **AND** it SHALL not summarize only overall MAE when severe-specific metrics are available

### Requirement: README snippets only include severe-aware results when explicitly eligible
Severe-aware ordinal estimator outputs SHALL enter README snippets only when the estimator verdict marks the relevant scope eligible.

#### Scenario: README snippet excludes non-eligible severe-aware results
- **WHEN** `burden_model/severe_aware_ordinal_estimator/summary/estimator_verdict.json` marks `readme_snippet_eligible` as false
- **THEN** `quantification_review/readme_results_snippet.md` SHALL NOT include severe-aware estimator results
- **AND** the combined review SHALL still link to the severe-aware estimator index for runtime review

#### Scenario: README snippet preserves claim boundary when eligible
- **WHEN** `readme_snippet_eligible` is true
- **THEN** `quantification_review/readme_results_snippet.md` SHALL state the eligible output type and scope
- **AND** it SHALL state that the result is predictive grade-equivalent, severe-risk, or ordinal-set evidence calibrated to current scored MR TIFF/ROI data
- **AND** it SHALL NOT describe the result as external validation, causal evidence, closed-capillary percent, or true tissue-area percent
