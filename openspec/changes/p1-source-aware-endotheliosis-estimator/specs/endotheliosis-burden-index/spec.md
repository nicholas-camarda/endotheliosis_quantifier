## ADDED Requirements

### Requirement: Combined burden reports include source-aware estimator verdicts
The combined quantification report SHALL include source-aware estimator verdict information when source-aware estimator artifacts are generated.

#### Scenario: Source-aware verdict appears in combined review
- **WHEN** `burden_model/source_aware_estimator/summary/estimator_verdict.json` exists
- **THEN** `quantification_review/quantification_review.html` SHALL include a source-aware estimator section
- **AND** it SHALL link to `burden_model/source_aware_estimator/INDEX.md`
- **AND** it SHALL show selected image candidate, selected subject candidate, upstream ROI adequacy status, hard blockers, scope limiters, reportable scopes, non-reportable scopes, and claim boundary
- **AND** it SHALL link to `burden_model/source_aware_estimator/summary/metrics_by_split.csv` when present
- **AND** it SHALL link to or embed the capped source-aware summary figures when present
- **AND** it SHALL prioritize the estimator verdict over internal candidate metrics

#### Scenario: Results summaries include verdict-level rows
- **WHEN** source-aware estimator artifacts exist
- **THEN** `quantification_review/results_summary.csv` SHALL include verdict-level source-aware rows for hard-blocker status, scope-limiter status, image-reportable status, subject-reportable status, and README-snippet eligibility
- **AND** `quantification_review/results_summary.csv` SHALL include an upstream ROI adequacy row when `diagnostics/upstream_roi_adequacy.json` exists
- **AND** `quantification_review/results_summary.csv` SHALL include source-aware training/apparent, validation, and testing-availability summary rows when `summary/metrics_by_split.csv` exists
- **AND** `quantification_review/results_summary.md` SHALL summarize source-aware behavior in human-readable prose
- **AND** neither summary SHALL require the reader to inspect `internal/candidate_metrics.csv` to know whether the estimator is usable for the current claim

### Requirement: README snippets only include source-aware results when explicitly reportable
Source-aware estimator outputs SHALL enter README snippets only when the estimator verdict marks the relevant scope reportable.

#### Scenario: README snippet excludes non-reportable estimator scopes
- **WHEN** `estimator_verdict.json` marks `readme_snippet_eligible` as false
- **THEN** `quantification_review/readme_results_snippet.md` SHALL NOT include source-aware estimator results
- **AND** the combined review SHALL still link to the source-aware estimator index for runtime review

#### Scenario: README snippet includes claim boundary when eligible
- **WHEN** `estimator_verdict.json` marks `readme_snippet_eligible` as true
- **THEN** `quantification_review/readme_results_snippet.md` SHALL state whether the reportable result is image-level, subject-level, or aggregate-only
- **AND** it SHALL state that the estimate is a grade-equivalent burden estimate calibrated to the current scored MR TIFF/ROI data
- **AND** it SHALL NOT describe the result as external validation, causal evidence, closed-capillary percent, or true tissue-area percent

### Requirement: Burden artifact ergonomics are enforced for experimental estimators
Experimental burden estimators SHALL be organized so the first artifact a reader opens explains the output tree and trust status.

#### Scenario: Experimental estimator index is required
- **WHEN** any experimental burden estimator writes more than one artifact subtree under `burden_model/`
- **THEN** it SHALL include an `INDEX.md` at that estimator subtree root
- **AND** the index SHALL identify human-facing, diagnostic, prediction, evidence, and internal artifacts
- **AND** the index SHALL state what can be trusted, what is limited, and what should not be reported

#### Scenario: Experimental internals are separated from summaries
- **WHEN** experimental estimator candidate metrics, diagnostic feature tables, or exhaustive candidate outputs are written
- **THEN** they SHALL live under an `internal/` or `diagnostics/` role folder
- **AND** human-facing verdict files SHALL live under `summary/`
- **AND** the workflow SHALL NOT duplicate the same experimental table into multiple role folders without a named consumer
