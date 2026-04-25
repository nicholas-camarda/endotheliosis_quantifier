# openspec-change-governance Specification

## Purpose
Define repository rules for OpenSpec proposal explicitness, open-question classification, and apply-time implementation gates.
## Requirements
### Requirement: Proposal and design artifacts use explicit decision structure
Repository OpenSpec changes SHALL express decisions and unresolved questions in a parseable explicitness structure rather than relying on one free-form ambiguity bucket.

#### Scenario: Proposal or design contains no unresolved questions
- **WHEN** a repository OpenSpec proposal or design artifact is complete and no unresolved questions remain
- **THEN** it SHALL include a `## Explicit Decisions` section that records the exact names, paths, workflow IDs, config filenames, or other concrete choices fixed at spec time
- **AND** it MAY omit `## Open Questions`

#### Scenario: Proposal or design contains unresolved questions
- **WHEN** a repository OpenSpec proposal or design artifact still contains unresolved questions
- **THEN** it SHALL include `## Explicit Decisions` and `## Open Questions`
- **AND** every question under `## Open Questions` SHALL begin with exactly one status tag: `[resolve_before_apply]`, `[audit_first_then_decide]`, or `[defer_ok]`

#### Scenario: Audit-first question is recorded
- **WHEN** an open question is marked `[audit_first_then_decide]`
- **THEN** the question text SHALL identify the audit target, evidence source, or inspection step that will determine the answer

### Requirement: Repo-local proposal guidance prefers exact names over placeholders
Repo-local OpenSpec proposal generation SHALL prefer exact workflow IDs, module paths, function names, config filenames, CLI commands, and output roots whenever they can be decided at proposal or design time.

#### Scenario: Exact implementation surfaces can be named
- **WHEN** a proposal or design can reasonably determine the intended workflow ID, module path, function name, config filename, or output root
- **THEN** it SHALL record the exact name rather than using vague placeholders such as `new workflow runner`, `new config`, `or its replacement`, or `where needed`

#### Scenario: Exact implementation surfaces cannot yet be named
- **WHEN** the repository genuinely cannot choose an exact surface before an audit or comparison
- **THEN** the artifact SHALL record that uncertainty as an explicitly tagged open question rather than hiding it inside vague prose

### Requirement: Apply flow blocks unresolved pre-implementation questions
Repository-local OpenSpec apply flow SHALL stop before implementation when unresolved `[resolve_before_apply]` questions remain in the selected change artifacts.

#### Scenario: Blocking question remains unresolved
- **WHEN** the repo-local apply flow is invoked for a change whose proposal or design still contains one or more `[resolve_before_apply]` questions
- **THEN** implementation SHALL pause before task execution
- **AND** the apply flow SHALL summarize the blocking questions and direct the operator to resolve or reclassify them first

#### Scenario: Only audit-first or deferred questions remain
- **WHEN** the repo-local apply flow is invoked for a change whose remaining open questions are limited to `[audit_first_then_decide]` or `[defer_ok]`
- **THEN** implementation MAY proceed
- **AND** the apply flow SHALL still surface those remaining questions to keep the operator aligned on what is intentionally undecided

### Requirement: Explicitness checker is available and validation-friendly
The repository SHALL provide a lightweight explicitness checker for active OpenSpec changes so ambiguous planning artifacts can be caught during local validation.

#### Scenario: Checker reviews a compliant change
- **WHEN** the explicitness checker is run against an active change whose proposal and design follow the repo-local governance rules
- **THEN** it SHALL exit successfully

#### Scenario: Checker finds unresolved blocking questions
- **WHEN** the explicitness checker is run against an active change whose proposal or design still contains `[resolve_before_apply]` questions
- **THEN** it SHALL fail
- **AND** it SHALL identify the blocking lines that prevented apply-readiness

#### Scenario: Checker finds untagged or high-risk vague ambiguity
- **WHEN** the explicitness checker finds an untagged open question or a high-risk ambiguity pattern in an active change artifact
- **THEN** it SHALL fail and report the location
- **AND** the report SHALL distinguish blocking unresolved questions from style-level explicitness findings

### Requirement: Broad audit changes preserve review evidence
Repository-wide audit and streamlining changes SHALL preserve review evidence in the active OpenSpec change directory before implementation decisions are made.

#### Scenario: Repo-wide audit artifacts are required
- **WHEN** an OpenSpec change proposes broad repository review, cleanup, streamlining, reproducibility, integrity, or robustness work across multiple repo surfaces
- **THEN** the change SHALL define a review artifact location under `openspec/changes/<change-name>/review/`
- **AND** implementation SHALL preserve lane reports or equivalent evidence artifacts there before applying accepted cleanup changes

#### Scenario: Audit findings remain distinct from implementation decisions
- **WHEN** a broad audit lane produces findings
- **THEN** the change SHALL preserve the finding evidence separately from the implementation decision
- **AND** it SHALL record whether each finding is accepted, deferred, or rejected for the current change

### Requirement: Audit-first questions name their deciding evidence
Repository OpenSpec changes SHALL make audit-first open questions traceable to the evidence source that will resolve them.

#### Scenario: Audit-first question is actionable
- **WHEN** a proposal, design, or task plan includes an `[audit_first_then_decide]` question
- **THEN** the question SHALL name the audit target, lane report, command output, source file, runtime artifact, or action-register field that will determine the answer

#### Scenario: Audit-first decision is resolved during apply
- **WHEN** implementation reaches the point where an audit-first question has been answered
- **THEN** the answer SHALL be reflected in the review synthesis, action register, or updated OpenSpec artifact before dependent implementation edits proceed

