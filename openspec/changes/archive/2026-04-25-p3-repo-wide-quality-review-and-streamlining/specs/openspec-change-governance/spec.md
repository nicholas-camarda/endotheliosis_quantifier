## ADDED Requirements

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
