## ADDED Requirements

### Requirement: Future execution-surface changes declare logging participation
OpenSpec changes that add or modify supported execution surfaces SHALL explicitly state how the changed surface participates in the repo-wide execution logging contract.

#### Scenario: Proposal changes a supported execution surface
- **WHEN** an OpenSpec proposal adds or modifies an `eq` CLI command, direct `python -m eq...` module entrypoint, workflow runner, high-level pipeline function, training function, evaluation function, quantification function, or subprocess worker orchestration
- **THEN** the proposal or design states whether the surface emits function-level logger events, attaches durable entrypoint capture, tee-captures subprocess output, or intentionally remains a low-level helper with no durable logging responsibility
- **AND** it names the expected runtime log root when durable capture is part of the surface

#### Scenario: Tasks validate logging behavior
- **WHEN** an OpenSpec change modifies a supported execution surface
- **THEN** its task plan includes focused validation for logging behavior or explicitly states why existing logging-contract tests cover the change
- **AND** the validation includes handler-cleanup or duplicate-log protection when the change attaches logging handlers

#### Scenario: Explicitness checker protects the logging decision
- **WHEN** repository governance checks are run for a change that modifies execution surfaces
- **THEN** the checks can require a logging-contract note in proposal, design, tasks, or specs before the change is treated as apply-ready
