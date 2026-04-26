## ADDED Requirements

### Requirement: FastAI and historical-doc cleanup closes repo-wide review findings
Repo-wide quality follow-up SHALL include a concrete cleanup pass for FastAI lint hygiene, fail-closed training behavior, historical documentation accessibility, historical documentation quarantine from current guidance, and reuse-first repo streamlining.

#### Scenario: FastAI hygiene finding is accepted
- **WHEN** implementation begins for `p1-align-fastai-practices-and-archive-history`
- **THEN** the accepted repo-wide quality findings include FastAI wildcard imports, active training fallback behavior, required artifact warning-only failures, and historical docs in current guidance surfaces
- **AND** each accepted finding has a named implementation target and validation command

#### Scenario: Existing tooling is inventoried before adding scripts
- **WHEN** implementation needs a documentation quarantine check, active-code hygiene check, link check, path check, CLI check, or OpenSpec check
- **THEN** the implementation first inspects existing scripts, pytest files, path helpers, CLI dry-runs, and docs index surfaces that can perform the job
- **AND** the implementation reuses or extends an existing surface when it can express the requirement cleanly
- **AND** any new standalone script includes a task-closeout note explaining why no existing surface was appropriate

#### Scenario: Duplicate tooling is removed or avoided
- **WHEN** implementation finds multiple scripts, tests, helpers, or docs pages that perform substantially the same validation or guidance role
- **THEN** the change consolidates to one canonical active surface or records a specific reason both must remain
- **AND** current docs point to the canonical active surface rather than parallel alternatives

#### Scenario: Cleanup validation is recorded
- **WHEN** implementation is complete
- **THEN** the change records the reuse-first inventory outcome, any new-script justification, `ruff check .`, `pytest -q`, `eq --help`, OpenSpec explicitness, and OpenSpec strict validation results in the final implementation notes or task closeout
- **AND** any residual non-green result includes a concrete next action rather than being silently deferred

#### Scenario: Current and historical documentation boundaries are reviewed
- **WHEN** the documentation cleanup is complete
- **THEN** current workflow docs and historical archive docs are separately reviewed
- **AND** the review confirms that historical docs are accessible through `docs/HISTORICAL_NOTES.md` or `docs/README.md`
- **AND** the review confirms that current docs do not present historical FastAI fallback or compatibility material as active workflow guidance
