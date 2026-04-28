## ADDED Requirements

### Requirement: Workflow configs use the repo-wide execution logging contract
Workflow config execution through `eq run-config` SHALL capture the same function-level logging events available to direct execution, while preserving the `logs/run_config` durable log contract.

#### Scenario: Run-config captures workflow events
- **WHEN** `eq run-config` executes any supported workflow config
- **THEN** it writes a durable parent log under `$EQ_RUNTIME_ROOT/logs/run_config/<run_id>/<timestamp>.log`
- **AND** the parent log captures logger events emitted by the workflow runner and supported high-level functions
- **AND** workflow-specific manual log-handle plumbing is not maintained as a second durable logging system

#### Scenario: Run-config dry-run records planned execution
- **WHEN** `eq run-config --config <config> --dry-run` executes a supported workflow config
- **THEN** the log records the workflow ID, run ID, config path, runtime root, dry-run status, planned commands or direct workflow action, and log path
- **AND** it does not require private data, model artifacts, or long-running training to validate the logging contract

#### Scenario: Run-config subprocess output is tee-captured
- **WHEN** a run-config workflow launches subprocess worker commands
- **THEN** worker stdout and stderr are tee-captured into the run-config log by the shared execution logging helper
- **AND** the workflow still fails closed on nonzero worker return codes
