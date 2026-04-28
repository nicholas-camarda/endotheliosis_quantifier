## ADDED Requirements

### Requirement: High-level functions emit operational logging events
Supported high-level functions SHALL emit useful logger events for the work they perform without independently configuring global logging handlers or creating durable log files.

#### Scenario: Imported high-level function runs under caller-owned logging
- **WHEN** a supported high-level function is called from Python with a caller-configured logger or test `caplog`
- **THEN** it logs the surface name, resolved major inputs, output root or artifact path, seed or deterministic split identity when applicable, backend/device when applicable, row/image/example counts when available, major decision gates, and completion status
- **AND** it does not create `$EQ_RUNTIME_ROOT/logs/` files unless the caller attached the repo execution logging context

#### Scenario: Function fails before completion
- **WHEN** a supported high-level function raises an exception after logging has started
- **THEN** it logs enough failure context to identify the surface, failing step, resolved inputs or command context available at failure time, and exception message
- **AND** it re-raises rather than hiding the failure behind a successful log line

### Requirement: Top-level execution surfaces create durable runtime logs
Supported top-level execution surfaces SHALL attach durable runtime log capture before substantive work begins and SHALL remove temporary handlers after success or failure.

#### Scenario: Run-config execution starts
- **WHEN** `eq run-config --config <config>` starts a supported workflow
- **THEN** the run writes a durable log under `$EQ_RUNTIME_ROOT/logs/run_config/<run_id>/<timestamp>.log`
- **AND** the log records `SURFACE=run_config`, config path, workflow ID, run ID, runtime root, dry-run status, command line, Python interpreter, and log path before workflow work begins

#### Scenario: Direct module execution starts
- **WHEN** a supported direct module entrypoint such as `python -m eq.training.train_mitochondria` or `python -m eq.training.compare_glomeruli_candidates` starts
- **THEN** the run writes a durable log under `$EQ_RUNTIME_ROOT/logs/direct/<surface>/<run_id>/<timestamp>.log`
- **AND** the log records the stable direct surface identifier, command line, runtime root, dry-run status when applicable, and log path before substantive work begins

#### Scenario: Handler cleanup after execution
- **WHEN** a supported top-level execution surface completes or fails
- **THEN** any temporary file handlers attached by the execution logging context are removed
- **AND** a later test or command does not receive duplicate log records from the previous run

### Requirement: Subprocess workers are captured by the parent execution log
Workflow runners that launch subprocess workers SHALL tee worker stdout and stderr into the parent durable execution log while preserving console visibility.

#### Scenario: Worker command succeeds
- **WHEN** a workflow runner launches a subprocess worker such as `eq.training.train_mitochondria`, `eq.training.train_glomeruli`, or `eq.training.compare_glomeruli_candidates`
- **THEN** the parent durable log records the exact command, worker stdout, worker stderr, return code, and elapsed time
- **AND** the parent workflow log remains the readable single trail for that orchestration run

#### Scenario: Worker command fails
- **WHEN** a subprocess worker exits with a nonzero return code
- **THEN** the parent durable log records the command, return code, captured output, failure status, and log path
- **AND** the workflow raises the subprocess failure rather than silently continuing

### Requirement: Runtime log placement stays outside the Git checkout
Execution logs SHALL be written under the configured runtime root and SHALL NOT create repo-root runtime directories.

#### Scenario: Temporary runtime root is configured
- **WHEN** tests or users set `EQ_RUNTIME_ROOT` to a temporary runtime root
- **THEN** execution logs are written under that runtime root
- **AND** the repository checkout does not gain root-level `logs/`, `output/`, `models/`, `raw_data/`, `derived_data/`, or generated runtime directories from logging setup

### Requirement: Logging contract tests cover the supported ecosystem
The repository SHALL maintain tests that exercise logging behavior across the supported execution ecosystem rather than only testing `eq run-config`.

#### Scenario: Execution surface matrix is tested
- **WHEN** the logging contract test suite runs
- **THEN** it covers `eq run-config`, direct workflow modules, direct training modules, direct candidate-comparison module execution, direct quantification workflow execution, imported high-level functions, subprocess success, subprocess failure, and handler cleanup
- **AND** it uses dry-runs, monkeypatching, synthetic fixtures, or temporary runtime roots so the contract suite does not require long training runs or private runtime artifacts

#### Scenario: Logging validation is part of change completion
- **WHEN** the logging change is marked complete
- **THEN** validation includes `python -m pytest -q tests/test_execution_logging_contract.py`, relevant existing CLI/workflow tests, `python3 scripts/check_openspec_explicitness.py openspec/changes/p1-repo-wide-execution-logging-contract`, and `env OPENSPEC_TELEMETRY=0 openspec validate p1-repo-wide-execution-logging-contract --strict`

### Requirement: Logs contain operationally useful milestones without changing scientific claims
Execution logs SHALL help an operator determine what ran, what inputs and outputs were used, where time was spent, what decisions were made, and why a run failed, without treating execution success as scientific validation.

#### Scenario: Scientific workflow completes
- **WHEN** a segmentation, transport-audit, concordance, or quantification workflow completes
- **THEN** the durable log records artifact paths, data counts, model artifact references, backend/device, run status, and elapsed time
- **AND** the log does not describe model convergence, MPS execution, or output generation as evidence of scientific promotion unless a separate promotion gate records that conclusion
