# repo-wide-execution-logging Specification

## Purpose
TBD - created by archiving change p1-repo-wide-execution-logging-contract. Update Purpose after archive.
## Requirements
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
- **AND** the run ID is derived from `--run-id`, config `run.name`, output/model stem, or generated timestamp in that order

#### Scenario: Handler cleanup after execution
- **WHEN** a supported top-level execution surface completes or fails
- **THEN** any temporary file handlers attached by the execution logging context are removed
- **AND** a later test or command does not receive duplicate log records from the previous run

#### Scenario: Generic CLI subcommand is classified
- **WHEN** implementation inventories live `eq` subcommands in `src/eq/__main__.py`
- **THEN** every subcommand is classified as automatic runtime logging, explicit `--log-file` only, function-event-only, low-level helper, retired, or unsupported
- **AND** the docs distinguish automatic runtime logs from explicit `eq --log-file` capture for subcommands that are not automatic runtime-log surfaces

#### Scenario: Logging setup preserves execution handlers
- **WHEN** base console logging setup and execution durable logging are both active
- **THEN** console setup does not erase an active execution-log handler
- **AND** imported high-level functions do not call `setup_logging(...)`

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

#### Scenario: Candidate availability failure is intentionally nonfatal
- **WHEN** candidate-comparison training intentionally records a failed candidate as unavailable rather than failing the entire comparison workflow
- **THEN** the durable log and comparison artifacts record the failing command or function, return code or exception, unavailable status, and promotion-decision impact
- **AND** this nonfatal path is tested separately from generic subprocess hard-failure behavior

### Requirement: Runtime log placement stays outside the Git checkout
Execution logs SHALL be written under the configured runtime root and SHALL NOT create repo-root runtime directories.

#### Scenario: Temporary runtime root is configured
- **WHEN** tests or users set `EQ_RUNTIME_ROOT` to a temporary runtime root
- **THEN** execution logs are written under that runtime root
- **AND** the repository checkout does not gain root-level `logs/`, `output/`, `models/`, `raw_data/`, `derived_data/`, or generated runtime directories from logging setup

#### Scenario: Log path helper uses canonical path contract
- **WHEN** execution log paths are resolved
- **THEN** the implementation uses the canonical logs root from `src/eq/utils/paths.py`
- **AND** direct surface names reject path separators, parent-directory segments, and unsafe path fragments

### Requirement: Logging contract tests cover the supported ecosystem
The repository SHALL maintain tests that exercise logging behavior across the supported execution ecosystem rather than only testing `eq run-config`.

#### Scenario: Execution surface matrix is tested
- **WHEN** the logging contract test suite runs
- **THEN** it covers all six committed `eq run-config` workflows, direct workflow modules, direct training modules, direct candidate-comparison module execution, direct quantification workflow execution, imported high-level functions, subprocess success, subprocess hard failure, intentional nonfatal candidate unavailability, path-safety rejection, setup-order safety, and handler cleanup
- **AND** it uses dry-runs, monkeypatching, synthetic fixtures, or temporary runtime roots so the contract suite does not require long training runs or private runtime artifacts

#### Scenario: Logging validation is part of change completion
- **WHEN** the logging change is marked complete
- **THEN** validation includes `python -m pytest -q tests/test_execution_logging_contract.py`, relevant existing CLI/workflow tests, `python3 scripts/check_openspec_explicitness.py openspec/changes/p1-repo-wide-execution-logging-contract`, `python -m ruff check .`, and `env OPENSPEC_TELEMETRY=0 openspec validate p1-repo-wide-execution-logging-contract --strict`
- **AND** repo-wide lint failures are fixed as part of completion rather than left as a known blocker to future logging-contract regression checks

### Requirement: Logs contain operationally useful milestones without changing scientific claims
Execution logs SHALL help an operator determine what ran, what inputs and outputs were used, where time was spent, what decisions were made, and why a run failed, without treating execution success as scientific validation.

#### Scenario: Scientific workflow completes
- **WHEN** a segmentation, transport-audit, concordance, or quantification workflow completes
- **THEN** the durable log records artifact paths, data counts, model artifact references, backend/device, run status, and elapsed time
- **AND** the log does not describe model convergence, MPS execution, or output generation as evidence of scientific promotion unless a separate promotion gate records that conclusion

### Requirement: New run-config workflows SHALL emit workflow-level execution milestones
Any workflow added to `SUPPORTED_WORKFLOWS` in `src/eq/run_config.py` SHALL emit workflow-level operational milestones into the active execution log context.

#### Scenario: New workflow starts under run-config
- **WHEN** a workflow runner is dispatched by `eq run-config --config <config>` for a workflow added after this logging contract
- **THEN** the workflow log includes `EXECUTION_LOG=<path>` when an execution log context is active
- **AND** includes phase-level start/completion events for split/input resolution, dependency preflight, major baseline/evaluation blocks, and summary artifact writes
- **AND** avoids creating parallel ad-hoc log files outside the active execution logging context

### Requirement: CLI setup failures are visible
The `eq` CLI SHALL fail visibly when mode setup or runtime directory setup fails for commands that require those resources.

#### Scenario: Runtime directory cannot be created
- **WHEN** an `eq` command requires a runtime directory and directory creation fails
- **THEN** the CLI SHALL exit nonzero
- **AND** the error SHALL identify the failed path

#### Scenario: Mode parsing is invalid
- **WHEN** CLI mode parsing receives an invalid mode
- **THEN** the CLI SHALL exit nonzero
- **AND** it SHALL NOT default to `AUTO`

#### Scenario: Import-time environment setup fails
- **WHEN** optional import-time environment setup cannot complete
- **THEN** the package SHALL NOT silently suppress the failure for commands that require that setup

### Requirement: CLI startup does not globally mutate MPS fallback
The `eq` CLI SHALL NOT set `PYTORCH_ENABLE_MPS_FALLBACK=1` globally for all Darwin commands.

#### Scenario: Non-training command runs on Darwin
- **WHEN** a non-training `eq` command starts on Darwin
- **THEN** CLI startup SHALL NOT set `PYTORCH_ENABLE_MPS_FALLBACK=1`
- **AND** MPS fallback remains an explicit training or validation command concern

