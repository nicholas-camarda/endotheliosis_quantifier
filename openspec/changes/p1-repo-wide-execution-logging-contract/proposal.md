## Why

Direct module and function execution currently produces less durable operational evidence than `eq run-config`, even when the same scientific workflow code is doing the work. Logging should be a repo-wide execution contract: functions emit meaningful events, top-level execution surfaces capture those events to durable runtime logs, and future workflow changes must preserve that boundary.

## What Changes

- Add a shared execution logging contract for supported `eq` CLI commands, direct `python -m eq...` module entrypoints, workflow runners, and imported high-level functions.
- Add `src/eq/utils/execution_logging.py` as the single helper surface for run identity, log-path resolution, temporary file-handler attachment, subprocess stdout/stderr teeing, and structured run-start/run-complete/run-failure records.
- Refactor ad hoc logging and `print(...)` status output in supported workflow modules into logger events while preserving useful console output from entrypoints.
- Replace candidate-comparison-specific `_emit(..., log_handle)` and `logs/run_config/...` plumbing with the shared helper while preserving the existing `logs/run_config/<run_id>/<timestamp>.log` contract for `eq run-config`.
- Add durable direct-run logs under `$EQ_RUNTIME_ROOT/logs/direct/<surface>/<run_id>/<timestamp>.log` for supported direct module execution.
- Add tests that prove logging behavior across the full supported ecosystem: `eq run-config`, direct workflow module execution, direct training module execution, quantification workflow execution, imported high-level function calls, subprocess tee capture, failure capture, and handler cleanup.
- Update docs and OpenSpec governance so future function, CLI, and pipeline changes must declare how they participate in the repo-wide execution logging contract.

## Capabilities

### New Capabilities
- `repo-wide-execution-logging`: Defines the execution logging contract across `eq` CLI commands, workflow runners, direct module entrypoints, subprocess workers, imported high-level functions, runtime log placement, and test coverage.

### Modified Capabilities
- `workflow-config-entrypoints`: Clarifies that `eq run-config` is one capture surface for the same function-level logging events, not the only path that produces useful logs.
- `openspec-change-governance`: Requires future OpenSpec changes that add or modify supported execution surfaces to state their logging behavior and validation.

## Impact

- Affected code: `src/eq/utils/logger.py`, `src/eq/utils/execution_logging.py`, `src/eq/__main__.py`, `src/eq/run_config.py`, `src/eq/training/run_glomeruli_candidate_comparison_workflow.py`, `src/eq/evaluation/run_glomeruli_transport_audit_workflow.py`, `src/eq/evaluation/run_highres_glomeruli_concordance_workflow.py`, `src/eq/quantification/run_endotheliosis_quantification_workflow.py`, `src/eq/training/train_mitochondria.py`, `src/eq/training/train_glomeruli.py`, `src/eq/training/compare_glomeruli_candidates.py`, and high-level quantification surfaces in `src/eq/quantification/pipeline.py`.
- Affected tests: add focused logging-contract tests under `tests/` and extend CLI/workflow tests that already cover dry-run and workflow dispatch.
- Affected docs: `README.md`, `docs/OUTPUT_STRUCTURE.md`, `docs/ONBOARDING_GUIDE.md`, and `docs/TECHNICAL_LAB_NOTEBOOK.md` should describe current log locations and the function-versus-entrypoint responsibility split.
- Artifact boundary: runtime logs remain outside Git under `$EQ_RUNTIME_ROOT/logs/`; repo source files must not create repo-root `logs/`, `output/`, or generated runtime directories.

## Explicit Decisions

- Change name: `p1-repo-wide-execution-logging-contract`.
- Shared helper module: `src/eq/utils/execution_logging.py`.
- Existing logger module remains: `src/eq/utils/logger.py`.
- `eq run-config` log root remains `$EQ_RUNTIME_ROOT/logs/run_config/<run_id>/<timestamp>.log`.
- Direct module log root is `$EQ_RUNTIME_ROOT/logs/direct/<surface>/<run_id>/<timestamp>.log`.
- Function responsibility: supported high-level functions emit logger events and do not configure global handlers or independently create durable log files.
- Entrypoint responsibility: `eq` CLI, `eq run-config`, and direct module `main()` functions attach durable handlers, tee subprocess output when launching workers, and remove temporary handlers on completion or failure.
- Governance propagation: `openspec-change-governance` will require future execution-surface changes to include a logging-contract note and validation command.
