## Context

The repo already has Python logging utilities in `src/eq/utils/logger.py`, many high-level functions call `get_logger(...)`, and `eq --log-file` can attach a file handler for top-level `eq` commands. The durable workflow log that operators actually rely on is concentrated in `src/eq/training/run_glomeruli_candidate_comparison_workflow.py`, where `_emit(..., log_handle)` manually tees subprocess output to `$EQ_RUNTIME_ROOT/logs/run_config/<run_id>/<timestamp>.log`.

That creates two operational classes of execution:

- `eq run-config` candidate-comparison runs produce durable, useful run logs.
- Direct module runs, direct workflow functions, and imported high-level functions may emit logger events but do not consistently create durable runtime logs or capture subprocess stdout/stderr.

This change should make logging a repo-wide execution contract without turning every helper function into a hidden filesystem writer.

## Goals / Non-Goals

**Goals:**

- Make function-level logging useful by default for supported high-level work: resolved inputs, output roots, counts, device/backend, seed, major decisions, artifact paths, elapsed time, and failure context.
- Make durable log capture available for every supported top-level execution surface, not only `eq run-config`.
- Preserve the current `eq run-config` log location while moving implementation to a shared helper.
- Capture subprocess stdout/stderr through the same helper so training workers and workflow orchestrators leave a single readable trail.
- Add a test matrix that covers the entire supported execution ecosystem and future-proofs handler cleanup.
- Add OpenSpec governance so future execution-surface changes must state logging behavior and validation.

**Non-Goals:**

- This change does not promote any model, change training math, alter segmentation thresholds, or revise quantification estimands.
- This change does not add a second workflow runner, alternate path system, or compatibility alias.
- This change does not make low-level utility helpers create durable log files as an import side effect.
- This change does not write logs into the Git checkout.

## Decisions

### Decision 1: Separate event emission from durable capture

Supported high-level functions SHALL emit logger events. Entrypoints SHALL configure durable capture.

Rationale: imported functions need useful logs when a caller configures logging, but they must remain composable in tests, notebooks, and other functions. If every function calls `setup_logging(...)` or creates files, handler duplication and hidden runtime writes become likely.

Alternative considered: make each function create its own log file. Rejected because nested workflows would create fragmented logs, duplicate handlers, and runtime side effects from ordinary imports or unit tests.

### Decision 2: Add one shared execution logging helper

Add `src/eq/utils/execution_logging.py` with these exact responsibilities:

- `ExecutionLogContext`: surface name, run ID, log path, runtime root, dry-run flag, config path, command line, start time, and status.
- `resolve_execution_log_path(...)`: deterministic path construction for `run_config` and direct surfaces.
- `execution_log_context(...)`: context manager that attaches a temporary file handler to the `eq` logger family and removes it on exit.
- `log_execution_start(...)`, `log_execution_success(...)`, and `log_execution_failure(...)`: common operational records.
- `run_logged_subprocess(...)`: stdout/stderr teeing for subprocess workers with return-code failure context.

Rationale: this keeps the behavior centralized without replacing Python logging itself.

Alternative considered: extend only `src/eq/utils/logger.py`. Rejected because the new behavior is run-context and runtime-path oriented, while `logger.py` should remain the low-level logger setup surface.

### Decision 3: Preserve run-config path semantics and add direct-run semantics

`eq run-config` SHALL continue writing to:

```text
$EQ_RUNTIME_ROOT/logs/run_config/<run_id>/<timestamp>.log
```

Direct supported module execution SHALL write to:

```text
$EQ_RUNTIME_ROOT/logs/direct/<surface>/<run_id>/<timestamp>.log
```

The `<surface>` value SHALL be a stable identifier such as `train_mitochondria`, `train_glomeruli`, `compare_glomeruli_candidates`, `glomeruli_transport_audit`, `highres_glomeruli_concordance`, or `endotheliosis_quantification`.

Rationale: preserving `logs/run_config` avoids breaking existing operator expectations while making direct execution discoverable and non-conflicting.

Alternative considered: put all logs under `logs/run_config`. Rejected because direct module execution is not config orchestration and should be distinguishable.

### Decision 4: Refactor prints and ad hoc emitters into logger events

Supported workflow modules SHALL replace operational `print(...)` and `_emit(..., log_handle)` status output with logger calls or the shared subprocess tee helper. Entry points may still print concise terminal-facing summaries when that is part of CLI UX, but the durable log must receive the same substantive information.

Rationale: print-only status is not capturable through logging handlers, and workflow-specific tee code makes behavior drift.

Alternative considered: leave print output and wrap `sys.stdout` globally. Rejected because global stream replacement is brittle and harder to test than explicit subprocess teeing plus logger events.

### Decision 5: Treat logging as an ecosystem test surface

Add `tests/test_execution_logging_contract.py` and extend focused existing tests so validation covers:

- `eq run-config --config <committed config> --dry-run` writes or predicts the correct `logs/run_config` path.
- Direct workflow module dry-runs write `logs/direct/<surface>/...` logs.
- Direct training module dry-runs or bounded monkeypatched executions write direct logs without launching long training.
- Imported high-level functions emit logger events under `caplog` but do not create log files unless a caller attaches the execution context.
- Subprocess stdout/stderr is captured in the parent durable log.
- Failure logs include surface, command, exception or return code, and log path.
- Temporary handlers are removed after success and failure.

Rationale: the user-facing failure mode is ecosystem drift, not a single missing log statement.

Alternative considered: test only `eq run-config`. Rejected because the whole problem is that non-run-config execution is currently second-class.

### Decision 6: Propagate through OpenSpec governance

Update `openspec-change-governance` so future changes that add or modify supported CLI commands, module entrypoints, workflow runners, high-level pipeline functions, or subprocess worker orchestration must document logging participation and validation.

Rationale: this is how the decision persists beyond the immediate refactor.

Alternative considered: document the convention only in README. Rejected because README guidance does not force future implementation plans to acknowledge the logging contract.

## Risks / Trade-offs

- [Risk] File handlers may duplicate events or remain attached after failures. -> Mitigation: implement context-manager cleanup and tests that count handlers before and after success and failure.
- [Risk] Direct execution tests could launch expensive training or require private runtime assets. -> Mitigation: use dry-runs, monkeypatch subprocess/training calls, and synthetic temporary runtime roots for contract tests.
- [Risk] Logs could accidentally be written under the repository checkout. -> Mitigation: resolve through `$EQ_RUNTIME_ROOT` and add tests that fail if repo-root `logs/` is created.
- [Risk] Too much INFO logging can obscure useful signals. -> Mitigation: define required INFO milestones and keep detailed object dumps at DEBUG.
- [Risk] Subprocess workers can still configure their own log files. -> Mitigation: parent workflows capture stdout/stderr, while worker entrypoints use the shared direct logging surface when invoked independently.
- [Risk] Existing docs may continue to imply that only `eq run-config` logs are useful. -> Mitigation: update docs to describe `logs/run_config` and `logs/direct` as current behavior.

## Migration Plan

1. Add `src/eq/utils/execution_logging.py` and focused unit tests for path resolution, context-manager handler cleanup, start/success/failure records, and subprocess tee capture.
2. Wire `src/eq/__main__.py` and `src/eq/run_config.py` to use the shared context without changing public command names.
3. Refactor `src/eq/training/run_glomeruli_candidate_comparison_workflow.py` to remove `_emit(..., log_handle)` and `_run_config_log_path(...)` in favor of shared helpers.
4. Wire direct module entrypoints for `train_mitochondria`, `train_glomeruli`, `compare_glomeruli_candidates`, transport audit, high-resolution concordance, and quantification workflow.
5. Add or extend tests across CLI, workflow dry-runs, direct module dry-runs, imported-function `caplog`, subprocess failure, and handler cleanup.
6. Update `README.md`, `docs/OUTPUT_STRUCTURE.md`, `docs/ONBOARDING_GUIDE.md`, and `docs/TECHNICAL_LAB_NOTEBOOK.md`.
7. Update OpenSpec governance and run strict validation.

Rollback is straightforward because the change should be behaviorally additive at the interface level: remove shared helper wiring and restore prior entrypoint handler setup if the shared helper causes handler duplication. The rollback must preserve any tests that caught true missing function-level events.

## Explicit Decisions

- Helper module: `src/eq/utils/execution_logging.py`.
- Primary contract tests: `tests/test_execution_logging_contract.py`.
- Run-config log root: `$EQ_RUNTIME_ROOT/logs/run_config/<run_id>/<timestamp>.log`.
- Direct module log root: `$EQ_RUNTIME_ROOT/logs/direct/<surface>/<run_id>/<timestamp>.log`.
- Surfaces covered in the first implementation: `eq run-config`, `src/eq/run_config.py`, `src/eq/__main__.py`, `src/eq/training/run_glomeruli_candidate_comparison_workflow.py`, `src/eq/evaluation/run_glomeruli_transport_audit_workflow.py`, `src/eq/evaluation/run_highres_glomeruli_concordance_workflow.py`, `src/eq/quantification/run_endotheliosis_quantification_workflow.py`, `src/eq/training/train_mitochondria.py`, `src/eq/training/train_glomeruli.py`, `src/eq/training/compare_glomeruli_candidates.py`, and high-level quantification functions in `src/eq/quantification/pipeline.py`.
- Future propagation surface: `openspec/specs/openspec-change-governance/spec.md`.
