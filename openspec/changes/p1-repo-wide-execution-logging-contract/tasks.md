## 1. Baseline Inventory

- [ ] 1.1 Inventory every supported top-level execution surface that can run substantive work: `eq` subcommands in `src/eq/__main__.py`, `eq run-config`, direct workflow modules, direct training modules, direct candidate-comparison module execution, direct quantification workflow execution, and imported high-level pipeline functions.
- [ ] 1.2 Classify every live `eq` subcommand and every direct module with `if __name__ == "__main__"` as `entrypoint_capture`, `global_log_file_only`, `subprocess_worker`, `high_level_function_events_only`, `low_level_helper_no_durable_logging`, or `retired_or_unsupported`.
- [ ] 1.3 Record the inventory in `openspec/changes/p1-repo-wide-execution-logging-contract/execution-surface-inventory.md` with exact module/function names, current logging behavior, required target behavior, and planned tests.
- [ ] 1.4 Confirm no implementation will create repo-root `logs/`, `output/`, `models/`, `raw_data/`, `derived_data/`, or other runtime artifact directories.
- [ ] 1.5 Resolve the generic `eq --log-file` versus automatic runtime-log boundary for each `eq` subcommand, and record which commands remain explicit-log-file-only instead of automatic `$EQ_RUNTIME_ROOT/logs/...` surfaces.
- [ ] 1.6 Confirm all six supported `eq run-config` workflow IDs are in scope: `segmentation_mitochondria_pretraining`, `segmentation_glomeruli_transfer`, `glomeruli_candidate_comparison`, `glomeruli_transport_audit`, `highres_glomeruli_concordance`, and `endotheliosis_quantification`.

## 2. Shared Execution Logging Helper

- [ ] 2.1 Add `src/eq/utils/execution_logging.py` with `ExecutionLogContext`, `resolve_execution_log_path(...)`, `execution_log_context(...)`, `log_execution_start(...)`, `log_execution_success(...)`, `log_execution_failure(...)`, and `run_logged_subprocess(...)`.
- [ ] 2.2 Preserve `src/eq/utils/logger.py` as the low-level logger setup surface and avoid moving runtime-root policy into it.
- [ ] 2.3 Ensure `resolve_execution_log_path(...)` uses the canonical logs root from `src/eq/utils/paths.py::get_logs_path()` rather than creating a parallel runtime path resolver.
- [ ] 2.4 Ensure run-config log path resolution preserves `$EQ_RUNTIME_ROOT/logs/run_config/<run_id>/<timestamp>.log`.
- [ ] 2.5 Ensure direct log path resolution writes to `$EQ_RUNTIME_ROOT/logs/direct/<surface>/<run_id>/<timestamp>.log`.
- [ ] 2.6 Reject direct surface names with path separators, parent-directory segments, empty names, or unsafe path fragments.
- [ ] 2.7 Implement direct-run ID derivation in this order: explicit `--run-id`, config `run.name`, output/model stem, generated timestamp ID; record the derivation source in the log.
- [ ] 2.8 Add handler cleanup logic that removes temporary file handlers on success and failure without clearing unrelated caller-owned handlers.
- [ ] 2.9 Add setup-order protection so `setup_logging(...)` cannot erase an active execution-log handler, or ensure `setup_logging(...)` is never called while execution-log handlers are active.
- [ ] 2.10 Add helper-level tests for path resolution, surface-name safety, run-ID derivation, start/success/failure records, subprocess stdout/stderr tee capture, nonzero return-code failure capture, `setup_logging(...)` interaction, and handler cleanup.

## 3. Wire Run-Config and Workflow Runners

- [ ] 3.1 Update `src/eq/run_config.py` so `run_config(...)` attaches the shared run-config execution context before dispatching all six supported workflow configs.
- [ ] 3.2 Update `src/eq/run_config.py::_run(...)` so `segmentation_mitochondria_pretraining` and `segmentation_glomeruli_transfer` subprocesses use `run_logged_subprocess(...)` and are tee-captured like other workflows.
- [ ] 3.3 Refactor `src/eq/training/run_glomeruli_candidate_comparison_workflow.py` to remove `_emit(..., log_handle)`, `_run_config_log_path(...)`, and custom subprocess teeing in favor of `src/eq/utils/execution_logging.py`.
- [ ] 3.4 Update `src/eq/evaluation/run_glomeruli_transport_audit_workflow.py` to emit logger events and direct-run durable logs instead of print-only operational status.
- [ ] 3.5 Update `src/eq/evaluation/run_highres_glomeruli_concordance_workflow.py` to emit logger events and direct-run durable logs instead of print-only operational status.
- [ ] 3.6 Update `src/eq/quantification/run_endotheliosis_quantification_workflow.py` to emit logger events and direct-run durable logs instead of print-only operational status.
- [ ] 3.7 Ensure every workflow log records workflow ID, config path, runtime root, run ID or direct surface ID, dry-run status, output roots, artifact references, and completion/failure status.
- [ ] 3.8 Preserve intentional nonfatal candidate unavailability in `src/eq/training/compare_glomeruli_candidates.py` only when durable logs and comparison artifacts record command context, return code or exception, unavailable status, and promotion-decision impact.

## 4. Wire Direct Training, Comparison, and Quantification Surfaces

- [ ] 4.1 Update `src/eq/training/train_mitochondria.py` direct module execution to attach direct-run durable logging while leaving `train_mitochondria_with_datablock(...)` as function-event-only when imported.
- [ ] 4.2 Update `src/eq/training/train_glomeruli.py` direct module execution to attach direct-run durable logging while leaving `train_glomeruli_with_datablock(...)` and `train_glomeruli_with_transfer_learning(...)` as function-event-only when imported.
- [ ] 4.3 Update `src/eq/training/compare_glomeruli_candidates.py` direct module execution to attach direct-run durable logging and record comparison output roots, model roots, candidate families, promotion decision status, and failure context.
- [ ] 4.4 Update high-level quantification functions in `src/eq/quantification/pipeline.py` so imported calls emit start/input/output/count/decision/failure logger events without independently attaching file handlers.
- [ ] 4.5 Replace operational `print(...)` calls in supported execution surfaces with logger events, preserving concise CLI summaries only when they remain useful terminal UX.
- [ ] 4.6 Remove or relocate imported-path calls to `setup_logging(...)`, including helper-level calls such as `find_best_mitochondria_model(...)`, so imported functions cannot erase an active execution context.

## 5. Ecosystem-Wide Tests

- [ ] 5.1 Add `tests/test_execution_logging_contract.py` covering helper path resolution, handler cleanup, durable log creation, subprocess tee success/failure, and no repo-root runtime directory creation.
- [ ] 5.2 Add `test_resolve_execution_log_path_preserves_run_config_and_direct_contracts` proving run-config and direct paths resolve under the canonical runtime log root, direct surface names reject unsafe fragments, and no path is relative to the repo checkout.
- [ ] 5.3 Add `test_execution_log_context_records_start_success_and_removes_only_own_handler` proving start/success metadata are recorded and caller-owned handlers remain attached while temporary execution handlers are removed.
- [ ] 5.4 Add `test_execution_log_context_records_failure_and_reraises_without_handler_leak` proving exceptions are logged and re-raised without duplicate-handler leakage.
- [ ] 5.5 Add `test_setup_logging_does_not_erase_active_execution_log_handler` or equivalent coverage proving handler setup ordering cannot drop durable execution logs.
- [ ] 5.6 Add `test_run_logged_subprocess_tees_stdout_and_stderr_on_success` and `test_run_logged_subprocess_logs_nonzero_return_code_and_raises`.
- [ ] 5.7 Add or extend CLI tests proving `eq run-config --config configs/mito_pretraining_config.yaml --dry-run`, `configs/glomeruli_finetuning_config.yaml --dry-run`, `configs/glomeruli_candidate_comparison.yaml --dry-run`, `configs/glomeruli_transport_audit.yaml --dry-run`, `configs/highres_glomeruli_concordance.yaml --dry-run`, and `configs/endotheliosis_quantification.yaml --dry-run` produce or report the expected `logs/run_config` durable log path without launching long training.
- [ ] 5.8 Add direct workflow module tests for `eq.training.run_glomeruli_candidate_comparison_workflow`, `eq.evaluation.run_glomeruli_transport_audit_workflow`, `eq.evaluation.run_highres_glomeruli_concordance_workflow`, and `eq.quantification.run_endotheliosis_quantification_workflow` using dry-runs, monkeypatching, and temporary runtime roots.
- [ ] 5.9 Add direct training module tests for `eq.training.train_mitochondria`, `eq.training.train_glomeruli`, and `eq.training.compare_glomeruli_candidates` using dry-runs or monkeypatched bounded execution so tests do not require real training artifacts.
- [ ] 5.10 Add imported-function tests using `caplog` to prove high-level training and quantification functions emit useful events but do not create durable log files unless wrapped by the execution logging context.
- [ ] 5.11 Add failure-path tests proving nonzero subprocess return codes and raised exceptions are logged with surface, command or function context, log path, and failure status before being re-raised.
- [ ] 5.12 Add candidate-comparison nonfatal-unavailability tests proving unavailable candidates are recorded in durable logs and comparison artifacts rather than silently swallowed.
- [ ] 5.13 Extend `tests/test_glomeruli_resize_screening_workflow.py` with a resize-screening failure/fallback test proving shared subprocess logging records failed command, return code, fallback command, resize decision, and shared log path.
- [ ] 5.14 Add duplicate-handler regression tests proving repeated success and failure runs do not multiply log lines.
- [ ] 5.15 Add a governance regression test, for example under `tests/unit/test_openspec_explicitness.py`, proving execution-surface changes without logging-contract and docs-impact notes fail the repo-local checker while compliant fixtures pass.

## 6. Documentation and Governance Propagation

- [ ] 6.1 Update `README.md` to state that `eq run-config` remains the primary YAML front door and that direct supported module entrypoints also write durable logs under `$EQ_RUNTIME_ROOT/logs/direct/`.
- [ ] 6.2 Update `docs/OUTPUT_STRUCTURE.md` with current log roots: `logs/run_config/<run_id>/` and `logs/direct/<surface>/<run_id>/`.
- [ ] 6.3 Update `docs/ONBOARDING_GUIDE.md` and `docs/TECHNICAL_LAB_NOTEBOOK.md` so operators know where to find logs for run-config, direct module, and imported-function workflows.
- [ ] 6.4 Update `docs/SEGMENTATION_ENGINEERING_GUIDE.md` checklist so future segmentation/training workflow changes include execution-surface logging classification and validation.
- [ ] 6.5 Update docs to distinguish automatic runtime logs from explicit `eq --log-file` capture for generic CLI subcommands classified as `global_log_file_only`.
- [ ] 6.6 Update `scripts/check_openspec_explicitness.py` or add an adjacent repo-local check so future OpenSpec changes that touch execution surfaces must include a logging-contract note, validation rationale, and docs-impact decision.
- [ ] 6.7 Update `openspec/changes/p1-repo-wide-execution-logging-contract/execution-surface-inventory.md` after implementation to show which surfaces are covered and which low-level helpers intentionally remain event-only or silent.

## 7. Validation

- [ ] 7.1 Run `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest -q tests/test_execution_logging_contract.py`.
- [ ] 7.2 Run `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest -q tests/integration/test_cli.py tests/test_glomeruli_resize_screening_workflow.py tests/test_training_entrypoint_contract.py tests/test_glomeruli_candidate_comparison.py`.
- [ ] 7.3 Run focused quantification tests affected by logging instrumentation, including `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest -q tests/unit/test_quantification_pipeline.py`.
- [ ] 7.4 Run the full repo test suite with `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest -q`.
- [ ] 7.5 Run `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m ruff check .`.
- [ ] 7.6 Run `python3 scripts/check_openspec_explicitness.py openspec/changes/p1-repo-wide-execution-logging-contract`.
- [ ] 7.7 Run `env OPENSPEC_TELEMETRY=0 openspec validate p1-repo-wide-execution-logging-contract --strict`.
- [ ] 7.8 Run `env OPENSPEC_TELEMETRY=0 openspec validate --specs --strict`.
- [ ] 7.9 Run `rg -n "logs/direct|logs/run_config" README.md docs/OUTPUT_STRUCTURE.md docs/ONBOARDING_GUIDE.md docs/TECHNICAL_LAB_NOTEBOOK.md docs/SEGMENTATION_ENGINEERING_GUIDE.md` and confirm current docs describe both direct and run-config log roots where relevant.
- [ ] 7.10 Run the new governance regression test that fails OpenSpec changes touching execution surfaces without logging-contract and docs-impact notes.
