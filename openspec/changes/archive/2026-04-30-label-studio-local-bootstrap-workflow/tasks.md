## 1. Bootstrap Planning Tests

- [x] 1.1 Add tests for recursive image discovery, supported suffix filtering, `source_relative_path`, `source_filename`, and `subject_hint` generation.
- [x] 1.2 Add tests for runtime path planning under `get_active_runtime_root() / "labelstudio"` and custom `--runtime-root`.
- [x] 1.3 Add tests for dry-run output that does not call Docker or the Label Studio API.

## 2. Bootstrap Module

- [x] 2.1 Add `src/eq/labelstudio/bootstrap.py` with image discovery and task payload generation.
- [x] 2.2 Add runtime layout planning for `data/`, `media/`, `imports/`, and `bootstrap/`.
- [x] 2.3 Add Docker command planning and Docker availability checks.
- [x] 2.4 Add Label Studio API helpers for readiness polling, project create/reuse, label config update, and task import using stdlib HTTP calls.
- [x] 2.5 Add dry-run execution that writes the task manifest and prints the planned URL/project/image count without starting Docker or calling the API.

## 3. CLI Integration

- [x] 3.1 Add an `eq labelstudio start --images <image-dir>` subcommand in `src/eq/__main__.py`.
- [x] 3.2 Add CLI options for `--project-name`, `--runtime-root`, `--port`, `--container-name`, `--docker-image`, `--username`, `--password`, `--api-token`, `--timeout-seconds`, and `--dry-run`.
- [x] 3.3 Ensure missing image directories fail before Docker or API work starts.

## 4. Documentation

- [x] 4.1 Update `docs/LABEL_STUDIO_GLOMERULUS_GRADING.md` with the one-command prospective-user workflow.
- [x] 4.2 Document the runtime artifact directory and local-only scope of the bootstrap command.

## 5. Validation

- [x] 5.1 Run focused bootstrap and glomerulus grading tests with `/Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest`.
- [x] 5.2 Run `ruff check` on changed Python files.
- [x] 5.3 Run `openspec validate label-studio-local-bootstrap-workflow --strict`.
- [x] 5.4 Run `python3 scripts/check_openspec_explicitness.py label-studio-local-bootstrap-workflow`.
