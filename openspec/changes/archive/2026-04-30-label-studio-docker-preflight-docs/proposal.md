## Why

`eq labelstudio start --images <dir>` must work for a prospective user without requiring them to know how Docker Desktop works. The current command can fail with a generic Docker daemon error even when Docker Desktop is installed but not running, and the README does not provide the copy-paste setup path.

## Explicit Decisions

- Change name: `label-studio-docker-preflight-docs`.
- Modified capability: `label-studio-local-bootstrap`.
- The command SHALL attempt to start Docker Desktop automatically on macOS when Docker is installed but the daemon is not running.
- If Docker Desktop is missing on macOS, the command SHALL fail with an explicit install command: `brew install --cask docker`.
- README SHALL include the copy-paste prospective-user path for `eq labelstudio start --images`.

## What Changes

- Improve Docker preflight in `src/eq/labelstudio/bootstrap.py`.
- Add focused tests for Docker Desktop auto-start and missing-install diagnostics.
- Update README with user-facing commands for installing/starting the local Label Studio workflow.
- Keep runtime artifacts outside Git and preserve the existing `eq labelstudio start --images <dir>` CLI surface.

## Capabilities

### New Capabilities

- None.

### Modified Capabilities

- `label-studio-local-bootstrap`: Add explicit Docker Desktop install/start preflight and README quick-start documentation for the one-command Label Studio workflow.

## Impact

- Affected code:
  - `src/eq/labelstudio/bootstrap.py`
  - `tests/unit/test_labelstudio_bootstrap.py`
- Affected docs:
  - `README.md`
  - `docs/LABEL_STUDIO_GLOMERULUS_GRADING.md` if needed for command details.
- Compatibility:
  - Existing `eq labelstudio start --images` behavior remains the same when Docker is already running.
  - Dry-run mode remains side-effect free and does not attempt to start Docker.

## logging-contract

This change does not add a durable logging root or subprocess tee. Docker preflight diagnostics continue to use normal `eq` command output and exceptions.

## docs-impact

README must expose the one-command Label Studio workflow and Docker prerequisite in copy-paste form. Supporting docs may keep longer explanations.

## Open Questions

- [defer_ok] Should a future installer command manage Docker installation directly? This change only provides explicit install instructions and auto-starts an installed Docker Desktop app.
