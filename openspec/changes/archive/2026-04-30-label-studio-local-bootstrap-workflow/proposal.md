## Why

The glomerulus-instance Label Studio contract is not usable by collaborators or lab admins if setup requires manual XML copy/paste, task JSON authoring, and separate command-line glue. The next workflow must let an admin point `eq` at an image directory and open a ready-to-grade Label Studio project with one command.

## Explicit Decisions

- Change name: `label-studio-local-bootstrap-workflow`.
- New capability: `label-studio-local-bootstrap`.
- Primary command: `eq labelstudio start --images <image-dir>`.
- Default project title: `EQ Glomerulus Grading`.
- Default Label Studio URL: `http://localhost:8080`.
- Default runtime root: `get_active_runtime_root() / "labelstudio"`.
- Checked-in grading config: `configs/label_studio_glomerulus_grading.xml`.
- The first implementation uses local Docker Label Studio plus its HTTP API with a deterministic local admin account and token.
- The command recursively imports supported image files: `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`.
- Image task data MUST preserve `source_relative_path` and `subject_hint` metadata.
- The command is operational bootstrap only. It does not run segmentation, quantification, model second review, MedSAM/SAM, or adjudication.

## What Changes

- Add `eq labelstudio start --images <image-dir>` to create a local Label Studio runtime, start or reuse a Docker container, create/update a grading project from `configs/label_studio_glomerulus_grading.xml`, import image tasks, and print the project URL.
- Add a supporting `eq.labelstudio.bootstrap` module that owns image discovery, task manifest creation, Docker command planning, readiness polling, and Label Studio API calls.
- Add a deterministic local runtime layout under the active runtime root:
  - `labelstudio/data/`
  - `labelstudio/media/`
  - `labelstudio/imports/`
  - `labelstudio/bootstrap/`
- Add dry-run planning support so the command can be tested without requiring Docker or a running Label Studio service.
- Add tests for recursive image discovery, task payload generation, runtime path boundaries, dry-run output, and fail-closed behavior for missing image directories or missing Docker when not in dry-run.

## Capabilities

### New Capabilities

- `label-studio-local-bootstrap`: Defines one-command local Label Studio startup and project bootstrap from an image directory.

### Modified Capabilities

- None. The existing `label-studio-glomerulus-grading` contract remains the annotation/validation contract consumed by this bootstrap workflow.

## Impact

- Affected code:
  - `src/eq/__main__.py` adds an `eq labelstudio start` CLI surface.
  - `src/eq/labelstudio/bootstrap.py` owns local bootstrap behavior.
  - `src/eq/labelstudio/glomerulus_grading.py` remains the validation/export contract and is not broadened into deployment logic.
  - `src/eq/utils/paths.py` is reused through `get_active_runtime_root()`; no new path system is introduced.
- Affected config:
  - Reuses `configs/label_studio_glomerulus_grading.xml`.
- Affected runtime artifacts:
  - Runtime files are written under `get_active_runtime_root() / "labelstudio"` unless `--runtime-root` overrides the root.
  - No images, Label Studio databases, task imports, exports, or generated project artifacts are committed to Git.
- Affected tests:
  - Add focused unit tests for bootstrap planning and CLI dry-run behavior.
- Compatibility:
  - Existing quantification and Label Studio glomerulus grading validation remain unchanged.
  - Existing manually run Label Studio instances are not modified unless the user points the command at the same URL/container.

## logging-contract

This change adds a local execution surface but does not create a separate durable logging root or subprocess teeing system. The command uses normal `eq` CLI logging and prints actionable startup/project information to the terminal; if later promoted into YAML-first execution, durable command capture must use the existing `eq run-config` logging contract.

## docs-impact

Documentation must make the prospective-user path explicit: one command points at an image directory and opens a configured Label Studio project. Docs must distinguish this local bootstrap from model deployment, second review, and production/multi-user hosting.

## Open Questions

- [audit_first_then_decide] Which exact Label Studio API response fields identify an existing matching project reliably enough for idempotent reuse? Audit against a local Docker Label Studio instance during implementation.
- [defer_ok] Should `eq labelstudio start` automatically open the browser on macOS, or only print the URL by default with `--open` opt-in?
- [defer_ok] Should production/shared hosting use the same command or a separate deployment command? This change targets local/admin bootstrap only.
