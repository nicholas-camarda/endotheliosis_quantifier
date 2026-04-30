## Context

The repo now has a glomerulus-instance Label Studio grading contract, but a prospective admin still cannot run it without manual Label Studio setup. The collaborator-facing goal requires a one-command local path: point at a directory of images, start Label Studio, import the images, and open a ready-to-grade project.

Label Studio supports Docker startup with `LABEL_STUDIO_USERNAME`, `LABEL_STUDIO_PASSWORD`, `LABEL_STUDIO_USER_TOKEN`, local file serving, and HTTP API project/task import. This change wraps those pieces in `eq` while keeping heavy Label Studio data outside Git.

## Goals / Non-Goals

**Goals:**

- Add `eq labelstudio start --images <image-dir>`.
- Recursively discover supported images and preserve relative path metadata.
- Create a runtime-local Label Studio workspace under the active runtime root.
- Start or reuse a local Docker Label Studio container.
- Configure the glomerulus grading project from `configs/label_studio_glomerulus_grading.xml`.
- Import discovered images as Label Studio tasks.
- Print the Label Studio URL and project URL.
- Provide dry-run planning for tests and for users who want to inspect actions before starting Docker.

**Non-Goals:**

- Do not deploy segmentation, quantification, model second review, MedSAM/SAM, or adjudication.
- Do not implement production/shared hosting.
- Do not vendor Label Studio into `eq-mac`.
- Do not commit imported images, Label Studio databases, task manifests, exports, or generated runtime files.
- Do not replace the glomerulus-instance grading parser.

## Explicit Decisions

- CLI surface: `eq labelstudio start --images <image-dir>`.
- Implementation owner: `src/eq/labelstudio/bootstrap.py`.
- CLI wiring owner: `src/eq/__main__.py`.
- Default runtime directory: `get_active_runtime_root() / "labelstudio"`.
- Runtime subdirectories: `data/`, `media/`, `imports/`, `bootstrap/`.
- Default container name: `eq-labelstudio`.
- Default Docker image: `heartexlabs/label-studio:latest`.
- Default username: `eq-admin@example.local`.
- Default password: `eq-labelstudio`.
- Default API token: `eq-local-token`.
- Default project title: `EQ Glomerulus Grading`.
- Default port: `8080`.
- Supported image suffixes: `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`.
- Local task image URLs use `/data/local-files/?d=<relative-path>` under the mounted media root.
- The command copies no image bytes by default; it mounts the source image directory read-only into the Label Studio container and writes generated task manifests under the runtime root.

## Decisions

### Use Docker first

Docker is the first supported one-command backend because it avoids installing Label Studio into `eq-mac` and gives the command a predictable runtime boundary. The command should fail clearly when Docker is missing unless `--dry-run` is used.

Alternative considered: install/run `label-studio` directly in the Python environment. That risks dependency conflicts with PyTorch/MPS and violates the separation between annotation app and scientific runtime.

### Use JSON export/API contract first

The command should use the Label Studio HTTP API to create/update the project and import tasks, but should also write the generated import JSON under `labelstudio/imports/` for auditability.

Alternative considered: ask the user to import JSON manually. That does not meet the one-command requirement.

### Preserve source paths as metadata

Each task should include `source_relative_path`, `source_filename`, and `subject_hint`. The first `subject_hint` rule is the first relative path component when images are nested; otherwise it is the image stem.

Alternative considered: flatten all image identity to filenames. That loses useful animal/kidney folder context.

### Keep dry-run complete

`--dry-run` should build the same plan and task manifest but not call Docker or the Label Studio API. This makes testing deterministic and lets admins see exactly what will be imported.

Alternative considered: only test helper functions. That would leave the actual command behavior under-specified.

## Risks / Trade-offs

- Label Studio API behavior may vary by version → Mitigation: isolate HTTP calls in `bootstrap.py`, test request construction, and fail with actionable diagnostics when the API is incompatible.
- Docker may not be installed or running → Mitigation: fail before modifying runtime state unless `--dry-run` is used.
- Port 8080 may already be in use → Mitigation: make `--port` configurable and include the chosen URL in all API calls and output.
- Large image directories may create huge task imports → Mitigation: recursive discovery is deterministic; future changes can add batching if needed.
- Local file serving can expose too much filesystem state → Mitigation: mount only the requested image directory read-only and set `LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/label-studio/media`.

## Migration Plan

1. Add tests for image discovery, dry-run planning, task payload generation, and CLI dispatch.
2. Add `src/eq/labelstudio/bootstrap.py` with no model dependencies.
3. Wire `eq labelstudio start`.
4. Add docs showing `eq labelstudio start --images /path/to/images`.
5. Validate with focused tests, ruff, OpenSpec validation, and explicitness checks.

Rollback is simple: do not run the new command. Existing Label Studio grading parser and quantification workflows remain unchanged.

## Open Questions

- [audit_first_then_decide] Which Label Studio API response fields identify an existing matching project reliably enough for idempotent reuse? Audit against a local Docker Label Studio instance during implementation.
- [defer_ok] Should browser opening be opt-in through `--open` or enabled by default on local macOS? Stage 1 can print URLs only.
- [defer_ok] Should production/shared hosting use the same command or a separate deployment command? This change targets local/admin bootstrap only.