## 1. Docker Preflight

- [x] 1.1 Add tests for macOS Docker Desktop auto-start when `docker info` initially fails.
- [x] 1.2 Add tests for missing Docker Desktop install diagnostics.
- [x] 1.3 Implement Docker preflight using `docker info`, macOS `open -a Docker`, and timeout polling.
- [x] 1.4 Create required Label Studio Local Files import storage entries before importing image tasks.
- [x] 1.5 Recreate incompatible local Docker containers while preserving the mounted Label Studio data directory.

## 2. User Documentation

- [x] 2.1 Update `README.md` with copy-paste Label Studio startup commands and Docker prerequisite.
- [x] 2.2 Confirm `docs/LABEL_STUDIO_GLOMERULUS_GRADING.md` remains consistent with README.
- [x] 2.3 Document the default local Docker Label Studio login and clarify that conda `label-studio` password resets do not affect the Docker instance.

## 3. Validation

- [x] 3.1 Run focused Label Studio tests.
- [x] 3.2 Run ruff on changed files.
- [x] 3.3 Run `openspec validate label-studio-docker-preflight-docs --strict`.
- [x] 3.4 Run `python3 scripts/check_openspec_explicitness.py label-studio-docker-preflight-docs`.
