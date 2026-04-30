## 1. Spec and contract foundation

- [ ] 1.1 Audit Label Studio image version + JSON export fields for predictions vs annotations (document findings in `implementation-notes.md`)
- [ ] 1.2 Finalize `configs/label_studio_medsam_hybrid.yaml` schema (mask release binding, companion URLs, enforcement flags) and add example file to Git
- [ ] 1.3 Add path helpers in `src/eq/utils/paths.py` for YAML + generated-mask release resolution (reuse existing helpers when present)

## 2. CLI ergonomics

- [ ] 2.1 Extend `src/eq/__main__.py` `labelstudio start` parser with positional `<image-dir>` while keeping `--images`
- [ ] 2.2 Thread parsed config + hybrid flags into `run_bootstrap`

## 3. Bootstrap preload + health

- [ ] 3.1 Implement mask release manifest reader + task prediction builder in `src/eq/labelstudio/bootstrap.py` or dedicated helper module justified in notes
- [ ] 3.2 Wire Local Files + Label Studio API import of hybrid tasks with optional predictions payload
- [ ] 3.3 Add companion health probe with fail-closed default and admin-only bypass flag surfaced via YAML
- [ ] 3.4 Update README + `docs/LABEL_STUDIO_GLOMERULUS_GRADING.md` with single-line collaborator command and pointer to YAML knobs

## 4. Export ingestion + validation

- [ ] 4.1 Extend `src/eq/labelstudio/glomerulus_grading.py` to emit hybrid lineage columns and enforce validator rules
- [ ] 4.2 Add fixtures under `tests/fixtures/labelstudio_glomerulus_instances/` covering auto preload, manual refine, box-assisted, contradictory lineage failure
- [ ] 4.3 Expand unit tests for parser + validator edge cases

## 5. Companion service (thin wrapper)

- [ ] 5.1 Define HTTP contract + README section for companion (separate small service or script) with MedSAM invocation details
- [ ] 5.2 Provide docker-compose fragment or documented launch steps keeping secrets out of Git

## 6. Validation gates

- [ ] 6.1 Run `openspec validate label-studio-medsam-hybrid-grading --strict`
- [ ] 6.2 Run `python3 scripts/check_openspec_explicitness.py label-studio-medsam-hybrid-grading`
- [ ] 6.3 Run focused pytest `tests/unit/test_labelstudio_*` and `ruff check` on touched modules
