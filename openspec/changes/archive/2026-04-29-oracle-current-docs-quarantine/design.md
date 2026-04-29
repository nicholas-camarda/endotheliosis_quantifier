## Context

Oracle finding 6 identified active documentation that still reads like immediate integration guidance for missing historical fallback code. `docs/INTEGRATION_GUIDE.md` and `docs/PIPELINE_INTEGRATION_PLAN.md` now begin with historical notes, but their bodies still include instructions to import `eq.inference.historical_glomeruli_inference`, call `setup_historical_environment`, and add fallback loading paths. That is not a harmless archive style: a reader can still follow the snippets and attempt unsupported modules.

The repo already has `docs/HISTORICAL_NOTES.md`, `docs/archive/`, and active docs surfaces. This change should reorganize and test the docs contract rather than adding more prose.

## Goals / Non-Goals

**Goals:**

- Ensure active docs describe only current supported workflows and fail-closed contracts.
- Move historical fallback instructions to `docs/archive/` with reference-only framing.
- Keep historical material discoverable from `docs/HISTORICAL_NOTES.md`.
- Add a reusable docs quarantine check using existing test/check infrastructure where possible.
- Keep README and onboarding guidance current-state only.

**Non-Goals:**

- Do not delete historical information merely because it is obsolete.
- Do not reintroduce historical loader code, namespace shims, or fallback inference support.
- Do not rewrite all documentation for style.
- Do not create a second docs index or separate docs site.

## Decisions

### Decision: Active Docs Cannot Contain Historical Operational Snippets

Active docs may mention that historical material exists, but they must not include code snippets, commands, copy instructions, or step-by-step plans for unsupported historical modules. The blocked active-doc patterns include:

- `eq.inference.historical_glomeruli_inference`
- `HistoricalGlomeruliInference`
- `setup_historical_environment`
- `use_historical_preprocessing`
- `historical fallback`
- instructions to copy historical inference modules into production code
- active "immediate integration" or "ready now" sections for missing historical paths

Alternative considered: keep the historical note at the top and leave the body untouched. That still presents unsupported instructions as executable guidance.

### Decision: Archive Retains Detail, Active Docs Retain Pointers

Detailed historical integration/planning material should move under `docs/archive/` with explicit reference-only headers. Active docs should either be rewritten as current workflow docs or replaced by short pointers to current docs plus `docs/HISTORICAL_NOTES.md`.

Alternative considered: delete the files. That loses potentially useful provenance and does not satisfy the user's preference to keep history accessible but quarantined.

### Decision: Reuse Existing Check Infrastructure

The quarantine check should be a pytest or extension to existing repo checks, not a one-off shell script, unless audit proves existing infrastructure cannot express the rule. It must scan active docs and exempt `docs/archive/` plus `docs/HISTORICAL_NOTES.md` where historical phrases are allowed with reference-only framing.

Alternative considered: rely on manual review. The same drift has already reappeared, so this needs an executable guard.

## Explicit Decisions

- Reuse checked surfaces: `docs/README.md`, `docs/HISTORICAL_NOTES.md`, `docs/archive/`, existing docs tests under `tests/`, and `scripts/check_openspec_explicitness.py` patterns if applicable.
- Active docs in scope: `README.md`, `docs/README.md`, `docs/INTEGRATION_GUIDE.md`, `docs/PIPELINE_INTEGRATION_PLAN.md`, `docs/SEGMENTATION_ENGINEERING_GUIDE.md`, `docs/ONBOARDING_GUIDE.md`, and `docs/TECHNICAL_LAB_NOTEBOOK.md`.
- Archive paths must remain under `docs/archive/`.
- No source-code compatibility support is part of this change.

## Risks / Trade-offs

- [Risk] Removing historical snippets from active docs may make old context harder to find. -> Mitigation: index the archive entries from `docs/HISTORICAL_NOTES.md`.
- [Risk] A phrase-based check may flag legitimate current docs. -> Mitigation: keep blocked patterns specific to missing historical modules and fallback operations, and exempt archive material.
- [Risk] Docs cleanup could sprawl. -> Mitigation: only edit active docs enough to remove unsupported operational guidance and point to current workflow docs.
- [Risk] Existing links may break after moves. -> Mitigation: update `docs/README.md`, `docs/HISTORICAL_NOTES.md`, and any docs link tests.

## Migration Plan

1. Inventory active docs for historical fallback modules, fallback-loader instructions, and unsupported operational snippets.
2. Move retained historical content into `docs/archive/` with reference-only headers.
3. Rewrite or replace active docs so they describe current supported workflows only.
4. Update `docs/HISTORICAL_NOTES.md` and `docs/README.md`.
5. Add a docs quarantine test/check that scans active docs and exempts archive/reference files.
6. Run docs-specific tests, `ruff check .`, `python -m pytest -q`, and OpenSpec validation.

## Open Questions

- [audit_first_then_decide] Whether `docs/INTEGRATION_GUIDE.md` should remain as a current guide or become a short redirect page after extraction. Decide by checking inbound links and current README/docs references.
- [audit_first_then_decide] Whether `docs/PIPELINE_INTEGRATION_PLAN.md` has any current-state content worth preserving outside archive. Decide by extracting all current command references and comparing them to `eq run-config` docs.
