## Why

Oracle finding 6 shows that active documentation still contains operational-looking instructions for missing historical fallback code such as `eq.inference.historical_glomeruli_inference`. Even with a short historical note at the top, the body still reads like an immediate integration guide and can send users toward unsupported modules and fallback workflows.

## What Changes

- **BREAKING** Active docs must not contain step-by-step instructions, code snippets, command examples, or priority plans for missing historical modules, legacy namespace shims, or fallback inference paths.
- Quarantine historical integration/planning material under `docs/archive/` and index it from `docs/HISTORICAL_NOTES.md` as reference-only material.
- Rewrite or remove active docs that currently look operational but refer to historical fallback code.
- Add a documentation quarantine check that scans active docs for known unsupported historical module names and fallback phrases.
- Update current docs to describe only the supported current workflow, especially `eq run-config`, current-namespace supported artifacts, and fail-closed model loading.
- Reuse existing docs index surfaces and pytest/check infrastructure before adding any new docs script.

## Capabilities

### New Capabilities

- None.

### Modified Capabilities

- `repo-wide-quality-review`: Active documentation review must reject operational guidance for missing historical modules, fallback loaders, and compatibility rescue paths.
- `workflow-config-entrypoints`: Current operational docs must present supported YAML workflows and current CLI entrypoints, not historical fallback integration plans.

## Explicit Decisions

- Change ID: `oracle-current-docs-quarantine`.
- Oracle finding covered: finding 6, active docs recommend missing historical fallback code.
- Active docs in scope: `README.md`, `docs/README.md`, `docs/INTEGRATION_GUIDE.md`, `docs/PIPELINE_INTEGRATION_PLAN.md`, `docs/SEGMENTATION_ENGINEERING_GUIDE.md`, `docs/ONBOARDING_GUIDE.md`, and `docs/TECHNICAL_LAB_NOTEBOOK.md`.
- Historical material may remain only under `docs/archive/` and must be reachable through `docs/HISTORICAL_NOTES.md`.
- Active docs must not include operational snippets importing `eq.inference.historical_glomeruli_inference`, calling `setup_historical_environment`, recommending automatic historical fallback loading, or instructing users to copy historical inference modules into production paths.
- The docs quarantine check should extend existing pytest or explicitness-check patterns where possible; do not add a standalone one-off script unless existing test infrastructure cannot express the rule.

## Open Questions

- [audit_first_then_decide] Which of `docs/INTEGRATION_GUIDE.md` and `docs/PIPELINE_INTEGRATION_PLAN.md` should remain as current docs after archive extraction versus be replaced by redirects to current workflow docs? Deciding audit target: references from `README.md`, `docs/README.md`, and any docs links tests.
- [audit_first_then_decide] Which historical phrases should be blocked exactly versus allowed inside `docs/archive/`? Deciding evidence source: current active docs and archived historical files after extraction.

## logging-contract

This change does not add or modify durable runtime logging. Validation evidence is produced by docs tests/checks and OpenSpec validation only.

## docs-impact

This change is documentation-facing by design. It quarantines historical material under `docs/archive/`, updates active docs to current-state guidance only, and adds a check so unsupported historical fallback operations cannot reappear in active docs.

## Impact

- Affected docs: `README.md`, `docs/README.md`, `docs/INTEGRATION_GUIDE.md`, `docs/PIPELINE_INTEGRATION_PLAN.md`, `docs/SEGMENTATION_ENGINEERING_GUIDE.md`, `docs/ONBOARDING_GUIDE.md`, `docs/TECHNICAL_LAB_NOTEBOOK.md`, `docs/HISTORICAL_NOTES.md`, and `docs/archive/`.
- Affected tests/checks: existing docs or OpenSpec explicitness tests under `tests/` or `scripts/check_openspec_explicitness.py` if reused.
- No source-code compatibility shims, fallback loaders, or historical rescue paths are introduced.
