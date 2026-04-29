## 1. Reuse-First Docs Audit

- [ ] 1.1 Inventory active docs for historical fallback modules, legacy namespace shims, fallback inference paths, unsupported command examples, and operational-looking historical plans.
- [ ] 1.2 Inventory existing docs index and check surfaces, including `docs/README.md`, `docs/HISTORICAL_NOTES.md`, `docs/archive/`, tests under `tests/`, and `scripts/check_openspec_explicitness.py`.
- [ ] 1.3 Decide whether `docs/INTEGRATION_GUIDE.md` and `docs/PIPELINE_INTEGRATION_PLAN.md` remain active current docs or become archive-only references, based on inbound links and current workflow relevance.
- [ ] 1.4 Record any new docs-check helper justification if existing pytest/check infrastructure cannot express the quarantine rule.

## 2. Archive Historical Material

- [ ] 2.1 Move retained historical content from `docs/INTEGRATION_GUIDE.md` to `docs/archive/` if the active guide cannot be rewritten without historical operational snippets.
- [ ] 2.2 Move retained historical content from `docs/PIPELINE_INTEGRATION_PLAN.md` to `docs/archive/` if the active plan cannot be rewritten as current workflow guidance.
- [ ] 2.3 Ensure every archived historical file has a reference-only header and does not present itself as current operational guidance.
- [ ] 2.4 Update `docs/HISTORICAL_NOTES.md` to index the archived historical integration/planning material.
- [ ] 2.5 Update `docs/README.md` so historical material is discoverable only through archive/reference paths.

## 3. Rewrite Active Docs To Current State

- [ ] 3.1 Remove active-doc snippets importing `eq.inference.historical_glomeruli_inference`, `HistoricalGlomeruliInference`, or `setup_historical_environment`.
- [ ] 3.2 Remove active-doc commands that execute `historical_glomeruli_inference.py` or instruct users to copy historical inference modules into production code.
- [ ] 3.3 Replace active fallback-loading language with current supported artifact loading and fail-closed behavior.
- [ ] 3.4 Ensure `README.md` and onboarding guidance point to current environment setup, `eq run-config`, and supported YAML workflows.
- [ ] 3.5 Ensure active segmentation docs describe current-namespace supported artifacts and do not recommend legacy fallback loaders.

## 4. Docs Quarantine Check

- [ ] 4.1 Add or extend an existing pytest/check surface to scan active docs for blocked historical module names and fallback-operation phrases.
- [ ] 4.2 Exempt `docs/archive/` and `docs/HISTORICAL_NOTES.md` only when historical content has reference-only framing.
- [ ] 4.3 Add test fixtures or assertions proving active docs fail on `eq.inference.historical_glomeruli_inference`, `setup_historical_environment`, and historical fallback loading.
- [ ] 4.4 Add assertions proving archived historical files remain allowed when indexed and framed as reference-only.
- [ ] 4.5 Ensure the check reports actionable file paths and matched phrases.

## 5. Link And Surface Consistency

- [ ] 5.1 Update any README/docs links that point to moved historical material.
- [ ] 5.2 Verify active docs that remain under their current names have current-state titles and introductory text.
- [ ] 5.3 Verify no active docs describe unsupported historical modules as immediate, proven, ready, fixed, or recommended current actions.
- [ ] 5.4 Verify docs do not introduce new workflow entrypoints or duplicate command guides.

## 6. Validation

- [ ] 6.1 Run the docs quarantine check or focused tests added by this change.
- [ ] 6.2 Run `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest -q`.
- [ ] 6.3 Run `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m ruff check .`.
- [ ] 6.4 Run `OPENSPEC_TELEMETRY=0 openspec validate oracle-current-docs-quarantine --strict`.
- [ ] 6.5 Run `python3 scripts/check_openspec_explicitness.py oracle-current-docs-quarantine`.
- [ ] 6.6 Record archived files, active docs rewritten, docs-check surface reused or added, validation results, and residual risks in the implementation closeout.
