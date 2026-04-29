# Implementation Notes

## Reuse-First Audit

- Existing docs index surfaces reused: `docs/README.md` and `docs/HISTORICAL_NOTES.md`.
- Existing test infrastructure reused: pytest under `tests/unit/`.
- No new standalone docs-check script was added.

## Historical Material Quarantined

- `docs/INTEGRATION_GUIDE.md` now remains as current-state guidance only.
- Detailed historical content moved to `docs/archive/fastai_legacy_integration.md`.
- `docs/PIPELINE_INTEGRATION_PLAN.md` now remains as current-state guidance only.
- Detailed historical content moved to `docs/archive/fastai_pipeline_integration_plan.md`.
- `docs/HISTORICAL_IMPLEMENTATION_ANALYSIS.md` now remains as a pointer only.
- Detailed historical content moved to `docs/archive/fastai_historical_implementation_analysis.md`.

## Docs Check

- Added `tests/unit/test_docs_quarantine.py`.
- The check scans active docs for historical fallback operations and allows archive docs only when they have reference-only framing and are indexed from `docs/HISTORICAL_NOTES.md`.

## Validation

- `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest tests/unit/test_docs_quarantine.py -q` passed.
- Targeted active-doc phrase scan for historical fallback operations returned no matches.
- `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest -q` passed: 251 passed, 3 skipped.
- `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m ruff check .` passed.
- `OPENSPEC_TELEMETRY=0 openspec validate oracle-current-docs-quarantine --strict` passed.
- `python3 scripts/check_openspec_explicitness.py oracle-current-docs-quarantine` passed.

## Residual Risk

- Full-suite validation emitted only existing library/runtime warnings from PyTorch, pandas, and NumPy.

## Postflight

- Re-read `proposal.md`, `design.md`, `tasks.md`, and spec deltas against the diff.
- Confirmed the implementation is limited to docs quarantine, docs index updates, a pytest regression check, and OpenSpec task/notes updates.
- `git diff --check` passed.
- `git diff --stat` showed only docs/OpenSpec task changes before force-adding ignored current-state stubs and archive files.
- Targeted active-doc scan found no historical fallback operation matches.
