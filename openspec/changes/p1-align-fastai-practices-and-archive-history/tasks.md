## 1. Reuse-First Inventory

- [x] 1.1 Inventory existing scripts, pytest files, path helpers, CLI dry-runs, and docs index surfaces that can support lint, historical-doc quarantine, active-code hygiene, link/path validation, and OpenSpec explicitness checks.
- [x] 1.2 Record which existing surfaces will be reused or extended, including `scripts/check_openspec_explicitness.py`, current pytest files under `tests/`, `src/eq/utils/paths.py`, `src/eq/run_config.py`, `src/eq/__main__.py`, `docs/HISTORICAL_NOTES.md`, and `docs/README.md`.
- [x] 1.3 If any new standalone script is still needed, document in the task closeout why existing pytest/helpers/scripts could not express the requirement cleanly.
- [x] 1.4 Identify duplicate or overlapping active docs/checks discovered during the inventory and either consolidate them or record why both must remain.

## 2. FastAI Import And Lint Hygiene

- [x] 2.1 Replace wildcard FastAI imports in `src/eq/training/train_glomeruli.py` with explicit imports for `Learner`, `unet_learner`, `resnet34`, `Dice`, `JaccardCoeff`, `load_learner`, and any other symbols actually used.
- [x] 2.2 Replace wildcard FastAI imports in `src/eq/training/train_mitochondria.py` with explicit imports for learner construction, metrics, callbacks, and any FastAI helpers actually used.
- [x] 2.3 Replace wildcard FastAI imports in `src/eq/training/transfer_learning.py` with explicit imports for learner construction, metrics, serialization, and mixed-precision helpers actually used.
- [x] 2.4 Resolve remaining `ruff check .` findings with mechanical, behavior-preserving edits only; avoid broad refactors unrelated to the lint gate.
- [x] 2.5 Add or update tests if import changes expose a training-entrypoint contract gap.

## 3. Fail-Closed Training Pairing

- [x] 3.1 Remove the secondary fallback mask-candidate search from `src/eq/data_management/datablock_loader.py::get_items_full_images`.
- [x] 3.2 Ensure direct full-image roots fail before model construction when `get_y_full(...)` cannot resolve masks for any image.
- [x] 3.3 Ensure manifest-backed `raw_data/cohorts` roots continue to enumerate only admitted rows with explicit image and mask paths.
- [x] 3.4 Add regression coverage for an unpaired direct training root that confirms no fallback pairing is attempted, reusing an existing DataBlock or training-contract test file if possible.
- [x] 3.5 Add regression coverage for a manifest-backed root that confirms explicit manifest paths remain accepted, reusing existing cohort/path tests if possible.

## 4. Required Artifact Handling

- [x] 4.1 Classify `save_splits(...)`, `save_training_history(...)`, `export_final_model(...)`, and `save_run_metadata(...)` outputs as required evidence artifacts in `src/eq/utils/run_io.py`.
- [x] 4.2 Change required artifact write failures from warning-only behavior to explicit exceptions that include the failed path.
- [x] 4.3 Keep optional plot-only artifacts warning-only, with artifact path and exception details in the warning.
- [x] 4.4 Add tests for required artifact failure behavior and optional plot warning behavior, reusing an existing training-entrypoint or run-I/O test file if possible.
- [x] 4.5 Verify supported training metadata still records package versions, data root, training mode, split provenance, and exported model path.

## 5. Trusted Current-Namespace Learner Loading

- [x] 5.1 Review `src/eq/data_management/model_loading.py` and transfer base loading in `src/eq/training/transfer_learning.py` for legacy namespace shims or implicit compatibility rescue paths.
- [x] 5.2 Ensure current loader behavior fails closed or records unavailable evidence for unsupported legacy artifacts without adding monkey patches or compatibility imports.
- [x] 5.3 Update tests to assert current-namespace loading is runtime compatibility evidence only, not scientific promotion evidence.
- [x] 5.4 Update docs to describe FastAI `load_learner` as trusted current-namespace pickle loading, with unsupported legacy artifacts routed to archive/reference docs.

## 6. Historical Documentation Quarantine

- [x] 6.1 Create `docs/archive/` if it does not exist.
- [x] 6.2 Reconcile the already-applied `oracle-current-docs-quarantine` archive outputs before moving docs; if the historical content was already archived and indexed, mark the corresponding P1 archive move as satisfied/no-op with evidence instead of recreating or duplicating archive pages.
- [x] 6.3 Move historical content from `docs/INTEGRATION_GUIDE.md` into `docs/archive/fastai_legacy_integration.md` with a historical/reference-only header only when it was not already quarantined by the prior docs change, then keep or recreate `docs/INTEGRATION_GUIDE.md` as current implementation guidance if the guide remains.
- [x] 6.4 Move retained content from `docs/PIPELINE_INTEGRATION_PLAN.md` into `docs/archive/fastai_pipeline_integration_plan.md` with a historical/reference-only header only when it was not already quarantined by the prior docs change.
- [x] 6.5 Move retained content from `docs/HISTORICAL_IMPLEMENTATION_ANALYSIS.md` into `docs/archive/fastai_historical_implementation_analysis.md` with a historical/reference-only header only when it remains outside archive after the prior docs change.
- [x] 6.6 Update `docs/HISTORICAL_NOTES.md` as the index for the archived FastAI historical material.
- [x] 6.7 Update `docs/README.md` and `docs/HISTORICAL_NOTES.md` so archive material is accessible and discoverable without being presented as current workflow guidance.

## 7. Current Documentation Cleanup

- [x] 7.1 Audit `README.md` for historical FastAI helpers, fallback loader wording, migration framing, and legacy model rescue instructions; rewrite current workflow sections in present-state language only.
- [x] 7.2 Audit `docs/SEGMENTATION_ENGINEERING_GUIDE.md` for active guidance that conflicts with current FastAI trusted-artifact and full-image dynamic-patching contracts.
- [x] 7.3 Audit `docs/ONBOARDING_GUIDE.md` for historical setup or fallback material and keep the path through `eq-mac`, `eq run-config`, and current validation commands clear.
- [x] 7.4 Audit `docs/TECHNICAL_LAB_NOTEBOOK.md` for historical material that belongs in `docs/archive/` rather than active technical workflow sections.
- [x] 7.5 Add a documentation quarantine check by extending existing pytest/docs checks where possible; it must scan current docs, including `docs/INTEGRATION_GUIDE.md` if present, and fail if they contain active instructions for `historical_glomeruli_inference`, `setup_historical_environment`, legacy namespace shims, historical fallback loading, or workaround paths.
- [x] 7.6 Add or update an active-code hygiene check, reusing pytest or existing helper scripts where possible, that prevents historical FastAI shims, workaround branches, or compatibility rescue paths from reappearing in supported `src/eq/` workflows.

## 8. Validation

- [x] 8.1 Run `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m ruff check .`.
- [x] 8.2 Run `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest -q`.
- [x] 8.3 Run `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m eq --help`.
- [x] 8.4 Run `python3 scripts/check_openspec_explicitness.py openspec/changes/p1-align-fastai-practices-and-archive-history`.
- [x] 8.5 Run `OPENSPEC_TELEMETRY=0 openspec validate p1-align-fastai-practices-and-archive-history --strict`.
- [x] 8.6 Record the reuse-first inventory, any new-script justification, validation results, and residual risk in this change before implementation is considered complete.

## 9. Postflight And Archive Lifecycle

- [x] 9.1 Complete the per-change postflight required by `openspec/changes/ACTIVE_EXECUTION_ORDER.md`, including spec-to-diff review, completed-task evidence review, `git diff --check`, `git diff --stat`, and unrelated-edit inspection.
- [x] 9.2 Commit the implementation as `implement p1-align-fastai-practices-and-archive-history`.
- [ ] 9.3 Archive/sync with `openspec archive p1-align-fastai-practices-and-archive-history --yes`.
- [ ] 9.4 Run `openspec validate --specs --strict` after archive/sync.
- [ ] 9.5 Revalidate every remaining active change with `openspec validate <remaining-change> --strict` and `python3 scripts/check_openspec_explicitness.py <remaining-change>`.
- [ ] 9.6 Commit the archive/sync as `archive p1-align-fastai-practices-and-archive-history`.
