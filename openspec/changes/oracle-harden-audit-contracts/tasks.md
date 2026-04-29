## 0. Reuse-First Contract Inventory

- [ ] 0.1 Inventory existing owners before adding or centralizing helpers for segmentation preprocessing, threshold provenance, Label Studio grade extraction, canonical naming, manifest pairing, candidate provenance, config validation, run logging, quantification estimability, source-confounding diagnostics, and sklearn serialization.
- [ ] 0.2 Record for each owner whether the change will reuse, extend, or replace the existing surface, and why any new helper is required.
- [ ] 0.3 Confirm this change consumes the completed canonical quantification input contract for resolved labels, grouping identity, target-defining hashes, and override provenance rather than creating a second target-definition path.
- [ ] 0.4 Record the canonical-input handoff: this change owns current Label Studio extraction and historical backfill rejection, but it must not change the canonical target-definition version without updating provenance and tests.
- [ ] 0.5 Audit `configs/*.yaml`, `analysis_registry.yaml`, and active runtime artifact metadata to produce a key-by-key exact-artifact-handoff plan before config edits proceed; committed defaults must not depend on local-only runtime paths.

## 1. Confirm And Invert Fail-Open Tests

- [ ] 1.1 Audit tests that currently assert historical Label Studio backfill, latest-artifact selection, resize-screening fallback, fallback provenance, permissive mask lookup, or global MPS fallback behavior.
- [ ] 1.2 Replace the existing Label Studio backfill test with `test_labelstudio_latest_missing_grade_fails_closed` and `test_labelstudio_auto_discovery_does_not_search_git_history`.
- [ ] 1.3 Replace resize-screening fallback assertions with `test_resize_screening_fallback_config_is_rejected`.
- [ ] 1.4 Add or update candidate-comparison tests so requested training failure raises a workflow error instead of producing a supported unavailable-candidate report.

## 2. Harden Label Studio And Manifest Quantification Inputs

- [ ] 2.1 Remove `DEFAULT_HISTORICAL_SOURCES` auto-discovery and `fallback_latest_missing_grade` behavior from `src/eq/quantification/labelstudio_scores.py`.
- [ ] 2.2 Add a shared Label Studio grade extractor used by `src/eq/quantification/labelstudio_scores.py` and `src/eq/quantification/cohorts.py`.
- [ ] 2.3 Reject ambiguous multi-choice Label Studio grade results unless exactly one documented shared rule applies.
- [ ] 2.4 Make `src/eq/quantification/pipeline.py` keep `manifest.csv` roots in manifest mode and reject `score_source=labelstudio` or `annotation_source` for manifest-root runs.
- [ ] 2.5 Add tests for latest missing grade, no git-history auto-discovery, shared multi-choice handling, manifest root with Label Studio options, and absence of `raw_inventory.csv` in manifest mode.

## 3. Harden Manifest Training Pairing And Split Provenance

- [ ] 3.1 Update `src/eq/data_management/datablock_loader.py` so manifest-admitted training rows validate exact readable image/mask pairs before DataBlock construction.
- [ ] 3.2 Return explicit manifest pair records from manifest-backed item selection instead of bare image paths where pair identity matters.
- [ ] 3.3 Remove permissive parent-directory, normalized-stem, or root-level same-stem mask fallback searches from `src/eq/data_management/standard_getters.py`.
- [ ] 3.4 Thread explicit split seed from training workflow config into dynamic-patching split construction and run metadata.
- [ ] 3.5 Add tests for missing admitted image, missing admitted mask, alternate same-stem mask rejection, outside-`images/` mask lookup rejection, split-seed provenance, and explicit split-manifest precedence.

## 4. Unify Segmentation Preprocessing And Threshold Provenance

- [ ] 4.1 Consolidate segmentation inference preprocessing in `src/eq/inference/prediction_core.py` as the shared ImageNet-normalized preprocessing function.
- [ ] 4.2 Replace unit-scaled preprocessing in `src/eq/quantification/pipeline.py`, `src/eq/quantification/embeddings.py`, and `src/eq/inference/gpu_inference.py` with the shared function.
- [ ] 4.3 Remove hard-coded `0.01` threshold defaults from GPU inference and use explicit threshold input or `DEFAULT_PREDICTION_THRESHOLD`.
- [ ] 4.4 Record preprocessing and threshold provenance in embedding summaries, inference outputs, and quantification review artifacts.
- [ ] 4.5 Add tests for preprocessing equivalence to training ImageNet stats, GPU inference preprocessing, default threshold source, explicit threshold use, and provenance recording.

## 5. Fail Closed On ROI Geometry

- [ ] 5.1 Update union ROI extraction in `src/eq/quantification/pipeline.py` to require image/mask shape agreement before crop geometry is computed.
- [ ] 5.2 Remove all-positive fallback when every connected component is below `min_component_area`.
- [ ] 5.3 Record explicit ROI statuses such as `image_mask_size_mismatch` and `component_below_min_area` and prevent ROI crop writes for those rows.
- [ ] 5.4 Ensure quantification artifact manifests and review tables summarize non-crop ROI statuses.
- [ ] 5.5 Add tests for small components without all-positive fallback, image/mask size mismatch, no crop written for invalid ROI, and downstream exclusion of invalid ROI rows.

## 6. Harden Candidate Comparison, Artifact Handoffs, And Training Provenance

- [ ] 6.1 Make `_run_training_command` in `src/eq/training/compare_glomeruli_candidates.py` raise on nonzero subprocess status when training is requested.
- [ ] 6.2 Remove `_build_fallback_provenance` from supported transfer and no-base candidate paths and reject artifacts without current metadata before evaluation.
- [ ] 6.3 Remove `_latest_artifact_from_glob` and `_latest_pkl` supported handoff behavior from `src/eq/run_config.py` and require exact artifact paths in committed configs.
- [ ] 6.4 Remove `--allow-auto-base-model` and `find_best_mitochondria_model` from `src/eq/training/train_glomeruli.py`.
- [ ] 6.5 Reject resize-screening configs with `fallback_run_id` or `run_if: primary_failed` in `src/eq/training/run_glomeruli_candidate_comparison_workflow.py`.
- [ ] 6.6 Make split manifest, training history, git state, package versions, data root, training command, and training mode mandatory for supported training exports in `src/eq/utils/run_io.py` and training callers.
- [ ] 6.7 Add tests for requested training failure, missing current metadata rejection, compatibility audit labeling, exact artifact path validation, no auto-base parser flag, transfer without base failure, resize fallback rejection, and provenance-required export blocking.

## 7. Harden Shared Quantification Modeling Contracts

- [ ] 7.0 Treat Oracle findings 1, 2, and 3 as one shared modeling-contract problem: no estimable candidates, unestimable grouped folds, and unsupported serialization must be solved in reusable helpers rather than P3-local patches.
- [ ] 7.1 Extend `src/eq/quantification/modeling_contracts.py` with shared helpers for candidate/fold target estimability, empty-candidate result construction, hard-blocker payloads, insufficient-data verdict payloads, and supported sklearn model serialization.
- [ ] 7.2 Refactor `src/eq/quantification/endotheliosis_grade_model.py` so empty P3 candidate sets write `current_data_insufficient` artifacts with `no_estimable_candidate_features` or equivalent hard blockers instead of crashing during prediction concatenation.
- [ ] 7.3 Refactor severe and ordinal P3 candidate fitting so each grouped training fold is checked for required target-class support before sklearn fitting; unestimable folds or candidates must be recorded as hard blockers or skipped diagnostics, not uncaught sklearn errors.
- [ ] 7.4 Replace `pickle.dump` writes for `model/final_model.joblib` with the shared supported sklearn serialization helper, or rename the artifact contract to `.pkl` everywhere in code, specs, docs, and tests.
- [ ] 7.5 Add regression coverage for no estimable candidate features, one-class grouped severe folds, one-class grouped ordinal folds, supported `.joblib` serialization/loadability, and hard-blocker propagation into `summary/final_product_verdict.json` plus family diagnostics.
- [ ] 7.6 Add shared source/cohort confounding diagnostics and source-stratified minimum-support checks to the reusable modeling-contract helpers where supported by the current evaluator inputs.
- [ ] 7.7 Add tests proving a candidate or fold dominated by one source/cohort is blocked, downgraded, or explicitly diagnosed rather than promoted as deployable evidence.
- [ ] 7.8 Record in the implementation closeout which non-P3 quantification evaluators should be migrated next to the shared modeling-contract helpers; do not refactor older evaluators in this change unless needed to remove a fail-open path touched here.

## 8. Harden CLI Startup, Canonical Naming, And MPS Scope

- [ ] 8.1 Remove import-time auto-setup suppression from `src/eq/__init__.py` and fail visibly for commands that require failed setup.
- [ ] 8.2 Replace broad `except: pass` startup paths in `src/eq/__main__.py` with fail-closed errors for runtime directory creation and invalid mode parsing.
- [ ] 8.3 Remove global Darwin `PYTORCH_ENABLE_MPS_FALLBACK=1` mutation from CLI startup and keep fallback scoped to explicit real Mac segmentation training/validation commands.
- [ ] 8.4 Consolidate canonical name parsing between `src/eq/data_management/canonical_naming.py` and `src/eq/data_management/canonical_contract.py` and require full-match parsing.
- [ ] 8.5 Add tests for runtime directory creation failure, invalid mode rejection, non-training Darwin command not setting MPS fallback, canonical parser trailing-junk rejection, and consistent non-`T` subject handling.

## 9. Docs, Configs, And Validation

- [ ] 9.1 Update committed configs, especially `configs/glomeruli_candidate_comparison.yaml` and `configs/endotheliosis_quantification.yaml`, so they reflect exact artifact references and hardened options.
- [ ] 9.2 Update README/docs/OpenSpec-facing prose only where current behavior changes; avoid historical or migration framing in public docs.
- [ ] 9.3 Run targeted tests for each hardened surface added in this change.
- [ ] 9.4 Run `python -m pytest -q`.
- [ ] 9.5 Run `ruff check .` and fix actionable lint failures introduced by this change.
- [ ] 9.6 Run `openspec validate oracle-harden-audit-contracts --strict`.
- [ ] 9.7 Run `python3 scripts/check_openspec_explicitness.py oracle-harden-audit-contracts`.

## 10. Postflight And Archive Lifecycle

- [ ] 10.1 Complete the per-change postflight required by `openspec/changes/ACTIVE_EXECUTION_ORDER.md`, including spec-to-diff review, completed-task evidence review, `git diff --check`, `git diff --stat`, and unrelated-edit inspection.
- [ ] 10.2 Commit the implementation as `implement oracle-harden-audit-contracts`.
- [ ] 10.3 Archive/sync with `openspec archive oracle-harden-audit-contracts --yes`.
- [ ] 10.4 Run `openspec validate --specs --strict` after archive/sync.
- [ ] 10.5 Revalidate every remaining active change with `openspec validate <remaining-change> --strict` and `python3 scripts/check_openspec_explicitness.py <remaining-change>`.
- [ ] 10.6 Commit the archive/sync as `archive oracle-harden-audit-contracts`.
