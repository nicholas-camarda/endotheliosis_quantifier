## Why

Oracle's static repo review identified multiple fail-open paths that can produce plausible-looking segmentation or quantification artifacts while violating the repository's current data, model, provenance, and scientific-claim contracts. This change hardens those paths before further P3 quantification work treats the outputs as reliable evidence.

## What Changes

- **BREAKING** Remove automatic historical Label Studio score recovery from git history; latest annotations without a current grade fail closed or are reported as missing rather than backfilled.
- **BREAKING** Require manifest-backed segmentation training to validate every admitted image/mask pair and reject permissive alternate mask searches.
- **BREAKING** Require quantification embedding and GPU inference preprocessing to use the same ImageNet-normalized preprocessing contract as segmentation training.
- **BREAKING** Make requested glomeruli candidate training, base-artifact selection, and model provenance fail closed instead of fabricating unavailable candidates or fallback provenance.
- **BREAKING** Remove "latest artifact" glob selection for supported workflow handoffs; configs must name exact upstream artifact paths.
- **BREAKING** Remove union-ROI all-positive fallback behavior and require explicit image/mask shape agreement before ROI crop generation.
- **BREAKING** Require quantification model evaluators to use shared estimability, hard-blocker, insufficient-data verdict, and supported-serialization helpers rather than local ad hoc gates.
- **BREAKING** Keep manifest quantification roots in manifest mode even when score options are passed; reject raw-project Label Studio mode from `raw_data/cohorts` manifest roots.
- **BREAKING** Remove CLI startup/import-time environment degradation paths and global Darwin MPS fallback mutation outside explicit training/validation commands.
- Replace hard-coded GPU inference threshold behavior with an explicit or contract-default threshold that is recorded in inference provenance.
- Require supported training artifacts to include mandatory split, history, and git/provenance metadata, with non-supported status or failure when metadata cannot be recorded.
- Remove direct glomeruli auto-base-model discovery and resize-screening conditional rescue behavior from supported workflows.
- Consolidate canonical-name parsing and Label Studio grade extraction into single shared rules that reject ambiguous inputs.
- Thread explicit split seeds from workflow config into dynamic patching provenance.

## Capabilities

### New Capabilities

- None.

### Modified Capabilities

- `scored-only-quantification-cohort`: Label Studio current-score recovery, manifest-mode selection, shared grade extraction, and manifest quantification fail-closed behavior.
- `segmentation-training-contract`: Manifest-admitted image/mask pair validation, exact mask pairing, training split seed provenance, supported training artifact metadata, and removal of auto-base-model discovery.
- `glomeruli-candidate-comparison`: Requested candidate training failure handling, provenance rejection, threshold/preprocessing evidence, exact base-artifact handoffs, and removal of resize-screening rescue branches.
- `workflow-config-entrypoints`: Exact artifact references for workflow handoffs and rejection of latest-glob discovery or implicit rescue workflows.
- `endotheliosis-burden-index`: Segmentation-backbone embedding preprocessing, ROI extraction fail-closed behavior, inference threshold provenance, and manifest-root quantification mode.
- `quantification-burden-artifact-layout`: Provenance fields for preprocessing, thresholds, ROI status, and non-written crop statuses in quantification artifacts.
- `endotheliosis-grade-model`: Shared quantification modeling contracts for candidate/fold estimability, hard-blocker propagation, insufficient-data verdicts, and supported sklearn artifact serialization.
- `repo-wide-execution-logging`: CLI startup/runtime directory setup failures and explicit MPS fallback scope.
- `macos-mps-local-development`: Mac MPS fallback use only for explicit real training/validation commands, not global CLI import/startup mutation.

## Explicit Decisions

- The change name is `oracle-harden-audit-contracts`.
- Oracle review group label: `oracle-grade-model-and-fail-closed-contracts`.
- Oracle findings directly covered: finding 1, no-candidate P3 run crashes; finding 2, sparse folds can crash sklearn; finding 3, `.joblib` artifact is written with pickle. The broader fail-closed hardening work in this change also covers additional Oracle fail-open contract findings from the same audit family.
- Label Studio score recovery will remove `DEFAULT_HISTORICAL_SOURCES` auto-discovery and `fallback_latest_missing_grade` behavior from `src/eq/quantification/labelstudio_scores.py`.
- Manifest-paired training validation will be implemented in `src/eq/data_management/datablock_loader.py` and `src/eq/data_management/standard_getters.py`.
- Shared segmentation preprocessing will use one explicit function in `src/eq/inference/prediction_core.py` and will be consumed by `src/eq/quantification/pipeline.py`, `src/eq/quantification/embeddings.py`, and `src/eq/inference/gpu_inference.py`.
- Shared quantification modeling contracts will live in `src/eq/quantification/modeling_contracts.py`; `src/eq/quantification/endotheliosis_grade_model.py` must consume those helpers for candidate/fold estimability, insufficient-data hard blockers, gate payloads, and supported model serialization.
- Source/cohort confounding diagnostics and source-stratified support checks will be added to the shared quantification modeling contracts where evaluator inputs contain source or cohort fields.
- This change consumes the completed canonical quantification input contract for resolved labels, grouping identity, target-defining hashes, and override provenance. It owns current Label Studio extraction and historical backfill rejection, but it must not change the canonical target-definition version without updating provenance and tests.
- Glomeruli candidate provenance hardening will target `src/eq/training/compare_glomeruli_candidates.py`, `src/eq/training/train_glomeruli.py`, and `src/eq/training/run_glomeruli_candidate_comparison_workflow.py`.
- Exact artifact handoffs will target `src/eq/run_config.py` and `configs/glomeruli_candidate_comparison.yaml`.
- ROI fail-closed behavior will target `src/eq/quantification/pipeline.py`.
- CLI startup hardening will target `src/eq/__main__.py` and `src/eq/__init__.py`.
- Regression tests will be added under `tests/unit/` and existing tests that assert fail-open behavior will be replaced or inverted.

## Open Questions

- [audit_first_then_decide] Which current tests intentionally assert historical fallback behavior and should be inverted versus deleted? Deciding audit target: `tests/unit/test_quantification_pipeline.py`, `tests/test_glomeruli_resize_screening_workflow.py`, and candidate-comparison tests.
- [audit_first_then_decide] Which existing local runtime artifacts become non-supported after provenance hardening? Deciding audit target: current metadata fields in `$EQ_RUNTIME_ROOT/models/segmentation/` and the active `configs/*.yaml` references.

## logging-contract

Durable logging behavior remains the existing `eq run-config` parent-log contract plus direct module logging. This change makes setup, subprocess, preprocessing, ROI, artifact-handoff, model-estimability, and provenance failures visible rather than adding a second logging system.

## docs-impact

README/docs/config guidance must describe the current fail-closed score, manifest, preprocessing, ROI, handoff, modeling-contract, and provenance behavior after implementation. Public docs should avoid historical migration framing.

## Impact

- Affected code: `src/eq/quantification/labelstudio_scores.py`, `src/eq/quantification/cohorts.py`, `src/eq/quantification/pipeline.py`, `src/eq/quantification/embeddings.py`, `src/eq/quantification/modeling_contracts.py`, `src/eq/quantification/endotheliosis_grade_model.py`, `src/eq/inference/prediction_core.py`, `src/eq/inference/gpu_inference.py`, `src/eq/data_management/datablock_loader.py`, `src/eq/data_management/standard_getters.py`, `src/eq/data_management/canonical_naming.py`, `src/eq/data_management/canonical_contract.py`, `src/eq/training/compare_glomeruli_candidates.py`, `src/eq/training/train_glomeruli.py`, `src/eq/training/run_glomeruli_candidate_comparison_workflow.py`, `src/eq/utils/run_io.py`, `src/eq/run_config.py`, `src/eq/__main__.py`, and `src/eq/__init__.py`.
- Affected configs: `configs/glomeruli_candidate_comparison.yaml` and `configs/endotheliosis_quantification.yaml`.
- Affected artifact contracts: segmentation model metadata, candidate-comparison reports, embedding summaries, inference provenance, ROI status/crop outputs, quantification model verdicts, supported sklearn model files, and quantification review artifacts under `output/quantification_results/`.
- Existing local artifacts that rely on latest-glob handoffs, fallback provenance, historical Label Studio backfill, or hard-coded inference thresholds may become compatibility or historical artifacts until regenerated under the hardened contracts.
