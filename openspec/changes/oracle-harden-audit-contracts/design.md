## Context

Oracle's first implementation-audit lane found fail-open behavior across score recovery, manifest training, preprocessing, ROI extraction, quantification model evaluation, candidate comparison, artifact handoffs, training provenance, and CLI startup. These issues share one root problem: unsupported or ambiguous inputs can be converted into usable-looking outputs instead of being rejected or explicitly labeled as non-supported.

The affected workflows cross the main `eq` CLI, `eq run-config`, segmentation training, glomeruli candidate comparison, GPU inference, and endotheliosis quantification. The design therefore favors a staged fail-closed hardening pass with regression tests before any additional modeling claims are built on top of the current P3 quantification work.

## Goals / Non-Goals

**Goals:**

- Make current Label Studio scores, manifest-paired training inputs, segmentation preprocessing, ROI extraction, artifact handoffs, and supported model provenance fail closed.
- Remove compatibility and rescue paths that silently substitute older labels, alternate masks, latest artifacts, missing provenance, auto-discovered base models, or fallback resize workflows.
- Record preprocessing, threshold, ROI status, model estimability, split seed, and artifact provenance in outputs used by downstream review.
- Replace tests that assert fail-open behavior with regression tests for the hardened contracts.
- Preserve one canonical workflow surface through `eq run-config` and existing `src/eq` modules.

**Non-Goals:**

- Do not add support for legacy FastAI pickle artifacts, historical Label Studio auto-recovery, or old flat runtime layouts.
- Do not promote any segmentation model or quantification estimator scientifically.
- Do not run real Mac MPS training or validation inside this change unless a later implementation step explicitly needs unsandboxed local verification.
- Do not redesign the whole quantification model family or change the six-bin score rubric.
- Do not create fallback compatibility modes for users with stale local artifacts.

## Decisions

### Decision: Harden current contracts in dependency order

Implementation will start with data/label contracts, then preprocessing and ROI contracts, then shared quantification modeling contracts, then training/candidate provenance, then CLI/path behavior, then docs and OpenSpec validation.

Rationale: downstream model and artifact behavior cannot be interpreted until labels, image/mask pairs, preprocessing, and ROI geometry are trustworthy.

Alternative considered: implement all Oracle findings independently in any order. Rejected because it risks fixing downstream reports while upstream score or pairing corruption remains possible.

### Decision: Latest Label Studio annotation is authoritative

`src/eq/quantification/labelstudio_scores.py` will stop auto-discovering historical git sources through `DEFAULT_HISTORICAL_SOURCES` and will remove `fallback_latest_missing_grade`. Historical annotations can only be used through an explicit `annotation_source` such as `git:REV:path`, and missing current grades become missing-score statuses or hard failures according to the caller's contract.

Rationale: automatic historical backfill can silently replace a missing current grade with an older grade and corrupt every downstream model.

Alternative considered: keep auto-discovery but add warnings. Rejected because warnings do not prevent label substitution in generated artifacts.

### Decision: Manifest training uses explicit pair records

`src/eq/data_management/datablock_loader.py` will validate every admitted manifest training row before DataBlock construction and return explicit image/mask pair records rather than bare image paths. `src/eq/data_management/standard_getters.py` will require the exact mirrored mask path for images under `<root>/images/...`.

Rationale: the manifest is the training contract; permissive alternate searches can train on a different dataset than the manifest states.

Alternative considered: keep alternate searches as convenience fallback. Rejected because this repository treats data-contract mismatch as a hard failure.

### Decision: One segmentation-model preprocessing function

The supported segmentation inference/embedding path will use one function, owned by `src/eq/inference/prediction_core.py`, that applies the ImageNet normalization used by segmentation training. `pipeline.py`, `embeddings.py`, and `gpu_inference.py` will call that shared function, and generated summaries/provenance will record `inference_preprocessing="imagenet_normalized"`.

Rationale: encoder embeddings and segmentation probabilities must be contract-equivalent to training inputs before downstream quantification metrics are interpretable.

Alternative considered: normalize independently in each caller. Rejected because duplicated preprocessing will drift again.

### Decision: Requested training and artifact handoffs fail closed

`compare_glomeruli_candidates.py` will raise when requested training subprocesses fail and will reject supported candidate artifacts without current metadata. `_build_fallback_provenance`, `_latest_artifact_from_glob`, `_latest_pkl`, `--allow-auto-base-model`, and resize-screening `run_if: primary_failed` behavior will be removed from supported workflows.

Rationale: comparison reports may include explicitly unavailable or failed candidates only when the workflow is in a named audit/report mode, not when requested training or supported artifact evaluation is expected.

Alternative considered: preserve unavailable candidate rows for all failures. Rejected because it conflates report generation with successful supported training/evaluation.

### Decision: ROI extraction records non-crop statuses

Union ROI extraction will require image/mask shape agreement and will not fall back to the entire positive mask when all components fail the minimum-area gate. Rows with no eligible component will receive an explicit non-crop status such as `component_below_min_area`.

Rationale: tiny noise masks and shape mismatches must not become valid ROI crops.

Alternative considered: keep all-positive fallback but mark a warning. Rejected because downstream embeddings would still include invalid ROI evidence.

### Decision: Quantification model gates are shared contracts, not P3-local logic

`src/eq/quantification/modeling_contracts.py` will own reusable helpers for candidate/fold target estimability, empty-candidate result construction, hard-blocker payloads, insufficient-data verdict payloads, warning-capture interpretation, and supported sklearn model serialization. `src/eq/quantification/endotheliosis_grade_model.py` will use those helpers instead of local ad hoc gates.

Rationale: the same failure mode appears across quantification model families: no usable candidate features, one-class grouped training folds, numerically unstable fits, and model artifacts that look supported but cannot be loaded by the declared contract. These should fail closed in one reusable way so P3, burden, learned ROI, source-aware, severe-aware, and future evaluators do not each reinvent slightly different gates.

Alternative considered: patch the P3 selector directly for the current empty-candidate and sparse-fold crashes. Rejected because that would fix the immediate exception while preserving duplicated gate logic and making the next evaluator repeat the same mistake.

The shared helpers are not a new modeling framework. They are contract primitives: they decide whether a candidate/fold is estimable, how to record blockers when it is not, how to produce a bounded `current_data_insufficient` or diagnostic verdict without crashing, and how to write a supported sklearn artifact with a serialization method matching its filename.

### Decision: CLI startup is not a hidden environment manager

`src/eq/__main__.py` and `src/eq/__init__.py` will stop suppressing environment setup, mode parsing, and runtime directory failures. Darwin MPS fallback will not be set globally; it remains part of explicit real Mac segmentation training/validation commands.

Rationale: CLI startup must not silently change runtime behavior or hide setup failure.

Alternative considered: preserve import-time setup for convenience. Rejected because it violates the single explicit workflow contract.

## Explicit Decisions

- Oracle review group label: `oracle-grade-model-and-fail-closed-contracts`.
- Oracle review findings 1, 2, and 3 are implemented through the shared quantification modeling contract rather than through P3-local patches.
- The implementation order is: score contract, manifest pair contract, preprocessing/threshold contract, ROI contract, shared quantification modeling contract, candidate/provenance contract, workflow handoff contract, CLI/MPS contract, docs/spec/tests.
- Historical Label Studio sources require explicit caller input; no automatic git-history search is supported.
- Supported glomeruli base artifacts must be exact YAML paths, not latest-glob selections.
- `DEFAULT_PREDICTION_THRESHOLD` or an explicit caller threshold is the only default inference threshold contract; `0.01` is not supported as an implicit default.
- Quantification evaluators must not call sklearn fitting on an unestimable fold; they must record a hard blocker and continue to a bounded diagnostic or insufficient-data verdict.
- Quantification evaluators with source or cohort fields must use shared source/cohort confounding diagnostics and source-stratified support checks before any candidate is promoted as deployable evidence.
- The completed canonical quantification input contract owns resolved label provenance, grouping identity, target-defining hashes, and override provenance. This change may harden current Label Studio extraction and reject historical backfill, but it must not silently change the canonical target-definition version.
- A supported sklearn artifact filename and serialization method must agree; `.joblib` files are written with `joblib.dump`, while pickle output must use a `.pkl` contract.
- A training artifact missing mandatory split/history/git/provenance metadata is non-supported or the run fails before export.
- Tests that currently assert fallback behavior are implementation blockers and must be replaced before the change is considered complete.

## Risks / Trade-offs

- Existing local configs may stop running because they rely on latest-glob artifact discovery -> Mitigation: update committed configs to exact artifact references or mark unresolved local artifact selection as an implementation blocker.
- Existing local model artifacts may become historical/non-supported -> Mitigation: fail with provenance diagnostics and document the regeneration command rather than adding compatibility shims.
- Oracle findings may include false positives from static review -> Mitigation: each task begins with source-level confirmation and keeps only code-backed contract changes.
- Removing fallback behavior can reduce successful command completion in the short term -> Mitigation: tests and error messages must identify the exact missing score, pair, artifact, metadata, threshold, or ROI condition.
- Full real training validation is expensive -> Mitigation: unit and integration tests validate contract behavior; real MPS/CUDA runs are reserved for artifact promotion work.

## Migration Plan

1. Audit the current tests that assert fallback behavior and mark each for inversion or deletion.
2. Implement fail-closed label, manifest pair, preprocessing, threshold, ROI, quantification model-estimability, and artifact provenance checks behind the existing public commands.
3. Update committed YAML configs to use exact artifact paths where supported artifacts are required.
4. Regenerate only lightweight test fixtures and contract artifacts needed by unit/integration tests.
5. Run targeted tests for each hardened contract, then `python -m pytest -q`, `ruff check .`, and `openspec validate oracle-harden-audit-contracts --strict`.
6. Treat any runtime artifacts relying on removed fallback paths as historical/non-supported until regenerated under the new contracts.

## Open Questions

- [audit_first_then_decide] Which currently committed config keys can be resolved to exact supported artifact paths from repo-tracked config/registry alone, and which must remain user-provided explicit paths rather than local machine paths? Deciding evidence source: `configs/*.yaml`, `analysis_registry.yaml`, and active `$EQ_RUNTIME_ROOT/models/segmentation/` metadata; the required audit output is a key-by-key list that rejects local-only runtime metadata as committed defaults before config edits proceed.
- [audit_first_then_decide] Which candidate comparison unavailable-family report behavior remains valid for audit-only reports after training failures raise for requested supported runs? Evidence source: `src/eq/training/compare_glomeruli_candidates.py` report-mode call sites and tests.
