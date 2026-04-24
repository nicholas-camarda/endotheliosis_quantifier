## Why

The current glomeruli segmentation evidence is not rigorous enough to support README-facing performance claims or scientific promotion decisions: the latest high-Dice report uses a deterministic crop panel drawn from the admitted cohort root rather than a guaranteed held-out split, and the panel is foreground-heavy enough that broad oversegmentation can look strong. This change hardens the DataBlock, augmentation, preprocessing, training, evaluation, and documentation contract so candidate promotion is based on statistically defensible evidence rather than runtime compatibility or biased internal panels.

## What Changes

- Add pytest-backed validation tests for glomeruli segmentation under `tests/test_segmentation_validation_audit.py` and, for real local runtime artifacts, `tests/integration/test_glomeruli_segmentation_validation_audit_runtime.py`.
- Add reusable non-CLI audit helpers in `src/eq/training/segmentation_validation_audit.py` so tests can inspect data-contract, split-integrity, DataBlock sampling, augmentation, preprocessing, prediction-shape, metric-by-category, and documentation-claim behavior without exposing a new user-facing workflow.
- Modify candidate-comparison promotion evidence so deterministic review panels are held-out-only and report train/validation/test overlap explicitly before any `promoted` or `insufficient_evidence` decision is allowed.
- Modify segmentation-training provenance so exported glomeruli artifacts record enough split, sampler, augmentation, crop, preprocessing, and package information to audit whether training and promotion evaluation were statistically separable.
- Add hard failure behavior for biased promotion evidence: train/evaluation overlap, missing split provenance, foreground-heavy-only panels, all-foreground-like overcoverage, background false positives, or documentation claims that cite non-held-out evidence make a report not promotion-eligible for front-page claims.
- Separate runtime availability from promotion eligibility so the current scratch and transfer artifacts can remain available research-use candidates even when they do not support scientific promotion or README-facing performance claims.
- Keep `p2-add-negative-glomeruli-crop-supervision` separate: curated negative-crop supervision remains a related but distinct data-enrichment change, while this change hardens the validation and audit contract around whatever supervised data currently exists.

## Capabilities

### New Capabilities
- `segmentation-validation-audit`: Defines the pytest-backed glomeruli segmentation validation-audit contract, reusable audit helpers, Research Partner-style review lanes, and documentation-claim gates.

### Modified Capabilities
- `segmentation-training-contract`: Adds required split, DataBlock, augmentation, preprocessing, sampler, and training provenance needed to audit glomeruli segmentation artifacts.
- `glomeruli-candidate-comparison`: Tightens candidate promotion so deterministic manifests are held-out-only, category-balanced in a statistically meaningful way, and gated against oversegmentation and front-page documentation misuse.

## Impact

- Affected CLI/config surfaces: no new user-facing CLI or YAML config; existing `configs/glomeruli_candidate_comparison.yaml` and `eq.training.compare_glomeruli_candidates` behavior may be hardened.
- Affected modules: `src/eq/data_management/datablock_loader.py`, `src/eq/training/train_glomeruli.py`, `src/eq/training/transfer_learning.py`, `src/eq/training/compare_glomeruli_candidates.py`, `src/eq/training/promotion_gates.py`, and `src/eq/training/segmentation_validation_audit.py`.
- Affected tests: new or updated tests for split-overlap detection, deterministic manifest source restriction, DataBlock crop-distribution audit, augmentation alignment, preprocessing parity, prediction-shape gates, report schema generation, README/docs claim gating, and optional local runtime artifact checks.
- Affected artifacts: model sidecar metadata, split manifests, promotion reports, review panels, pytest-generated audit reports under test temp directories, `candidate_summary.csv`, `candidate_predictions.csv`, and docs that summarize current segmentation performance.

## Explicit Decisions

- No new `eq run-config` workflow, committed audit YAML, or user-facing audit CLI is introduced by this change.
- The new audit helper module is `src/eq/training/segmentation_validation_audit.py`.
- The primary unit/regression test file is `tests/test_segmentation_validation_audit.py`.
- The optional real-runtime integration test file is `tests/integration/test_glomeruli_segmentation_validation_audit_runtime.py`; it is run deliberately when local runtime artifacts or README-facing claims need verification.
- Pytest-generated audit reports are written under pytest-managed temporary directories, not under a stable runtime output root, unless candidate comparison itself is producing promotion artifacts.
- Candidate-comparison promotion reports remain the user-facing artifact surface for actual model evidence.
- Promotion-facing README or onboarding claims must cite only held-out promotion evidence that clears the hardened gates; compatibility-only, research-use-only, partly in-sample, not-promotion-eligible, or audit-failed reports may be discussed only as limitations or internal evidence.

## Open Questions

- [audit_first_then_decide] Whether the current admitted masked cohort can support a subject-held-out panel with enough positive, boundary, and background examples for every required category. The deciding evidence source is the optional runtime integration test `tests/integration/test_glomeruli_segmentation_validation_audit_runtime.py` when run against the current ProjectsRuntime artifacts.
- [defer_ok] Whether curated negative-crop supervision from `p2-add-negative-glomeruli-crop-supervision` should later become mandatory before scientific promotion. This change must expose the current negative/background limitation, but it does not need to solve negative-crop curation.
