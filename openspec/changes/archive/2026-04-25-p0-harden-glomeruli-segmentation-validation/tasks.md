## 1. Pytest Audit Surface

- [x] 1.1 Add `src/eq/training/segmentation_validation_audit.py` with pure helper functions for data-contract audit, split-overlap audit, DataBlock crop sampling audit, transform/preprocessing audit, prediction-shape audit, metric-by-category aggregation, and documentation-claim audit.
- [x] 1.2 Add `tests/test_segmentation_validation_audit.py` to exercise the helper functions with deterministic fixtures and synthetic artifacts.
- [x] 1.3 Add `tests/integration/test_glomeruli_segmentation_validation_audit_runtime.py` for explicit local-runtime checks against current ProjectsRuntime artifacts, guarded so it can be skipped when runtime artifacts are unavailable.
- [x] 1.4 Add current-failure reproduction helpers that reconstruct the exact mitochondria, glomeruli transfer, and glomeruli no-base/scratch panel examples from artifact paths, image paths, mask paths, crop boxes, resize policy, threshold, prediction tensor shape, and overlay-generation path.
- [x] 1.5 Confirm no new `eq run-config` workflow, committed audit YAML, or user-facing audit CLI is introduced by this change.

## 2. Training Provenance And Split Integrity

- [x] 2.1 Extend mitochondria base metadata in `src/eq/training/train_mitochondria.py` to record `mitochondria_training_scope`, `mitochondria_inference_claim_status`, physical `training/` item count, physical `testing/` item count, actual pretraining image/mask paths, split policy, resize/preprocessing settings, training command, and code/package provenance.
- [x] 2.2 Add tests proving that a mitochondria base trained with physical `testing/` included is allowed only as `mitochondria_training_scope=all_available_pretraining` and cannot support mitochondria held-out performance claims.
- [x] 2.3 Extend glomeruli training sidecar metadata in `src/eq/training/train_glomeruli.py` and `src/eq/training/transfer_learning.py` to record `split_seed`, `splitter_name`, `train_images`, `valid_images`, `manifest_rows`, lane/cohort counts, source image/mask size summaries, crop settings, positive-aware sampling settings, augmentation settings, mask-binarization semantics, learner preprocessing, candidate family, transfer-base artifact path, transfer-base training scope, and code/package provenance.
- [x] 2.4 Update split-manifest writing so `*_splits.json` records machine-readable `train_images` and `valid_images` for every glomeruli candidate artifact.
- [x] 2.5 Add a two-axis status classifier that preserves loadable artifacts as `runtime_use_status=available_research_use` while marking missing split or transfer-base provenance as `promotion_evidence_status=audit_missing`.
- [x] 2.6 Add tests proving split-overlap detection blocks promotion when a deterministic manifest includes training images or training subjects.
- [x] 2.7 Record the runtime decision outcome for mitochondria pretraining scope: keep `raw_data/mitochondria_data/training` only with `heldout_test_preserved`, or intentionally train a representation base on physical `training/` plus `testing` with `mitochondria_training_scope=all_available_pretraining` and mitochondria inference claims disabled.

## 3. DataBlock, Augmentation, And Preprocessing Audit

- [x] 3.1 Implement a DataBlock sampling audit that builds DataLoaders through `build_segmentation_dls_dynamic_patching` and writes `datablock_sampling_audit.csv` with train/validation crop foreground fractions and positive-crop support.
- [x] 3.2 Add transform-alignment checks that verify image and mask crops use identical spatial crop/resize semantics on fixture data.
- [x] 3.3 Add preprocessing-parity checks that record whether deterministic evaluation uses learner-consistent preprocessing and supported threshold semantics.
- [x] 3.4 Add resize-policy audit helpers that record source image/mask dimensions, `crop_size`, `output_size`, crop-to-output resize ratio, aspect-ratio policy, `resize_method`, image/mask interpolation, mask binarization after resize, prediction resize-back behavior, and threshold/resize ordering for training and deterministic evaluation.
- [x] 3.5 Add tests for crop-size propagation, positive-aware sampling metadata, image/mask transform alignment, binary mask preservation after resize, bounded foreground-area change for fixture shapes, resize-policy mismatch, prediction resize-back mismatch, threshold/resize ordering sensitivity, and preprocessing mismatch failure behavior.
- [x] 3.6 Record the runtime decision outcome for mitochondria `image_size/crop_size=256` and glomeruli `crop_size=512 -> output_size=256`: supported, rejected, or `resize_benefit_unproven`, with the evidence source named.

## 4. Candidate Comparison Hardening

- [x] 4.1 Update `src/eq/training/compare_glomeruli_candidates.py` so fresh candidate comparison persists a shared split before candidate training/promotion evaluation and builds deterministic manifests from that recorded held-out image set only, rather than relying only on post-hoc intersection of candidate metadata.
- [x] 4.2 Update existing-artifact comparison so `--transfer-model-path` and `--scratch-model-path` require auditable split provenance for promotion-facing decisions.
- [x] 4.3 Add transfer-base provenance and mitochondria-training-scope sections to `promotion_report.json`, `promotion_report.md`, and `promotion_report.html`, including base artifact path, `mitochondria_training_scope`, `mitochondria_inference_claim_status`, physical training/testing counts, actual fitted image/mask counts, and base resize/preprocessing policy.
- [x] 4.4 Add split-integrity sections to `promotion_report.json`, `promotion_report.md`, and `promotion_report.html`.
- [x] 4.5 Write `metric_by_category.csv` and include metrics by candidate family, category, cohort ID when available, and lane assignment when available; prediction rows must actually carry cohort/lane fields from the manifest when those fields exist.
- [x] 4.6 Write `prediction_shape_audit.csv` and gate excessive background foreground prediction, positive/boundary overcoverage, and category-specific failures even when aggregate Dice/Jaccard are high.
- [x] 4.7 Write `resize_policy_audit.csv` or equivalent structured report fields comparing held-out metrics, prediction-shape summaries, source-resolution distributions, resize ratios, and threshold/resize ordering across the current `512 -> 256` policy and a no-downsample or less-downsample sensitivity when feasible; if infeasible, record the concrete infeasibility evidence and keep resize benefit `resize_benefit_unproven`.
- [x] 4.8 Write `failure_reproduction_audit.csv` or equivalent structured fields for the current poor-performance panels, including one row per displayed example and a candidate-level root-cause classification; generic candidate-comparison prediction rows do not satisfy this until they trace the exact mitochondria, transfer, and scratch/no-base panels under review.
- [x] 4.9 Add a root-cause classifier that maps poor performance to `image_mask_pairing_error`, `transform_alignment_error`, `mask_binarization_error`, `class_channel_or_threshold_error`, `resize_policy_artifact`, `split_or_panel_bias`, `training_signal_insufficient`, `mitochondria_base_defect`, `negative_background_supervision_missing`, or `true_model_underfit` before any retraining is accepted as remediation.
- [x] 4.10 Ensure candidate-comparison report fields are covered by `tests/test_segmentation_validation_audit.py`, and mark promotion-facing decisions as `audit_missing` or `not_promotion_eligible` when required split/category/prediction-shape/resize-policy/transfer-base/root-cause fields are missing or fail promotion criteria.

## 5. Documentation Claim Gate

- [x] 5.1 Implement `documentation_claim_audit.md` generation for README and onboarding segmentation-performance claims in the relevant validation or candidate-comparison artifact surface, not only as an in-memory pytest helper.
- [x] 5.2 Update `README.md` and `docs/ONBOARDING_GUIDE.md` so current-performance tables cite only audit-passing held-out evidence or are relabeled/removed when only research-use-only, not-promotion-eligible, or internal biased evidence exists.
- [x] 5.3 Update `docs/TECHNICAL_LAB_NOTEBOOK.md` to distinguish runtime compatibility, internal held-out validation, biased/internal audit evidence, and scientific promotion status.
- [x] 5.4 Add tests or a lightweight checker that fails when README/onboarding current-performance tables cite a partly in-sample, research-use-only, not-promotion-eligible, or audit-missing report.

## 6. Validation

- [x] 6.1 Run `python3 -m py_compile src/eq/training/segmentation_validation_audit.py`.
- [x] 6.2 Run focused tests: `python -m pytest -q tests/test_glomeruli_candidate_comparison.py tests/test_segmentation_training_contract.py tests/test_training_entrypoint_contract.py tests/test_training_smoke_v2.py`.
- [x] 6.3 Run new audit-specific tests: `python -m pytest -q tests/test_segmentation_validation_audit.py`.
- [x] 6.4 Run `python scripts/check_openspec_explicitness.py openspec/changes/p0-harden-glomeruli-segmentation-validation`.
- [x] 6.5 Run `openspec validate p0-harden-glomeruli-segmentation-validation --strict`.
- [x] 6.6 Run the optional runtime integration test against current regenerated artifacts when validating promotion or README-facing claims: `python -m pytest -q tests/integration/test_glomeruli_segmentation_validation_audit_runtime.py`. The old April 24 report failed this check because candidate split provenance was missing from `promotion_report.json`.
- [x] 6.7 When the runtime integration test needs real Mac model loading or Metal access, run it unsandboxed with the `eq-mac` interpreter: `env PYTORCH_ENABLE_MPS_FALLBACK=1 MPLCONFIGDIR=/tmp/mpl_eq /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest -q tests/integration/test_glomeruli_segmentation_validation_audit_runtime.py`. Do not mark complete until it passes against the regenerated report or fails with an explicitly reported blocker.

## 7. Audit Record

- [x] 7.1 Add `audit-results.md` as the durable OpenSpec decision log for audit findings, validation evidence, production decisions, and remaining blockers.
- [x] 7.2 Update `audit-results.md` after exact panel reproduction, resize sensitivity, regenerated runtime integration, and unsandboxed Mac runtime checks complete.
