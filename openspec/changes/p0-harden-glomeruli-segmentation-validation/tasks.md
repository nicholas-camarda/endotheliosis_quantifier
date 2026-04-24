## 1. Pytest Audit Surface

- [ ] 1.1 Add `src/eq/training/segmentation_validation_audit.py` with pure helper functions for data-contract audit, split-overlap audit, DataBlock crop sampling audit, transform/preprocessing audit, prediction-shape audit, metric-by-category aggregation, and documentation-claim audit.
- [ ] 1.2 Add `tests/test_segmentation_validation_audit.py` to exercise the helper functions with deterministic fixtures and synthetic artifacts.
- [ ] 1.3 Add `tests/integration/test_glomeruli_segmentation_validation_audit_runtime.py` for explicit local-runtime checks against current ProjectsRuntime artifacts, guarded so it can be skipped when runtime artifacts are unavailable.
- [ ] 1.4 Add current-failure reproduction helpers that reconstruct the exact mitochondria, glomeruli transfer, and glomeruli no-base/scratch panel examples from artifact paths, image paths, mask paths, crop boxes, resize policy, threshold, prediction tensor shape, and overlay-generation path.
- [ ] 1.5 Confirm no new `eq run-config` workflow, committed audit YAML, or user-facing audit CLI is introduced by this change.

## 2. Training Provenance And Split Integrity

- [ ] 2.1 Extend mitochondria base metadata in `src/eq/training/train_mitochondria.py` to record `mitochondria_training_scope`, `mitochondria_inference_claim_status`, physical `training/` item count, physical `testing/` item count, actual pretraining image/mask paths, split policy, resize/preprocessing settings, training command, and code/package provenance.
- [ ] 2.2 Add tests proving that a mitochondria base trained with physical `testing/` included is allowed only as `mitochondria_training_scope=all_available_pretraining` and cannot support mitochondria held-out performance claims.
- [ ] 2.3 Extend glomeruli training sidecar metadata in `src/eq/training/train_glomeruli.py` and `src/eq/training/transfer_learning.py` to record `split_seed`, `splitter_name`, `train_images`, `valid_images`, crop settings, positive-aware sampling settings, augmentation settings, mask-binarization semantics, learner preprocessing, candidate family, transfer-base artifact path, transfer-base training scope, and code/package provenance.
- [ ] 2.4 Update split-manifest writing so `*_splits.json` records machine-readable `train_images` and `valid_images` for every glomeruli candidate artifact.
- [ ] 2.5 Add a two-axis status classifier that preserves loadable artifacts as `runtime_use_status=available_research_use` while marking missing split or transfer-base provenance as `promotion_evidence_status=audit_missing`.
- [ ] 2.6 Add tests proving split-overlap detection blocks promotion when a deterministic manifest includes training images or training subjects.

## 3. DataBlock, Augmentation, And Preprocessing Audit

- [ ] 3.1 Implement a DataBlock sampling audit that builds DataLoaders through `build_segmentation_dls_dynamic_patching` and writes `datablock_sampling_audit.csv` with train/validation crop foreground fractions and positive-crop support.
- [ ] 3.2 Add transform-alignment checks that verify image and mask crops use identical spatial crop/resize semantics on fixture data.
- [ ] 3.3 Add preprocessing-parity checks that record whether deterministic evaluation uses learner-consistent preprocessing and supported threshold semantics.
- [ ] 3.4 Add resize-policy audit helpers that record source image/mask dimensions, `crop_size`, `output_size`, crop-to-output resize ratio, aspect-ratio policy, `resize_method`, image/mask interpolation, mask binarization after resize, prediction resize-back behavior, and threshold/resize ordering for training and deterministic evaluation.
- [ ] 3.5 Add tests for crop-size propagation, positive-aware sampling metadata, image/mask transform alignment, binary mask preservation after resize, bounded foreground-area change for fixture shapes, resize-policy mismatch, prediction resize-back mismatch, threshold/resize ordering sensitivity, and preprocessing mismatch failure behavior.

## 4. Candidate Comparison Hardening

- [ ] 4.1 Update `src/eq/training/compare_glomeruli_candidates.py` so fresh candidate comparison persists a shared split before promotion evaluation and builds deterministic manifests from held-out images only.
- [ ] 4.2 Update existing-artifact comparison so `--transfer-model-path` and `--scratch-model-path` require auditable split provenance for promotion-facing decisions.
- [ ] 4.3 Add transfer-base provenance and mitochondria-training-scope sections to `promotion_report.json`, `promotion_report.md`, and `promotion_report.html`.
- [ ] 4.4 Add split-integrity sections to `promotion_report.json`, `promotion_report.md`, and `promotion_report.html`.
- [ ] 4.5 Write `metric_by_category.csv` and include metrics by candidate family, category, cohort ID when available, and lane assignment when available.
- [ ] 4.6 Write `prediction_shape_audit.csv` and gate excessive background foreground prediction, positive/boundary overcoverage, and category-specific failures even when aggregate Dice/Jaccard are high.
- [ ] 4.7 Write `resize_policy_audit.csv` or equivalent structured report fields comparing held-out metrics, prediction-shape summaries, source-resolution distributions, resize ratios, and threshold/resize ordering across the current `512 -> 256` policy and a no-downsample or less-downsample sensitivity when feasible.
- [ ] 4.8 Write `failure_reproduction_audit.csv` or equivalent structured fields for the current poor-performance panels, including one row per displayed example and a candidate-level root-cause classification.
- [ ] 4.9 Add a root-cause classifier that maps poor performance to `image_mask_pairing_error`, `transform_alignment_error`, `mask_binarization_error`, `class_channel_or_threshold_error`, `resize_policy_artifact`, `split_or_panel_bias`, `training_signal_insufficient`, `mitochondria_base_defect`, `negative_background_supervision_missing`, or `true_model_underfit` before any retraining is accepted as remediation.
- [ ] 4.10 Ensure candidate-comparison report fields are covered by `tests/test_segmentation_validation_audit.py`, and mark promotion-facing decisions as `audit_missing` or `not_promotion_eligible` when required split/category/prediction-shape/resize-policy/transfer-base/root-cause fields are missing or fail promotion criteria.

## 5. Documentation Claim Gate

- [ ] 5.1 Implement `documentation_claim_audit.md` generation for README and onboarding segmentation-performance claims.
- [ ] 5.2 Update `README.md` and `docs/ONBOARDING_GUIDE.md` so current-performance tables cite only audit-passing held-out evidence or are relabeled/removed when only research-use-only, not-promotion-eligible, or internal biased evidence exists.
- [ ] 5.3 Update `docs/TECHNICAL_LAB_NOTEBOOK.md` to distinguish runtime compatibility, internal held-out validation, biased/internal audit evidence, and scientific promotion status.
- [ ] 5.4 Add tests or a lightweight checker that fails when README/onboarding current-performance tables cite a partly in-sample, research-use-only, not-promotion-eligible, or audit-missing report.

## 6. Validation

- [ ] 6.1 Run `python3 -m py_compile src/eq/training/segmentation_validation_audit.py`.
- [ ] 6.2 Run focused tests: `python -m pytest -q tests/test_glomeruli_candidate_comparison.py tests/test_segmentation_training_contract.py tests/test_training_entrypoint_contract.py tests/test_training_smoke_v2.py`.
- [ ] 6.3 Run new audit-specific tests: `python -m pytest -q tests/test_segmentation_validation_audit.py`.
- [ ] 6.4 Run `python scripts/check_openspec_explicitness.py openspec/changes/p0-harden-glomeruli-segmentation-validation`.
- [ ] 6.5 Run `openspec validate p0-harden-glomeruli-segmentation-validation --strict`.
- [ ] 6.6 Run the optional runtime integration test against current artifacts when validating promotion or README-facing claims: `python -m pytest -q tests/integration/test_glomeruli_segmentation_validation_audit_runtime.py`.
- [ ] 6.7 When the runtime integration test needs real Mac model loading or Metal access, run it unsandboxed with the `eq-mac` interpreter: `env PYTORCH_ENABLE_MPS_FALLBACK=1 MPLCONFIGDIR=/tmp/mpl_eq /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest -q tests/integration/test_glomeruli_segmentation_validation_audit_runtime.py`.
