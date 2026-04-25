## ADDED Requirements

### Requirement: Mitochondria transfer base records training-scope provenance
Mitochondria artifacts used as glomeruli transfer bases SHALL record whether they preserved the physical mitochondria testing split or used all available mitochondria pairs for representation pretraining.

#### Scenario: Mitochondria base artifact is exported
- **WHEN** `src/eq/training/train_mitochondria.py` exports a mitochondria base artifact
- **THEN** its sidecar metadata SHALL record `mitochondria_training_scope`, `mitochondria_inference_claim_status`, physical `training/` image count, physical `testing/` image count, actual pretraining image paths, actual pretraining mask paths, split policy, resize/preprocessing policy, training command, code version, and package versions
- **AND** the artifact SHALL record whether the physical `raw_data/mitochondria_data/testing` root was included in model fitting

#### Scenario: All mitochondria data are used for representation pretraining
- **WHEN** a mitochondria base uses both physical `training/` and `testing/` roots for model fitting
- **THEN** it SHALL set `mitochondria_training_scope=all_available_pretraining`
- **AND** it SHALL set `mitochondria_inference_claim_status=not_applicable_for_inference_claim`
- **AND** it MAY be used as a glomeruli transfer base only when glomeruli promotion evidence remains held-out and audit-passing

#### Scenario: Mitochondria testing split is preserved
- **WHEN** a workflow reports mitochondria held-out segmentation performance or uses mitochondria held-out metrics for model selection
- **THEN** the physical `raw_data/mitochondria_data/testing` root SHALL remain excluded from mitochondria training
- **AND** the artifact SHALL set `mitochondria_training_scope=heldout_test_preserved`
- **AND** it SHALL set `mitochondria_inference_claim_status=heldout_evaluable`

#### Scenario: Mitochondria scope is missing for transfer
- **WHEN** a glomeruli transfer candidate references a mitochondria base artifact with missing or inconsistent mitochondria training-scope provenance
- **THEN** the transfer candidate MAY remain `runtime_use_status=available_research_use` if it loads and runs
- **AND** its promotion evidence SHALL be `audit_missing` until transfer-base provenance is resolved

### Requirement: Glomeruli artifacts record split and sampler provenance
Supported glomeruli segmentation artifacts SHALL record enough split, sampler, crop, augmentation, and preprocessing provenance to audit whether model training and promotion evaluation were statistically separable.

#### Scenario: Glomeruli model artifact is exported
- **WHEN** `src/eq/training/train_glomeruli.py` or `src/eq/training/transfer_learning.py` exports a glomeruli candidate artifact
- **THEN** its sidecar metadata SHALL record `data_root`, `training_mode`, `candidate_family`, `seed`, `split_seed`, `splitter_name`, `train_images`, `valid_images`, `source_image_size_summary`, `source_mask_size_summary`, `crop_size`, `image_size`, `output_size`, `crop_to_output_resize_ratio`, `aspect_ratio_policy`, `resize_method`, image interpolation, mask interpolation, mask-binarization-after-resize semantics, prediction resize-back assumptions, threshold/resize ordering assumptions, `positive_focus_p`, `min_pos_pixels`, `pos_crop_attempts`, augmentation settings, learner preprocessing, transfer-base artifact path when applicable, transfer-base `mitochondria_training_scope` when applicable, training command, code version, and package versions
- **AND** the split sidecar SHALL be machine-readable enough for `segmentation_validation_audit.py` to compare candidate splits against deterministic promotion manifests

#### Scenario: Split sidecar cannot be audited
- **WHEN** an exported glomeruli artifact lacks train/validation image identifiers or the identifiers cannot be resolved
- **THEN** the artifact SHALL be classified as `runtime_use_status=available_research_use` if it loads and runs in the supported environment
- **AND** it SHALL be classified as `promotion_evidence_status=audit_missing`
- **AND** it SHALL NOT be treated as scientifically promoted or used for README-facing current-performance claims

### Requirement: DataBlock sampling audit is available for supported training roots
Supported segmentation training roots SHALL be auditable through the same DataBlock construction path used for training.

#### Scenario: DataBlock audit samples a supported glomeruli root
- **WHEN** `segmentation_validation_audit.py` audits `$EQ_RUNTIME_ROOT/raw_data/cohorts` or `$EQ_RUNTIME_ROOT/raw_data/cohorts/<cohort_id>`
- **THEN** it SHALL build DataLoaders through `build_segmentation_dls_dynamic_patching`
- **AND** it SHALL report crop foreground distributions for train and validation batches without writing generated training data into the repository

#### Scenario: Static patch root is encountered during audit
- **WHEN** the audit target is a retired or active static patch root such as `image_patches/` or `mask_patches/`
- **THEN** the audit SHALL fail closed with the same unsupported-root policy as training
- **AND** it SHALL NOT convert that root into an active training or validation input

### Requirement: Resize policy is auditable for supported training roots
Supported segmentation training SHALL expose the crop-to-network resize policy clearly enough to test whether the policy is technically aligned and promotion-supporting.

#### Scenario: Dynamic-patching DataLoaders resize crops
- **WHEN** `build_segmentation_dls_dynamic_patching` builds glomeruli DataLoaders with a `crop_size` different from `output_size`
- **THEN** the training provenance SHALL identify source image/mask size summaries, selected crop size, final network input size, crop-to-output resize ratio, aspect-ratio policy, resize method, image interpolation, mask interpolation, mask binarization semantics, and threshold/resize ordering assumptions
- **AND** the audit SHALL be able to distinguish the current `512 -> 256` downsampling policy from no-downsample or less-downsample sensitivity runs

#### Scenario: Source-resolution distributions are promotion-relevant
- **WHEN** glomeruli artifacts are compared for promotion
- **THEN** their provenance SHALL be sufficient to compare train, validation, and deterministic promotion source-resolution distributions
- **AND** the candidate SHALL NOT be `promotion_eligible` when resolution distribution imbalance is unresolved

#### Scenario: Resize policy benefit is not established
- **WHEN** a training artifact uses downsampling but lacks held-out resize-sensitivity evidence
- **THEN** the artifact MAY remain `runtime_use_status=available_research_use`
- **AND** it SHALL NOT be classified as `promotion_eligible` on resize-dependent performance claims

### Requirement: Dynamic validation split is not promotion evidence by itself
Training-time validation metrics from stochastic dynamic crops SHALL NOT be sufficient evidence for scientific promotion.

#### Scenario: Training completes with validation metrics
- **WHEN** glomeruli training completes and records training-time validation Dice or Jaccard
- **THEN** those metrics SHALL be treated as optimization diagnostics
- **AND** scientific promotion SHALL still require the held-out deterministic promotion manifest and validation audit gates

#### Scenario: Training and promotion use the same data root
- **WHEN** candidate training and candidate promotion both reference the admitted cohort registry root
- **THEN** the promotion workflow SHALL use recorded split provenance to select held-out evaluation images only
- **AND** it SHALL mark promotion evidence as `not_promotion_eligible` if held-out selection cannot be verified

### Requirement: Training provenance distinguishes runtime support from scientific promotion
The artifact provenance contract SHALL continue to separate current-namespace runtime support from scientific model promotion.

#### Scenario: Artifact loads successfully
- **WHEN** a glomeruli artifact loads in the certified environment and can run inference
- **THEN** the sidecar SHALL allow `artifact_status=supported_runtime`
- **AND** `runtime_use_status` SHALL allow `available_research_use`
- **AND** `promotion_evidence_status` SHALL remain `audit_missing`, `insufficient_evidence_for_promotion`, `not_promotion_eligible`, or `promotion_eligible` according to the hardened validation audit rather than loadability alone

#### Scenario: Artifact passes hardened audit
- **WHEN** a glomeruli artifact clears split integrity, DataBlock audit, deterministic held-out metrics, prediction-shape gates, and documentation-claim gates
- **THEN** it MAY be marked as scientifically promoted by the promotion report
- **AND** the sidecar SHALL reference the exact promotion report and validation audit payload that justified the status
