## ADDED Requirements

### Requirement: Glomeruli artifacts record split and sampler provenance
Supported glomeruli segmentation artifacts SHALL record enough split, sampler, crop, augmentation, and preprocessing provenance to audit whether model training and promotion evaluation were statistically separable.

#### Scenario: Glomeruli model artifact is exported
- **WHEN** `src/eq/training/train_glomeruli.py` or `src/eq/training/transfer_learning.py` exports a glomeruli candidate artifact
- **THEN** its sidecar metadata SHALL record `data_root`, `training_mode`, `candidate_family`, `seed`, `split_seed`, `splitter_name`, `train_images`, `valid_images`, `crop_size`, `image_size`, `positive_focus_p`, `min_pos_pixels`, `pos_crop_attempts`, augmentation settings, mask-binarization semantics, learner preprocessing, training command, code version, and package versions
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
