# segmentation-validation-audit Specification

## Purpose
TBD - created by archiving change p0-harden-glomeruli-segmentation-validation. Update Purpose after archive.
## Requirements
### Requirement: Glomeruli segmentation validation audit is pytest-backed
The repository SHALL provide pytest-backed glomeruli segmentation validation checks that inspect the data, DataBlock, augmentation, preprocessing, training provenance, promotion evidence, prediction behavior, and documentation claims before model performance can be treated as promotion-facing.

#### Scenario: Validation audit runs in pytest
- **WHEN** `python -m pytest -q tests/test_segmentation_validation_audit.py` is executed
- **THEN** pytest SHALL exercise reusable audit helpers in `src/eq/training/segmentation_validation_audit.py`
- **AND** the tests SHALL cover split integrity, DataBlock sampling, transform/preprocessing parity, prediction-shape gates, metric-by-category behavior, and documentation-claim gates without requiring a new user-facing CLI or YAML workflow

#### Scenario: Runtime artifact audit is needed
- **WHEN** current ProjectsRuntime candidate artifacts, cohort manifests, or README-facing performance claims must be checked against real local data
- **THEN** the supported path SHALL be `python -m pytest -q tests/integration/test_glomeruli_segmentation_validation_audit_runtime.py`
- **AND** the integration test SHALL be explicitly run for those decision points rather than exposed as a normal user workflow

#### Scenario: Validation audit rerun is triggered by a decision point
- **WHEN** new scratch or transfer candidate artifacts are trained, the admitted cohort manifest changes, DataBlock/augmentation/preprocessing code changes, candidate-comparison gates change, downstream quantification begins depending on a segmentation artifact, or README/onboarding current-performance claims are added or refreshed
- **THEN** the focused pytest tests and, when real artifacts matter, the runtime integration pytest SHALL be the supported validation gate
- **AND** routine exploratory training runs SHALL NOT be required to run the audit unless they are being used for promotion, downstream dependency, or documentation claims

### Requirement: Validation audit reconstructs the actual data path
The validation audit SHALL reconstruct the concrete image/mask rows, cohort lanes, and filesystem roots used by glomeruli training and promotion evaluation.

#### Scenario: Admitted cohort registry is audited
- **WHEN** the audit target data root is `$EQ_RUNTIME_ROOT/raw_data/cohorts`
- **THEN** the audit SHALL read the admitted manifest rows used by `get_items_full_images`
- **AND** it SHALL report counts by `cohort_id`, `lane_assignment`, image path, mask path, subject ID, and mask availability
- **AND** it SHALL fail closed if an admitted row lacks a resolvable paired image or mask

#### Scenario: Direct paired root is audited
- **WHEN** the audit target data root is an active paired project root under `raw_data`
- **THEN** the audit SHALL enumerate the `images/` and `masks/` contract directly
- **AND** it SHALL fail closed if unpaired images or masks are present

### Requirement: Validation audit reproduces current poor-performance panels
The validation audit SHALL reproduce the current poor-performance evidence before classifying model quality or recommending retraining.

#### Scenario: Current visual panels are audited
- **WHEN** mitochondria, glomeruli transfer, or glomeruli no-base/scratch validation panels are used as evidence of poor performance
- **THEN** the audit SHALL reconstruct the source artifact path, source image path, source mask path, crop box, resize policy, threshold, prediction tensor shape, prediction resize-back method, and overlay-generation path for each displayed example
- **AND** it SHALL write `failure_reproduction_audit.csv` or an equivalent structured payload with one row per displayed example

#### Scenario: Panel cannot be reproduced
- **WHEN** a displayed poor-performance panel cannot be traced back to its model artifact, image, mask, crop, and preprocessing path
- **THEN** the audit SHALL classify the panel evidence as `audit_missing`
- **AND** it SHALL NOT allow the panel to justify retraining or documentation claims until the traceability gap is resolved

#### Scenario: Reproduction reveals an implementation defect
- **WHEN** reproduction shows image/mask pairing, transform alignment, mask binarization, class-channel, threshold, resize, or overlay-generation defects
- **THEN** the audit SHALL classify the root cause accordingly
- **AND** candidate retraining SHALL NOT be accepted as remediation until the implementation defect is fixed and the panel is regenerated

### Requirement: Validation audit inspects mitochondria transfer-base provenance
The validation audit SHALL inspect mitochondria base artifacts when they initialize a glomeruli transfer candidate.

#### Scenario: Transfer candidate references a mitochondria base
- **WHEN** a glomeruli transfer candidate is audited
- **THEN** the audit SHALL read the referenced mitochondria base sidecar
- **AND** it SHALL report the base artifact path, `mitochondria_training_scope`, `mitochondria_inference_claim_status`, physical training/testing counts, actual fitted image/mask counts, resize/preprocessing policy, and training command

#### Scenario: Mitochondria testing data were included in base training
- **WHEN** the mitochondria base reports `mitochondria_training_scope=all_available_pretraining`
- **THEN** the audit SHALL allow the base for representation transfer
- **AND** it SHALL block mitochondria held-out performance claims from that artifact
- **AND** it SHALL keep glomeruli promotion eligibility dependent on held-out glomeruli validation rather than mitochondria validation

#### Scenario: Mitochondria held-out claims are made
- **WHEN** a report, README section, or model sidecar claims mitochondria held-out performance
- **THEN** the audit SHALL require `mitochondria_training_scope=heldout_test_preserved`
- **AND** it SHALL require that the physical `raw_data/mitochondria_data/testing` root was excluded from fitting
- **AND** it SHALL mark the claim `not_promotion_eligible` if testing examples were included in training

### Requirement: Validation audit checks split integrity
The validation audit SHALL determine whether deterministic promotion evidence is statistically separated from training evidence.

#### Scenario: Promotion manifest overlaps training images
- **WHEN** a deterministic promotion manifest contains any image path recorded in a candidate artifact's training split
- **THEN** the audit SHALL mark the promotion evidence as `not_promotion_eligible`
- **AND** it SHALL record the overlapping paths in `split_overlap_audit.csv`
- **AND** it SHALL preserve `runtime_use_status=available_research_use` for artifacts that load and run

#### Scenario: Promotion manifest overlaps training subjects
- **WHEN** subject IDs are available and a deterministic promotion manifest contains any subject ID recorded in a candidate artifact's training split
- **THEN** the audit SHALL mark the promotion evidence as `not_promotion_eligible`
- **AND** it SHALL report whether the blocker is image-level overlap, subject-level overlap, or both
- **AND** it SHALL NOT imply that the artifact is unusable for exploratory research or runtime smoke testing

#### Scenario: Split provenance is missing
- **WHEN** a candidate artifact lacks machine-readable train/validation split provenance
- **THEN** the audit SHALL classify that artifact as `promotion_evidence_status=audit_missing`
- **AND** it SHALL NOT allow promotion-facing performance claims from that artifact
- **AND** it SHALL keep the artifact available for research/runtime use if model loading and inference work

### Requirement: Validation audit inspects DataBlock sampling and crop distribution
The validation audit SHALL measure the actual crop and mask distribution produced by the supported dynamic-patching DataBlock configuration.

#### Scenario: DataBlock sampling audit is generated
- **WHEN** the audit samples batches from `build_segmentation_dls_dynamic_patching`
- **THEN** `datablock_sampling_audit.csv` SHALL report split, batch index, example index, crop size, output size, mask foreground fraction, any-positive status, and whether the crop exceeds `min_pos_pixels`
- **AND** pytest-generated audit payloads SHALL summarize foreground-fraction distributions separately for train and validation crops when a report payload is needed for assertions

#### Scenario: Positive-aware sampling creates biased validation evidence
- **WHEN** validation crops are foreground-heavy enough that all-foreground prediction becomes a competitive baseline
- **THEN** the audit SHALL record `foreground_heavy_validation_panel`
- **AND** promotion-facing claims SHALL be marked not eligible until a held-out panel with adequate background and boundary evidence is available

### Requirement: Validation audit checks augmentation and preprocessing parity
The validation audit SHALL verify that training and deterministic evaluation use compatible geometric, resizing, mask-binarization, and learner preprocessing semantics.

#### Scenario: Image and mask transforms are checked for alignment
- **WHEN** the audit exercises the DataBlock item transforms
- **THEN** it SHALL verify that image and mask crops use the same crop coordinates and spatial transforms
- **AND** it SHALL fail closed if the transformed mask no longer aligns with the transformed image geometry

#### Scenario: Evaluation preprocessing is checked
- **WHEN** a candidate artifact is evaluated on deterministic crops
- **THEN** the audit SHALL record whether evaluation uses learner-consistent preprocessing and threshold semantics
- **AND** it SHALL mark promotion-facing claims not eligible if evaluation uses a bespoke transform path that cannot be justified against the learner's training preprocessing

### Requirement: Validation audit treats resize policy as a promotion sensitivity
The validation audit SHALL evaluate crop-to-network resizing as a methodological sensitivity rather than assuming the configured resize policy improves model performance.

#### Scenario: Resize policy is recorded
- **WHEN** glomeruli training or deterministic candidate evaluation is audited
- **THEN** the audit SHALL record source image size, source mask size, `crop_size`, `output_size`, crop-to-output resize ratio, aspect-ratio policy, resize method, image interpolation, mask interpolation, mask binarization after resize, prediction resize-back method, and whether predictions were thresholded before or after resize-back
- **AND** those fields SHALL be available in the structured audit payload used by pytest and candidate-comparison reports
- **AND** those fields SHALL be summarized by split, cohort ID, lane assignment when available, and deterministic review category

#### Scenario: Resize policy differs between training and evaluation
- **WHEN** deterministic evaluation uses crop, resize, interpolation, mask-binarization, threshold, or prediction resize-back semantics that differ from candidate training provenance
- **THEN** the audit SHALL mark promotion-facing evidence as `not_promotion_eligible`
- **AND** the report SHALL identify the mismatch as a resize-policy failure rather than a generic preprocessing warning

#### Scenario: Source-resolution distributions differ across evidence splits
- **WHEN** train, validation, or deterministic promotion examples have materially different source image dimensions, source mask dimensions, crop-to-output resize ratios, or physical-resolution metadata when available
- **THEN** the audit SHALL mark promotion-facing evidence as `insufficient_evidence_for_promotion`
- **AND** the report SHALL identify the affected split, cohort ID, lane assignment, or review category rather than pooling the metric silently

#### Scenario: Current downsampling policy is promotion-facing
- **WHEN** a glomeruli candidate trained with the current `crop_size=512` and `output_size=256` policy is proposed for promotion or README-facing performance claims
- **THEN** the audit SHALL compare held-out metrics, prediction-shape summaries, resize-ratio distributions, and threshold/resize ordering against a no-downsample or less-downsample sensitivity when local memory and supported hardware permit
- **AND** if the sensitivity cannot be run, the report SHALL record `resize_benefit_unproven`
- **AND** the claim SHALL NOT state or imply that `512 -> 256` resizing improves performance unless the held-out sensitivity supports that conclusion

#### Scenario: Resize corrupts binary mask semantics
- **WHEN** image/mask transform fixtures show that mask resizing produces non-binary labels, changes foreground area beyond declared tolerance for simple shapes, or misaligns mask geometry from image geometry
- **THEN** the audit SHALL fail the transform/preprocessing contract
- **AND** candidate promotion SHALL be blocked until the resize and mask-binarization path is corrected or explicitly revalidated

#### Scenario: Resize changes foreground burden
- **WHEN** downsampling or resize-back materially changes truth foreground fraction, prediction foreground fraction, boundary thickness, or background-crop false-positive burden
- **THEN** `resize_policy_audit.csv` or equivalent structured fields SHALL report the direction and size of the change by candidate family and review category
- **AND** candidate promotion SHALL fail when the favorable aggregate metric depends on a resize-induced overcoverage or background false-positive artifact

### Requirement: Validation audit uses Research Partner review lanes
The validation audit SHALL organize conclusions into implementation, statistical, scientific, robustness, and documentation-consistency lanes.

#### Scenario: Validation audit report is rendered
- **WHEN** a pytest-generated audit report is rendered for debugging or integration-test evidence
- **THEN** it SHALL include sections for implementation audit, statistical validity, scientific interpretation, robustness tests, and documentation consistency
- **AND** each section SHALL separate direct evidence from inference

#### Scenario: Claim type is reported
- **WHEN** a report or README claim is audited
- **THEN** the audit SHALL classify the claim as descriptive, predictive/prognostic, associational, causal, or external-validity related
- **AND** it SHALL block unsupported causal or external-validity language for internal segmentation panels

### Requirement: Validation audit classifies poor-performance root causes before remediation
The validation audit SHALL classify why candidate performance is poor before the implementation treats retraining or new supervision as the solution.

#### Scenario: Poor performance is detected
- **WHEN** aggregate metrics, category metrics, prediction-shape gates, or visual panels show poor mitochondria or glomeruli segmentation performance
- **THEN** the audit SHALL assign one or more root-cause classes from `image_mask_pairing_error`, `transform_alignment_error`, `mask_binarization_error`, `class_channel_or_threshold_error`, `resize_policy_artifact`, `split_or_panel_bias`, `training_signal_insufficient`, `mitochondria_base_defect`, `negative_background_supervision_missing`, or `true_model_underfit`
- **AND** it SHALL record the evidence supporting each class

#### Scenario: Root cause implies a code or evaluation defect
- **WHEN** the root cause is pairing, transform, binarization, class-channel, threshold, resize, split, or panel-generation related
- **THEN** the remediation SHALL be code/evaluation correction followed by reevaluation
- **AND** fresh training SHALL NOT be considered a valid fix until the defect is corrected

#### Scenario: Root cause implies insufficient supervision
- **WHEN** the root cause is `negative_background_supervision_missing` or `training_signal_insufficient`
- **THEN** the report SHALL identify whether `p1-add-negative-glomeruli-crop-supervision` is the required remediation path
- **AND** it SHALL keep current artifacts available only as research-use candidates until the supervision gap is addressed and reevaluated

### Requirement: Documentation claims are gated by validation audit evidence
README and onboarding performance summaries SHALL cite only held-out, audit-passing validation evidence for current segmentation performance tables.

#### Scenario: README cites a not-promotion-eligible report
- **WHEN** `documentation_claim_audit.md` finds a README or onboarding table citing a compatibility-only, research-use-only, partly in-sample, not-promotion-eligible, or audit-failed promotion report
- **THEN** the audit SHALL mark the documentation claim as `not_promotion_eligible`
- **AND** the implementation SHALL update the documentation to remove or relabel the performance table before the change is complete

#### Scenario: README cites audit-passing evidence
- **WHEN** README or onboarding performance text cites a report that passes the hardened validation audit
- **THEN** the documentation SHALL state that the evidence is internal held-out validation
- **AND** it SHALL NOT describe the result as external validation, clinical readiness, or unconditional scientific promotion

