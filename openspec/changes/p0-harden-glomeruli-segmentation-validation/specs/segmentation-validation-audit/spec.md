## ADDED Requirements

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
