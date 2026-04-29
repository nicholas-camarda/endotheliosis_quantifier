# quantification-burden-artifact-layout Specification

## Purpose
Define the canonical burden-model artifact layout so primary burden-index outputs and contained estimator subtrees remain navigable and unambiguous.
## Requirements
### Requirement: Burden-model artifacts use contained subtrees
Quantification burden-model artifacts SHALL be organized by model or estimator family rather than mixing primary-model buckets and estimator subtrees at the same semantic level.

#### Scenario: Primary burden-index artifacts are generated
- **WHEN** the primary burden-index evaluator writes artifacts
- **THEN** they SHALL be written under `burden_model/primary_burden_index/`
- **AND** the serialized model SHALL be under `burden_model/primary_burden_index/model/`
- **AND** aggregate cohort or interval tables SHALL be under `burden_model/primary_burden_index/summaries/`
- **AND** the evaluator SHALL NOT write current primary burden-index artifacts to the retired top-level primary-model or aggregate-summary folders

#### Scenario: Burden-model root is opened
- **WHEN** an operator opens `burden_model/`
- **THEN** `burden_model/INDEX.md` SHALL identify the first-read files for each contained model or estimator subtree
- **AND** it SHALL distinguish primary burden-index outputs, learned ROI outputs, source-aware estimator outputs, and severe-aware ordinal estimator outputs

#### Scenario: Estimator subtree summary naming is used
- **WHEN** an estimator subtree writes a `summary/` directory
- **THEN** `summary/` SHALL mean first-read verdict, metrics, manifest, or figure artifacts for that subtree
- **AND** plural `summaries/` SHALL be reserved for aggregate table outputs such as cohort summaries or interval summaries

#### Scenario: Learned ROI artifacts are generated
- **WHEN** the learned ROI evaluator writes artifacts
- **THEN** it SHALL write `burden_model/learned_roi/INDEX.md`
- **AND** it SHALL write first-read summary artifacts under `burden_model/learned_roi/summary/`
- **AND** typed validation, calibration, diagnostics, evidence, candidates, and feature-set outputs SHALL remain in their typed folders

### Requirement: Quantification artifacts expose preprocessing, threshold, and ROI statuses
Quantification output artifacts SHALL preserve enough provenance to audit preprocessing, thresholding, and ROI inclusion decisions.

#### Scenario: Quantification review artifacts are written
- **WHEN** `eq run-config --config configs/endotheliosis_quantification.yaml` completes
- **THEN** quantification review artifacts SHALL include the segmentation inference preprocessing contract
- **AND** they SHALL include any segmentation threshold value used to generate masks or probabilities consumed by quantification
- **AND** they SHALL include counts by ROI status, including rows excluded because of size mismatch or below-threshold components

#### Scenario: ROI crop is not written
- **WHEN** a row has no valid ROI crop because of a fail-closed ROI status
- **THEN** artifact manifests and review tables SHALL record the non-written crop status
- **AND** downstream model inputs SHALL exclude that row unless a future explicit non-ROI contract supports it

### Requirement: Historical or compatibility artifacts are labeled
Quantification and segmentation review artifacts SHALL distinguish supported runtime artifacts from compatibility or historical artifacts.

#### Scenario: Artifact lacks supported provenance
- **WHEN** a quantification or segmentation report references an artifact that lacks current supported provenance
- **THEN** the report SHALL label the artifact as historical, compatibility, or non-supported
- **AND** the artifact SHALL NOT be used as evidence for scientific promotion
