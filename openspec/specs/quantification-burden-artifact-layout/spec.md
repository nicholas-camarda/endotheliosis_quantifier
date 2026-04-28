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
