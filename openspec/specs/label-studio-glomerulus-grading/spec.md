# label-studio-glomerulus-grading Specification

## Purpose

Define the Label Studio-facing contract for human-first glomerulus-instance grading. The contract ensures each human grade is linked to exactly one complete glomerulus region, cut-off glomeruli are excluded explicitly, named grader provenance is preserved, and exported records can support later second review, adjudication, variability analysis, and downstream rollups.

## Requirements

### Requirement: Primary grading SHALL use glomerulus instances as the atomic unit
The system SHALL represent each gradeable unit as a stable `image_id + glomerulus_instance_id` record rather than as a whole-image average score.

#### Scenario: Multiple complete glomeruli in one image
- **WHEN** a Label Studio task contains two complete glomerulus regions in one image
- **THEN** the ingested records include two distinct `glomerulus_instance_id` values under the same `image_id`

#### Scenario: Historical image-average score is encountered
- **WHEN** an input provides only an image-level average score without per-glomerulus regions
- **THEN** the system rejects it for the glomerulus-instance contract and reports that image-level averages are legacy baseline data, not per-glomerulus ground truth

### Requirement: Complete glomerulus grades SHALL be linked to exactly one ROI or region
The system SHALL require every human grade to link to one complete glomerulus ROI, mask, or Label Studio region reference.

#### Scenario: Grade has a linked complete region
- **WHEN** a Label Studio annotation contains a grade linked to a region marked complete
- **THEN** the system emits a glomerulus-level record containing the grade, `image_id`, `glomerulus_instance_id`, and region or ROI reference

#### Scenario: Grade has no linked region
- **WHEN** a Label Studio annotation contains a grade that cannot be linked to a glomerulus region
- **THEN** validation fails with a hard error and no glomerulus-level record is emitted for that grade

#### Scenario: One region has duplicate active grades from one grader
- **WHEN** one grader submits duplicate active grades for the same `image_id + glomerulus_instance_id` in the same grading pass
- **THEN** validation fails with a duplicate-grade error

### Requirement: Cutoff glomeruli SHALL be excluded explicitly
The system SHALL represent cut-off or partial glomeruli as excluded candidates with `exclusion_reason = cutoff_partial_glomerulus` and SHALL prevent them from receiving grades or contributing to rollups, training labels, or model-performance metrics.

#### Scenario: Cutoff glomerulus is marked excluded
- **WHEN** a Label Studio annotation marks a glomerulus region as cut off or partial
- **THEN** the system emits an excluded candidate record with no human grade and `exclusion_reason = cutoff_partial_glomerulus`

#### Scenario: Cutoff glomerulus has a grade
- **WHEN** a Label Studio annotation assigns a grade to a region marked cut off or partial
- **THEN** validation fails with an excluded-region-grade error

#### Scenario: Rollup receives excluded candidate
- **WHEN** downstream rollup preparation reads an excluded cutoff candidate
- **THEN** the excluded candidate does not contribute to image, kidney, animal, training, or model-performance score aggregates

### Requirement: Grader provenance SHALL come from Label Studio annotation identity
The system SHALL preserve named human grader provenance from Label Studio annotation data, including annotation ID, task ID, `completed_by` identity when available, timestamps, and lead time.

#### Scenario: Export includes completed_by object
- **WHEN** a Label Studio export includes `completed_by` with user ID, email, first name, and last name
- **THEN** the system preserves the user ID and available display fields in the glomerulus-level provenance record

#### Scenario: Export includes completed_by ID only
- **WHEN** a Label Studio export includes `completed_by` as an ID only
- **THEN** the system preserves that ID and marks unavailable display fields as absent rather than inventing names

#### Scenario: Grader provenance is missing
- **WHEN** a submitted primary grade lacks usable annotation or grader identity provenance
- **THEN** validation fails with a missing-provenance error

### Requirement: Exports SHALL preserve glomerulus-level provenance for downstream rollups
The system SHALL emit glomerulus-level records that preserve image identity, glomerulus identity, ROI or region reference, completeness status, exclusion reason, human grade, grader provenance, annotation provenance, and export provenance.

#### Scenario: Complete glomerulus is exported
- **WHEN** a complete glomerulus has a valid linked grade and grader provenance
- **THEN** the export includes the glomerulus-level record with enough provenance to trace the grade back to its Label Studio task, annotation, user, and region

#### Scenario: Rollup-ready export is generated
- **WHEN** downstream rollup preparation is requested
- **THEN** the export keeps source glomerulus record IDs so image, kidney, or animal averages can be traced back to included complete glomeruli

### Requirement: Stage 1 SHALL remain model-blind during primary grading
The system SHALL NOT expose model grade suggestions during primary human grading in this contract stage.

#### Scenario: Primary grading config is generated
- **WHEN** the primary Label Studio grading configuration is created
- **THEN** it does not include visible model grade, model confidence, model disagreement, or model decision-state fields for the grader to review before submitting a primary grade

#### Scenario: Model fields appear in an input export
- **WHEN** Stage 1 ingestion encounters model grade or model decision-state fields
- **THEN** the system does not use those fields to satisfy primary human grade requirements
