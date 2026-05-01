## MODIFIED Requirements

### Requirement: Exports SHALL preserve glomerulus-level provenance for downstream rollups
The system SHALL emit glomerulus-level records that preserve image identity, glomerulus identity, ROI or region reference, completeness status, exclusion reason, human grade, grader provenance, annotation provenance, export provenance, and when hybrid ingestion is enabled SHALL additionally preserve `mask_release_id`, canonical `mask_source`, `proposal_kind`, `region_edit_state`, latest authoritative timestamps, supersession references sufficient to trace replacements of prior drafts, companion interaction identifiers sufficient to audit iterative corrections derived from ML assists.

#### Scenario: Complete glomerulus is exported
- **WHEN** a complete glomerulus has a valid linked grade and grader provenance
- **THEN** the export includes the glomerulus-level record with enough provenance to trace the grade back to its Label Studio task, annotation, user, region, and hybrid lineage enums when assists were used

#### Scenario: Rollup-ready export is generated
- **WHEN** downstream rollup preparation is requested
- **THEN** the export keeps source glomerulus record IDs so image, kidney, or animal averages can be traced back to included complete glomeruli and hybrid lineage distinguishes training-ready rows from rejected proposals

#### Scenario: Auto preload with brush refinements graded
- **WHEN** a complete glomerulus receives a finalized grade after a grader modifies MedSAM-derived geometry sourced from preload proposals
- **THEN** lineage records `proposal_kind=auto_preload`, `mask_release_id=<release>`, `region_edit_state=human_refined_boundary`, authoritative numeric grade linkage remains one-to-one with final region ID

#### Scenario: Box-assisted region without preload coverage
- **WHEN** a grader submits a complete glomerulus after invoking box-assisted MedSAM because preload proposals were unavailable
- **THEN** lineage records `proposal_kind=box_assisted_manual`, retains companion timestamps or request identifiers when present, persists edit state classifications identical schema as preload path

## ADDED Requirements

### Requirement: Graded glomerulus regions SHALL use brush-compatible geometry in hybrid workflows

When **`label-studio-medsam-hybrid-grading`** is enabled, exported glomerulus-instance records MUST derive from **brushlabels** (or equivalent mask raster semantics) for `glomerulus_roi` (or successor region name), consistent with human mask editing; rectangle-only annotations MUST NOT be the sole graded mask primitive for complete glomeruli unless a future capability change explicitly supersedes this contract.

#### Scenario: Export after brush refinement

- **WHEN** a grader finalizes a complete glomerulus after brush edits on MedSAM-sourced geometry
- **THEN** the export encodes region geometry in the brush-compatible format consumed by `load_glomerulus_grading_records` and validators

### Requirement: Validator MUST reject contradictory lineage payloads
Hybrid exports MUST fail validation when enumerated lineage fields contradict geometry references (for example preload metadata referencing missing release identifiers) preventing silent ingestion of contradictory training-ready tables.

#### Scenario: Contradictory lineage
- **WHEN** export JSON lists `proposal_kind=auto_preload` but omits referenced `mask_release_id` binding or references unknown registry entries
- **THEN** parsers raise hard validation failures before emitting dataframe rows downstream
