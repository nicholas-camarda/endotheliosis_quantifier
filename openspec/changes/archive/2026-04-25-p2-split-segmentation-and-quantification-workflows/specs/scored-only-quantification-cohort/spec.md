## MODIFIED Requirements

### Requirement: Segmentation transport audit gate
The system SHALL require a dedicated cohort-specific segmentation transport-audit workflow before any scored-only cohort is admitted into downstream grading. The transport-audit workflow SHALL identify the segmentation artifact used, stratify reviewed examples by cohort and treatment group, record whether predictions remain non-degenerate and grading-usable on the audited slice, and keep those audit outputs separate from downstream quantification artifacts.

#### Scenario: Cohort fails transport audit
- **WHEN** the segmenter produces degenerate, missing, or grading-invalid glomerulus predictions on the audited scored-only cohort slice
- **THEN** the cohort SHALL be marked excluded for grading use
- **AND** the system SHALL write an explicit exclusion artifact with the failure reason under `output/segmentation_evaluation/`

#### Scenario: Cohort passes transport audit
- **WHEN** the segmenter produces non-degenerate, review-approved glomerulus predictions on the audited harmonized scored-only cohort slice
- **THEN** the cohort SHALL be eligible for predicted-ROI grading artifact generation
- **AND** the transport-audit record SHALL identify the explicit segmentation artifact and prediction outputs that downstream grading may consume

#### Scenario: Transport audit remains separate from quantification
- **WHEN** the dedicated transport-audit workflow is executed for an external cohort
- **THEN** it SHALL NOT train the ordinal model, emit quantification metrics, or write quantification review artifacts under `output/quantification_results/`
- **AND** downstream grading SHALL require a separate quantification workflow invocation

#### Scenario: MR transport audit tests high-resolution preprocessing
- **WHEN** the cohort uses giant whole-field source images such as the MR TIFF batches
- **THEN** the transport audit SHALL explicitly record the preprocessing or tiling strategy used for inference
- **AND** it SHALL fail closed if the cohort cannot be processed reliably under that strategy

### Requirement: MR phase-1 use is external concordance only
The system SHALL treat MR as an external concordance and transport-evaluation cohort in phase 1 rather than as a training-expansion cohort. MR rows may be harmonized, verified, and used for segmentation inference and concordance reporting, but SHALL NOT be surfaced as training-admitted predicted-ROI grading rows in phase 1.

#### Scenario: MR is blocked from training admission in phase 1
- **WHEN** the implementation builds downstream training inputs from the unified manifest
- **THEN** it SHALL exclude `cohort_id=vegfri_mr` rows from training admission even if MR harmonization and transport audit succeed

#### Scenario: MR concordance compares human and inferred medians
- **WHEN** the implementation runs the phase 1 MR concordance workflow
- **THEN** it SHALL compare the human image-level median derived from the workbook replicates against the inferred image-level median derived from segmented glomerulus predictions on the copied giant TIFF images

#### Scenario: MR inference path is explicit
- **WHEN** the implementation runs the phase 1 MR concordance workflow
- **THEN** it SHALL tile the copied giant TIFF images, run segmentation on those tiles, extract accepted inferred glomerulus ROIs, grade the accepted ROIs, and aggregate inferred ROI grades to an image-level median before concordance reporting

#### Scenario: MR outputs are evaluation artifacts
- **WHEN** the implementation completes for the current MR cohort
- **THEN** the output artifacts for `cohort_id=vegfri_mr` SHALL be transport-audit and concordance-evaluation artifacts rather than training-set expansion or general quantification artifacts
