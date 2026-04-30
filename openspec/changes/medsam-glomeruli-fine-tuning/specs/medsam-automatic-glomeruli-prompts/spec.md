## MODIFIED Requirements

### Requirement: Broad MedSAM mask generation is isolated and gated
The workflow SHALL prevent broad replacement-like generation unless automatic-prompt or fine-tuned MedSAM evidence has passed explicit gates, and reusable generated-mask releases SHALL use the canonical generated-mask derived-data layout.

#### Scenario: Broad derived-mask generation is requested
- **WHEN** the workflow is configured to generate MedSAM automatic masks beyond the 20-row manual pilot
- **THEN** it SHALL require a completed pilot or fine-tuning summary path whose gates passed
- **AND** reusable generated masks SHALL be written under `derived_data/generated_masks/glomeruli/<mask_source>/<mask_release_id>/`
- **AND** fine-tuned MedSAM generated masks SHALL use `derived_data/generated_masks/glomeruli/medsam_finetuned/<mask_release_id>/`
- **AND** it SHALL write a manifest mapping each derived mask to its source image, proposal source, candidate artifact or checkpoint, threshold, MedSAM checkpoint, command provenance, and generation status
- **AND** reusable generated-mask releases SHALL be indexed in the central registry `derived_data/generated_masks/glomeruli/manifest.csv`

#### Scenario: Generated masks would be written into raw data
- **WHEN** any configured output path resolves under `raw_data/cohorts/**/images` or `raw_data/cohorts/**/masks`
- **THEN** the workflow SHALL fail closed with a clear path-isolation error
- **AND** it SHALL NOT write generated masks, overlays, metrics, or provenance into raw cohort directories

### Requirement: Primary generated-segmenter transition is gated
The system SHALL treat MedSAM automatic or fine-tuned masks as a primary generated glomeruli segmentation candidate only after the relevant generated-mask gates pass.

#### Scenario: Automatic prompt pilot passes gates
- **WHEN** `summary.json` reports that proposal recall, automatic MedSAM metrics, prompt failure rate, area-ratio guardrails, and overlay review gates passed
- **THEN** the workflow MAY mark `medsam_automatic_glomeruli` as the recommended generated-mask candidate in generated provenance
- **AND** broad generated-mask release creation MAY be enabled under `derived_data/generated_masks/glomeruli/medsam_automatic/<mask_release_id>/`
- **AND** manual masks SHALL remain labeled as reference masks, not generated masks

#### Scenario: Fine-tuned MedSAM gates pass
- **WHEN** a fine-tuned MedSAM checkpoint passes fixed-split generated-mask adoption gates
- **THEN** the workflow MAY mark `medsam_finetuned_glomeruli` as the recommended generated-mask candidate in generated provenance
- **AND** generated-mask release creation MAY be enabled under `derived_data/generated_masks/glomeruli/medsam_finetuned/<mask_release_id>/`
- **AND** release rows in `derived_data/generated_masks/glomeruli/manifest.csv` SHALL record `adoption_tier=oracle_level_preferred`
- **AND** manual masks SHALL remain labeled as reference masks, not generated masks

#### Scenario: Fine-tuned MedSAM improves but misses oracle-level gates
- **WHEN** a fine-tuned MedSAM checkpoint improves over current automatic MedSAM and current segmenters but misses configured oracle-level gates
- **THEN** the workflow MAY write a generated-mask release only for explicit downstream comparison
- **AND** release rows in `derived_data/generated_masks/glomeruli/manifest.csv` SHALL record `adoption_tier=improved_candidate_not_oracle`
- **AND** the workflow SHALL NOT mark that release as the preferred generated glomeruli segmentation source

#### Scenario: Downstream configs opt into MedSAM generated masks
- **WHEN** a later config or workflow uses MedSAM generated masks as the generated glomeruli segmentation source
- **THEN** it SHALL record `mask_source=medsam_automatic_glomeruli`, `mask_source=medsam_finetuned_glomeruli`, or an equivalent explicit provenance field
- **AND** it SHALL record the central generated-mask registry path or generated-mask release manifest path, MedSAM checkpoint, proposal source, threshold, adoption tier, and run ID
- **AND** it SHALL preserve a fallback or comparator reference to the current segmenter artifacts until downstream feature and grading stability have been reviewed

#### Scenario: Documentation is updated after successful pilot or fine-tuning release
- **WHEN** automatic-prompt or fine-tuned MedSAM generated-mask release gates and downstream review gates pass
- **THEN** `docs/TECHNICAL_LAB_NOTEBOOK.md` SHALL describe the passing MedSAM release as the current preferred generated glomeruli segmentation candidate
- **AND** workflow config notes SHALL describe how to opt into the release manifest path and `mask_source`
- **AND** documentation SHALL state that raw cohort masks and manual reference masks are not overwritten
