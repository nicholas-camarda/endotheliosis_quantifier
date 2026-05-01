## Purpose
Define the completed MedSAM automatic-prompt glomeruli workflow, including current-segmenter proposal-box generation, automatic MedSAM mask evaluation, gated derived-mask generation, primary generated-segmenter transition criteria, and evidence-gated fine-tuning recommendations.
## Requirements
### Requirement: Automatic pilot reuses admitted manual-mask validation inputs
The system SHALL validate automatic MedSAM prompts on admitted manual-mask rows before generating broad derived masks.

#### Scenario: Automatic pilot inputs are selected
- **WHEN** `medsam_automatic_glomeruli_prompts` runs without an explicit input CSV
- **THEN** it SHALL read `raw_data/cohorts/manifest.csv` under the active runtime root
- **AND** it SHALL select the same eligible input class as `medsam_manual_glomeruli_comparison`: `admission_status=admitted`, `lane_assignment` in `manual_mask_core` or `manual_mask_external`, non-empty image and mask paths, and existing image and mask files
- **AND** it SHALL sample deterministically across cohorts and subjects with a default 20-row target

#### Scenario: Automatic pilot input manifest is written
- **WHEN** pilot input rows are selected
- **THEN** the workflow SHALL write `inputs.csv`
- **AND** each row SHALL include `manifest_row_id`, `cohort_id`, `lane_assignment`, `source_sample_id`, `image_path`, `mask_path`, `selection_rank`, and `selection_reason`

### Requirement: Proposal boxes come from tiled current-segmenter probabilities
The workflow SHALL generate automatic MedSAM prompt boxes from tiled current-segmenter probability maps rather than manual masks or whole-field direct resize.

#### Scenario: Current segmenter probabilities are generated
- **WHEN** the workflow evaluates a selected pilot image
- **THEN** it SHALL use tiled full-field current-segmenter inference
- **AND** it SHALL record candidate artifact path, threshold, tile size, stride, expected size, preprocessing contract, and tile count
- **AND** it SHALL NOT pass the whole high-resolution image directly through `PredictionCore` resize-to-model-input preprocessing

#### Scenario: Proposal boxes are derived
- **WHEN** a current-segmenter probability map is available
- **THEN** the workflow SHALL threshold it using configured proposal thresholds
- **AND** it SHALL derive connected-component boxes with configured minimum component area, maximum component area, padding, image-bound clipping, overlapping-box merge IoU, and maximum boxes per image
- **AND** it SHALL write `proposal_boxes.csv` with one row per generated, merged, skipped, or overflow proposal decision

#### Scenario: Proposal boxes are evaluated against manual components
- **WHEN** manual masks are available for selected pilot rows
- **THEN** the workflow SHALL compute proposal recall against manual connected components
- **AND** it SHALL write `proposal_recall.csv` with candidate artifact, threshold, manifest row, cohort, lane assignment, manual component count, matched manual component count, missed manual component count, proposal count, overflow count, and box-summary fields

### Requirement: MedSAM automatic masks use proposal boxes
The workflow SHALL run MedSAM using automatic proposal boxes and label outputs as automatic-prompt evidence.

#### Scenario: MedSAM automatic inference runs
- **WHEN** automatic proposal boxes are available for a selected image
- **THEN** the workflow SHALL run MedSAM using `/Users/ncamarda/mambaforge/envs/medsam/bin/python`, `/Users/ncamarda/Projects/MedSAM`, and `/Users/ncamarda/Projects/MedSAM/work_dir/MedSAM/medsam_vit_b.pth`
- **AND** it SHALL save one union automatic MedSAM mask per selected image, candidate artifact, and selected proposal threshold under `medsam_auto_masks/`
- **AND** it SHALL record every prompt, command, checkpoint path, checkpoint hash when readable, device, return code, proposal source, and failure reason in provenance artifacts

#### Scenario: Automatic prompt fails
- **WHEN** MedSAM inference fails for an automatic prompt or image
- **THEN** the workflow SHALL record the failure in `prompt_failures.csv`
- **AND** it SHALL NOT silently substitute the manual mask, oracle MedSAM mask, current-segmenter mask, or an empty mask as a successful automatic MedSAM output

### Requirement: Automatic prompt evidence is compared with oracle and current evidence
The workflow SHALL compare automatic MedSAM masks against the same manual masks and report the automatic gap from oracle-prompt MedSAM.

#### Scenario: Automatic metrics are computed
- **WHEN** automatic MedSAM masks and manual masks are available for the same selected image
- **THEN** `metrics.csv` SHALL include method, prompt mode, candidate artifact, proposal threshold, manifest row, cohort, lane assignment, Dice, Jaccard, precision, recall, pixel accuracy, manual foreground fraction, prediction foreground fraction, area ratio, manual component count, prediction component count, and bbox summary fields
- **AND** current-segmenter baseline and oracle MedSAM metrics SHALL be included or referenced so automatic prompt performance can be interpreted against both

#### Scenario: Aggregates are written
- **WHEN** per-image metrics are available
- **THEN** `metric_by_source.csv` SHALL aggregate metrics by method, prompt mode, candidate artifact, proposal threshold, `cohort_id`, and `lane_assignment`
- **AND** `summary.json` SHALL report the best automatic prompt setting, the oracle gap, prompt failure count, proposal recall, and whether configured gates passed

#### Scenario: Review overlays are written
- **WHEN** a selected image has manual masks and automatic proposal outputs
- **THEN** the workflow SHALL write visual overlays or review panels showing raw image, manual mask, oracle MedSAM mask when available, automatic MedSAM mask, current-segmenter probability or mask, proposal boxes, missed manual components, and method labels

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

### Requirement: Automatic prompt conclusions remain audit-scoped
The workflow SHALL distinguish automatic-prompt mask agreement from model promotion, manual truth replacement, and downstream quantification validity.

#### Scenario: Automatic prompt summary is interpreted
- **WHEN** `summary.json` or report markdown is written
- **THEN** it SHALL state that automatic-prompt MedSAM performance depends on proposal-box localization
- **AND** it SHALL state that replacing raw/manual masks is not performed by this workflow
- **AND** it SHALL NOT update README-facing model performance claims or make downstream grading claims

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

### Requirement: Fine-tuning decision is evidence-gated
The workflow SHALL classify whether failures are due to localization, MedSAM boundary quality, or downstream integration before recommending fine-tuning.

#### Scenario: Proposal localization is insufficient
- **WHEN** proposal recall misses configured manual-component coverage gates while oracle MedSAM remains strong
- **THEN** `summary.json` SHALL recommend improving the automatic box proposer before opening a MedSAM fine-tuning change
- **AND** it SHALL identify missed-component counts and candidate/threshold settings that failed

#### Scenario: MedSAM boundaries are insufficient despite adequate localization
- **WHEN** proposal recall passes but automatic MedSAM Dice, Jaccard, area ratio, or overlay review fail gates
- **THEN** `summary.json` SHALL recommend opening a separate MedSAM/SAM fine-tuning proposal
- **AND** it SHALL identify the training evidence needed from admitted manual masks, MedSAM checkpoint provenance, and evaluation outputs

#### Scenario: Prompt-based MedSAM is sufficient
- **WHEN** proposal recall and automatic MedSAM mask-quality gates pass
- **THEN** `summary.json` SHALL recommend prompt-based MedSAM automatic generation before fine-tuning
- **AND** it SHALL NOT recommend fine-tuning as the next step unless a separate downstream review identifies a new failure mode

