## ADDED Requirements

### Requirement: Pilot selects admitted manual-mask rows deterministically
The system SHALL build the MedSAM/manual glomeruli comparison pilot from admitted manual-mask cohort manifest rows without using no-mask-smoke rows as manual reference.

#### Scenario: Pilot inputs are selected
- **WHEN** `medsam_manual_glomeruli_comparison` is run without an explicit input CSV
- **THEN** it SHALL read `raw_data/cohorts/manifest.csv` under the active runtime root
- **AND** it SHALL select only rows with `admission_status=admitted`, `lane_assignment` in `manual_mask_core` or `manual_mask_external`, non-empty `image_path`, non-empty `mask_path`, and existing image and mask files
- **AND** it SHALL sample deterministically across cohorts and subjects
- **AND** it SHALL target 20 rows with 10 `vegfri_dox` rows and 10 `lauren_preeclampsia` rows when enough eligible rows exist

#### Scenario: Pilot input manifest is written
- **WHEN** pilot input rows are selected
- **THEN** the workflow SHALL write `inputs.csv`
- **AND** each row SHALL include `manifest_row_id`, `cohort_id`, `lane_assignment`, `source_sample_id`, `image_path`, `mask_path`, `selection_rank`, and `selection_reason`

### Requirement: Generated masks and reports are isolated from raw data
The workflow SHALL treat raw cohort images and masks as read-only inputs and SHALL write generated artifacts only under the configured runtime output root.

#### Scenario: Output directory is not supplied
- **WHEN** the MedSAM/manual comparison workflow runs without an explicit output override
- **THEN** it SHALL write artifacts under `output/segmentation_evaluation/medsam_manual_glomeruli_comparison/<run_id>/` under the active runtime root
- **AND** it SHALL NOT write generated masks, overlays, metrics, or provenance into `raw_data/cohorts/**/images` or `raw_data/cohorts/**/masks`

#### Scenario: Generated mask artifacts are written
- **WHEN** MedSAM or current-segmenter masks are generated
- **THEN** MedSAM masks SHALL be written under `medsam_masks/`
- **AND** current-segmenter masks SHALL be written under `current_segmenter_masks/`
- **AND** visual comparison artifacts SHALL be written under `overlays/` or `review_panels/`

### Requirement: MedSAM oracle masks use manual-mask-derived component boxes
The first pilot SHALL evaluate MedSAM using oracle bounding-box prompts derived from existing manual masks and SHALL label those results as upper-bound prompt evidence.

#### Scenario: Oracle boxes are generated
- **WHEN** a selected manual mask contains foreground components
- **THEN** the workflow SHALL derive connected-component bounding boxes from the binarized manual mask
- **AND** it SHALL apply a configured padding policy without exceeding image bounds
- **AND** it SHALL write `oracle_boxes.csv` with box coordinates, component area, padding, source mask path, and prompt provenance

#### Scenario: MedSAM inference runs for oracle boxes
- **WHEN** oracle boxes are available for a selected image
- **THEN** the workflow SHALL run MedSAM using `/Users/ncamarda/mambaforge/envs/medsam/bin/python`, `/Users/ncamarda/Projects/MedSAM`, and `/Users/ncamarda/Projects/MedSAM/work_dir/MedSAM/medsam_vit_b.pth`
- **AND** it SHALL save one union MedSAM mask per selected image
- **AND** it SHALL record every prompt, command, checkpoint path, checkpoint hash when readable, device, return code, and failure reason in provenance artifacts

#### Scenario: MedSAM prompt fails
- **WHEN** MedSAM inference fails for a prompt or image
- **THEN** the workflow SHALL record the failure in `prompt_failures.csv`
- **AND** it SHALL NOT silently substitute the manual mask, current-segmenter mask, or an empty mask as a successful MedSAM output

### Requirement: Current segmenter baseline uses tiled full-field inference
The workflow SHALL compare the current glomeruli segmenter to manual masks using a tiled high-resolution inference path rather than whole-field single-pass resizing.

#### Scenario: Current segmenter baseline is generated
- **WHEN** the workflow generates current-segmenter masks for selected pilot images
- **THEN** it SHALL use tiled full-field inference with recorded `tile_size`, `stride`, `expected_size`, threshold, preprocessing contract, model artifact path, and tile count
- **AND** it SHALL write one current-segmenter mask per selected image and candidate artifact

#### Scenario: High-resolution image would be resized in a single pass
- **WHEN** a high-resolution pilot image is passed to current-segmenter inference
- **THEN** the workflow SHALL reject any path that sends the entire image directly through the low-level `PredictionCore` resize-to-model-input helper
- **AND** tests SHALL cover this guard so the invalid full-field single-pass path cannot be used as pilot evidence

### Requirement: Shared prediction core rejects unlabeled high-resolution direct resize
The shared `PredictionCore` preprocessing and prediction APIs SHALL reject unlabeled high-resolution full-field inputs before resizing them to model input size.

#### Scenario: Unlabeled high-resolution image reaches PredictionCore
- **WHEN** a caller passes a high-resolution image directly to `PredictionCore.preprocess_image`, `PredictionCore.preprocess_image_imagenet_normalized`, `PredictionCore.predict_segmentation_probability`, or `PredictionCore.predict_with_model` without labeling the input as a bounded local region
- **THEN** `PredictionCore` SHALL raise a clear error that instructs the caller to use tiled inference for full-field segmentation
- **AND** it SHALL NOT silently resize the full field to the model input size

#### Scenario: Bounded local region reaches PredictionCore
- **WHEN** a caller passes a bounded local image region with `input_role` equal to `tile`, `crop`, `patch`, or `roi`
- **THEN** `PredictionCore` MAY resize that bounded region to model input size
- **AND** prediction audit metadata SHALL record the input role used by the caller

#### Scenario: Legacy compatibility must resize a high-resolution full field
- **WHEN** a caller intentionally opts into high-resolution direct resize for a compatibility-only path
- **THEN** the caller SHALL pass an explicit override flag
- **AND** the prediction audit metadata SHALL record that high-resolution direct resize was explicitly allowed

### Requirement: Pilot reports comparable segmentation metrics and review artifacts
The workflow SHALL compare MedSAM oracle masks and current-segmenter masks against the same manual masks using explicit segmentation metrics and visual review outputs.

#### Scenario: Metrics are computed
- **WHEN** a generated mask and manual mask are available for the same selected image
- **THEN** `metrics.csv` SHALL include method, candidate artifact when applicable, manifest row, cohort, lane assignment, Dice, Jaccard, precision, recall, pixel accuracy, manual foreground fraction, prediction foreground fraction, area ratio, manual component count, prediction component count, and bbox summary fields

#### Scenario: Aggregates are written
- **WHEN** per-image metrics are available
- **THEN** `metric_by_source.csv` SHALL aggregate metrics by method, candidate artifact when applicable, `cohort_id`, and `lane_assignment`
- **AND** `summary.json` SHALL identify best descriptive agreement while stating that the pilot is audit evidence and not scientific promotion

#### Scenario: Review overlays are written
- **WHEN** a selected image has manual and generated masks
- **THEN** the workflow SHALL write visual overlays or review panels that show the raw image, manual mask, MedSAM mask when available, current-segmenter mask when available, oracle boxes, and method labels

### Requirement: Provenance and dependency status are explicit
The workflow SHALL make MedSAM dependency status, current model artifact status, selected inputs, and runtime outputs reproducible from generated provenance.

#### Scenario: Summary provenance is written
- **WHEN** the workflow starts
- **THEN** `summary.json` SHALL record workflow ID, run ID, runtime root, config path, MedSAM Python path, MedSAM repo path, MedSAM checkpoint path, selected current-segmenter artifact paths, input manifest path, output root, package/module availability, and command lines

#### Scenario: External MedSAM dependency is unavailable
- **WHEN** the configured MedSAM Python, repository, inference script, or checkpoint is missing
- **THEN** the workflow SHALL fail closed with a clear dependency error
- **AND** it SHALL NOT write generated mask artifacts that imply MedSAM completed successfully

### Requirement: Pilot conclusions remain audit-scoped
The workflow SHALL distinguish descriptive mask-agreement evidence from model promotion, downstream grading evidence, causal evidence, and external validation.

#### Scenario: Pilot summary is interpreted
- **WHEN** `summary.json` or report markdown is written
- **THEN** it SHALL state that oracle-prompt MedSAM results are upper-bound prompt evidence
- **AND** it SHALL state that automatic deployment performance requires a separate automatic-prompt evaluation
- **AND** it SHALL NOT update README-facing current model performance claims or label any segmentation model as scientifically promoted
