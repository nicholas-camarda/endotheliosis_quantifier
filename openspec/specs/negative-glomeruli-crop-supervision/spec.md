# negative-glomeruli-crop-supervision Specification

## Purpose
TBD - created by archiving change p1-add-negative-glomeruli-crop-supervision. Update Purpose after archive.
## Requirements
### Requirement: Negative glomeruli crops from unmasked source images require explicit annotation
The repository SHALL NOT treat arbitrary crops from larger MR/TIFF source images without masks as supported true negatives unless those crops have explicit negative annotation or equivalent provenance-backed source mapping.

#### Scenario: Unlabeled crop is sampled from a larger source image
- **WHEN** a crop is generated from a larger MR/TIFF source image that does not have a full segmentation mask
- **THEN** that crop remains source material only
- **AND** it SHALL NOT be presented as a supported negative glomeruli training example

#### Scenario: Crop receives explicit negative annotation
- **WHEN** a crop from an unmasked larger source image is explicitly reviewed and recorded as containing no glomerulus
- **THEN** the repository MAY treat it as a supported negative crop example
- **AND** the negative label must remain traceable to source image path, crop box, and review provenance

### Requirement: Negative crop manifests record auditable crop-level provenance
Supported negative glomeruli crop manifests SHALL contain enough information to reconstruct the source image, crop coordinates, negative label, and review provenance for every crop.

#### Scenario: Negative crop manifest row is accepted
- **WHEN** a manifest row is accepted as supported negative glomeruli supervision
- **THEN** it includes `negative_crop_id`, `source_image_path`, `source_image_sha256`, `source_cohort_id`, `crop_x_min`, `crop_y_min`, `crop_x_max`, `crop_y_max`, `coordinate_frame`, `label`, `annotation_status`, `reviewer_id`, `reviewed_at_utc`, `review_batch_id`, `review_protocol_version`, `negative_scope`, `source_mapping_method`, `source_mapping_status`, and `notes`
- **AND** `label` is `negative_glomerulus`
- **AND** `negative_scope` is `crop_only`

#### Scenario: Required manifest field is missing
- **WHEN** a negative crop manifest row is missing a required source, crop geometry, label, or review-provenance field
- **THEN** that row is not supported negative crop supervision
- **AND** training and promotion reports SHALL NOT count it as a true negative crop

### Requirement: Negative crop curation uses manifests rather than static patch dataset roots
Supported negative glomeruli crop supervision SHALL be expressed through manifests and related audit artifacts rather than by reviving static patch dataset directories as the active training interface.

#### Scenario: Negative crop curation artifacts are generated
- **WHEN** curated negative glomeruli crops are recorded from larger source images
- **THEN** the canonical generated outputs are manifests, audits, and review assets
- **AND** those generated outputs live under `derived_data/glomeruli_negative_crops/...`
- **AND** source TIFF images remain under `raw_data/cohorts/vegfri_mr/images/...` rather than being copied into a training-root patch directory

#### Scenario: Future training consumes curated negative crop supervision
- **WHEN** glomeruli training later uses curated negative crop supervision
- **THEN** the training contract continues to treat full-image dynamic patching as canonical
- **AND** negative crop manifests act as an additional supervised sampling input rather than becoming the canonical static training root

### Requirement: Negative crop supervision has crop-level interpretation only
Curated negative glomeruli supervision SHALL be interpreted as evidence that specific reviewed crop boxes lack glomeruli, not evidence that the entire source image lacks glomeruli.

#### Scenario: Negative crop is reported downstream
- **WHEN** a curated negative crop appears in training provenance, audit output, or promotion evidence
- **THEN** the report identifies it as crop-level negative supervision
- **AND** the report does not describe the source image as whole-image negative

