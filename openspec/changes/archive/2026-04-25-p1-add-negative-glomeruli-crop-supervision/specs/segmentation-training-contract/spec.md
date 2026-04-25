## ADDED Requirements

### Requirement: Unmasked large-image crops are not implicit negative supervision
Segmentation training SHALL NOT silently treat crops from larger unmasked glomeruli source images as supported negative supervision.

#### Scenario: Training pipeline inspects an unlabeled large-image crop source
- **WHEN** glomeruli training or curation code encounters larger MR/TIFF source images without full masks
- **THEN** those images are treated as source material only
- **AND** their unlabeled crops SHALL NOT be treated as true negative glomeruli examples by default

#### Scenario: Curated negative crop supervision is added later
- **WHEN** glomeruli training uses negative crop supervision from larger unmasked source images
- **THEN** those negative crops must come from an explicit annotation manifest or equivalent provenance-backed source mapping
- **AND** the resulting training provenance records that curated negative crop supervision was used

### Requirement: Curated negative crop manifests are additional sampler inputs
Segmentation training SHALL consume curated negative glomeruli crop manifests only as additional supervised sampler inputs while preserving full-image dynamic patching as the canonical glomeruli training contract.

#### Scenario: Training is configured with curated negative crops
- **WHEN** glomeruli training receives a supported negative crop manifest
- **THEN** the primary positive and mask-bearing data source remains the selected full-image root or manifest-backed `raw_data/cohorts` registry
- **AND** the negative crop manifest contributes reviewed crop boxes to the sampler without creating or requiring active `image_patches/` or `mask_patches/` directories

#### Scenario: Training provenance is written
- **WHEN** glomeruli training writes model provenance or run metadata
- **THEN** it records `negative_crop_supervision_status`, `negative_crop_manifest_path`, `negative_crop_manifest_sha256`, `negative_crop_count`, `negative_crop_source_image_count`, `negative_crop_review_protocol_version`, and `negative_crop_sampler_weight`
- **AND** absence of a curated manifest is recorded as `negative_crop_supervision_status=absent`
