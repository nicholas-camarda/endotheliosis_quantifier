## ADDED Requirements

### Requirement: Mask-derived background crops are supported negative supervision
Glomeruli training SHALL support background crop boxes from paired image/mask rows when the corresponding mask crop contains zero foreground pixels.

#### Scenario: Mask-derived background crop is accepted
- **WHEN** a crop box is generated from an admitted paired glomeruli image/mask row
- **AND** the mask crop contains zero foreground pixels
- **THEN** the crop SHALL be eligible for the `mask_derived_background` label
- **AND** it SHALL be eligible for supervised negative/background training evidence
- **AND** the source image, source mask, crop box, and zero-foreground validation result are recorded in a manifest or training provenance

#### Scenario: Mask-derived background crop overlaps foreground
- **WHEN** a proposed mask-derived background crop has any foreground pixels in the paired mask crop
- **THEN** it SHALL NOT be accepted as negative/background supervision
- **AND** the audit records the rejection count

### Requirement: Unreviewed MR/TIFF proposals are not trainable
Glomeruli training SHALL NOT use unmasked MR/TIFF crop proposals as negative supervision unless those rows have reviewed negative annotation status.

#### Scenario: MR/TIFF crop proposal is generated
- **WHEN** a crop proposal is generated from `raw_data/cohorts/vegfri_mr/images/`
- **THEN** it is recorded as `proposed_review_only`
- **AND** it SHALL NOT be consumed by training

#### Scenario: Reviewed MR/TIFF crop is accepted
- **WHEN** a curated negative manifest row has `label=negative_glomerulus`, `annotation_status=reviewed_negative`, and `negative_scope=crop_only`
- **THEN** training MAY consume that crop as supervised negative/background evidence
- **AND** the manifest path and hash are recorded in training provenance

### Requirement: Negative crop manifests are additional sampler inputs
Negative/background crop manifests SHALL be consumed as additional supervised sampler inputs while preserving full-image dynamic patching as the canonical glomeruli training mode.

#### Scenario: Training uses negative crop manifests
- **WHEN** glomeruli training is configured with a valid negative crop manifest
- **THEN** the DataBlock or sampler returns image crops and all-zero masks for those negative crop samples
- **AND** training still reads source pixels from canonical source image paths
- **AND** no active static `image_patches/` or `mask_patches/` training root is required

### Requirement: Training provenance records negative supervision state
Glomeruli training metadata SHALL disclose whether negative/background crop supervision was present.

#### Scenario: Training completes with negative supervision enabled
- **WHEN** a glomeruli model artifact is exported after using negative/background crop supervision
- **THEN** metadata records `negative_crop_supervision_status`, `negative_crop_manifest_path`, `negative_crop_manifest_sha256`, `negative_crop_count`, `mask_derived_background_crop_count`, `curated_negative_crop_count`, `negative_crop_source_image_count`, `negative_crop_review_protocol_version`, and `negative_crop_sampler_weight`

#### Scenario: Training completes without negative supervision
- **WHEN** no validated negative/background manifest is supplied
- **THEN** metadata records `negative_crop_supervision_status=absent`
 
### Requirement: Augmentation policy is explicit provenance
Glomeruli training metadata SHALL record the actual augmentation policy used by the DataBlock.

#### Scenario: Training uses the default FastAI augmentation policy
- **WHEN** glomeruli training builds DataLoaders with default batch transforms
- **THEN** metadata records the FastAI augmentation settings and repo constants
- **AND** it does not claim config-defined gaussian noise or brightness/contrast settings were active unless code actually applied them
