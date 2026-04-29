## ADDED Requirements

### Requirement: Manifest training validates admitted image-mask pairs
Manifest-backed segmentation training SHALL validate every admitted training row before constructing training items and SHALL fail if any selected row lacks an exact readable image or mask.

#### Scenario: Admitted image is missing
- **WHEN** manifest-backed training selects an admitted row whose image path is missing or unreadable
- **THEN** training SHALL fail before DataBlock construction
- **AND** the error SHALL identify the manifest row, image path, and mask path

#### Scenario: Admitted mask is missing
- **WHEN** manifest-backed training selects an admitted row whose mask path is missing or unreadable
- **THEN** training SHALL fail before DataBlock construction
- **AND** the error SHALL identify the manifest row, image path, and mask path

#### Scenario: Alternate same-stem mask exists outside the manifest pair
- **WHEN** a manifest row declares an image/mask pair and another same-stem mask exists outside the mirrored manifest path
- **THEN** training SHALL use only the manifest-declared pair
- **AND** it SHALL NOT substitute a parent-directory, normalized-stem, or root-level same-stem mask

### Requirement: Dynamic patching split seed is explicit provenance
Dynamic-patching segmentation training SHALL receive the split seed from the workflow or training config and SHALL record it in training provenance.

#### Scenario: Training config provides a split seed
- **WHEN** a segmentation training workflow config declares a split seed
- **THEN** `build_segmentation_datablock_dynamic_patching` SHALL use that split seed for the internal train/validation split
- **AND** exported run metadata SHALL record the split seed

#### Scenario: Explicit split manifest is provided
- **WHEN** a supported explicit split manifest is provided
- **THEN** the explicit split manifest SHALL determine train/validation membership
- **AND** any random split seed SHALL be recorded as not used for membership assignment

### Requirement: Supported training exports require mandatory provenance
Supported segmentation training exports SHALL require split manifests, training history, git/code state, data root, training mode, and package-version provenance.

#### Scenario: Split manifest cannot be written
- **WHEN** a training run cannot write the split manifest for a supported export
- **THEN** model export SHALL fail
- **AND** no artifact SHALL be labeled supported

#### Scenario: Training history cannot be written
- **WHEN** a training run cannot write training history for a supported export
- **THEN** model export SHALL fail
- **AND** no artifact SHALL be labeled supported

#### Scenario: Git state cannot be recorded
- **WHEN** git state cannot be determined for a supported export
- **THEN** the run SHALL fail or mark the artifact as non-supported before model export
