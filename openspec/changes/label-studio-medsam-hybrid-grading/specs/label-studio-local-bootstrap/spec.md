## MODIFIED Requirements

### Requirement: CLI SHALL bootstrap Label Studio from an image directory
The system SHALL provide `eq labelstudio start <image-dir>` where `<image-dir>` is a positional filesystem argument, AND SHALL additionally accept `eq labelstudio start --images <image-dir>` so automation retains the legacy flag spelling.

#### Scenario: Collaborator positional image directory is valid
- **WHEN** the user runs `eq labelstudio start /path/to/images`
- **THEN** the command discovers supported images recursively
- **AND** prepares a Label Studio project using `configs/label_studio_glomerulus_grading.xml` augmented by YAML-driven hybrid preload settings sourced from `configs/label_studio_medsam_hybrid.yaml` when present (or overridden via administrator `--config`)
- **AND** prints the Label Studio URL, collaborator login identifiers, optional project-specific URL targets, companion readiness summary when hybrid assists are mandated

#### Scenario: Legacy `--images` alias remains valid
- **WHEN** the user runs `eq labelstudio start --images /path/to/images`
- **THEN** the command behaves identically to the positional invocation

#### Scenario: Image directory is missing
- **WHEN** the positional path or `--images` path does not resolve to an accessible directory during bootstrap
- **THEN** the command fails before starting Docker or calling Label Studio
- **AND** the error identifies the missing argument naming pattern for deterministic automation logs

### Requirement: Bootstrap SHALL preserve image metadata for downstream grading
The system SHALL preserve each imported image's relative path metadata so later exports can be traced back to source files.

#### Scenario: Nested image directory is imported
- **WHEN** the image root contains `animal_1/kidney_a/image_001.tif`
- **THEN** the generated task includes `source_relative_path = animal_1/kidney_a/image_001.tif`
- **AND** includes `source_filename = image_001.tif`
- **AND** includes `subject_hint = animal_1`

#### Scenario: Flat image directory metadata for dry-run
- **WHEN** the image root contains `image_001.tif`
- **THEN** dry-run task generation includes `subject_hint = image_001`
- **AND** full Label Studio API bootstrap fails with an actionable error requiring images to be placed in at least one subfolder under the supplied image directory

## ADDED Requirements

### Requirement: YAML-first configuration MUST govern hybrid preload and companion coupling
Hybrid MedSAM mask release identifiers, companion service endpoints, enforced health probing, and restricted contingency bypass switches MUST reside in YAML so collaborators seldom pass long CLI argument lists beyond the positional image directory or optional administrator `--config`.

#### Scenario: Default collaborator profile
- **WHEN** operators omit `--config`
- **THEN** bootstrap reads `configs/label_studio_medsam_hybrid.yaml` resolved through `src/eq/utils/paths.py` helpers unless an explicit administrator override disables hybrid features

#### Scenario: Invalid YAML disables bootstrap
- **WHEN** the YAML violates the documented hybrid schema (`mask_release_id` missing while hybrid enabled without offline allowance, malformed URLs)
- **THEN** bootstrap terminates before Docker container creation or API imports with structured diagnostics referencing invalid keys only
