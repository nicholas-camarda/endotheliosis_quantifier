## ADDED Requirements

### Requirement: Active image-mask pairing is fail-closed
Supported segmentation training SHALL use one explicit image-mask pairing contract and SHALL NOT rescue missing masks with secondary fallback searches.

#### Scenario: Standard getter cannot resolve a mask
- **WHEN** `get_items_full_images(...)` inspects a full-image training root and `eq.data_management.standard_getters.get_y_full(...)` cannot resolve a mask for an image
- **THEN** the training-root inspection fails with an error identifying the unpaired image count and examples
- **AND** the loader does not attempt secondary glob, filename, or extension fallback pairing in the active training path

#### Scenario: Manifest-backed cohort root is used
- **WHEN** glomeruli training uses the manifest-backed `raw_data/cohorts` registry root
- **THEN** the loader enumerates only admitted manifest rows with explicit image and mask paths
- **AND** it does not infer additional mask paths outside the manifest contract

#### Scenario: Direct paired raw-data root is used
- **WHEN** segmentation training uses a direct paired `raw_data/...` root containing `images/` and `masks/`
- **THEN** the accepted mask path for each image is the current supported path resolved by `get_y_full(...)`
- **AND** unsupported alternate legacy filename patterns are handled by a separate curation or migration step, not by the active trainer

### Requirement: Training artifact completeness is part of runtime support
Supported segmentation artifacts SHALL include the required evidence files written by training and export helpers.

#### Scenario: Supported model artifact is exported
- **WHEN** a supported segmentation training run exports a model artifact
- **THEN** the run directory contains the exported learner, run metadata, split manifest, training history, package/version provenance, data-root provenance, and training-mode provenance
- **AND** missing required evidence files block supported runtime status

#### Scenario: Optional visual evidence is missing
- **WHEN** optional visualization artifacts are unavailable because plotting failed
- **THEN** the model can remain runtime-compatible only if required provenance and metrics artifacts exist
- **AND** any scientific promotion claim remains blocked until required visual review artifacts are produced by a validation or comparison workflow
