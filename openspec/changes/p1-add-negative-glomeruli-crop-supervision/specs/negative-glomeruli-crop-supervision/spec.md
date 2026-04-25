## ADDED Requirements

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

### Requirement: Negative crop curation uses manifests rather than static patch dataset roots
Supported negative glomeruli crop supervision SHALL be expressed through manifests and related audit artifacts rather than by reviving static patch dataset directories as the active training interface.

#### Scenario: Negative crop curation artifacts are generated
- **WHEN** curated negative glomeruli crops are recorded from larger source images
- **THEN** the canonical generated outputs are manifests, audits, and review assets
- **AND** those generated outputs live under `derived_data/...`

#### Scenario: Future training consumes curated negative crop supervision
- **WHEN** glomeruli training later uses curated negative crop supervision
- **THEN** the training contract continues to treat full-image dynamic patching as canonical
- **AND** negative crop manifests act as an additional supervised sampling input rather than becoming the canonical static training root
