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
