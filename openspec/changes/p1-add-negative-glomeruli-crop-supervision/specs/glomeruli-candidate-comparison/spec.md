## ADDED Requirements

### Requirement: Candidate comparison discloses negative crop supervision coverage
Glomeruli candidate comparison and promotion reports SHALL disclose whether curated negative crop supervision was used to train each candidate and whether deterministic evaluation includes negative-crop coverage.

#### Scenario: Candidate was trained without curated negative crop supervision
- **WHEN** a glomeruli candidate report is generated for a candidate whose training provenance has no supported negative crop manifest
- **THEN** the report records `negative_crop_supervision_status=absent`
- **AND** it does not imply that unmasked MR/TIFF crops were used as true negative supervision

#### Scenario: Candidate was trained with curated negative crop supervision
- **WHEN** a glomeruli candidate report is generated for a candidate whose training provenance includes a supported negative crop manifest
- **THEN** the report records the negative manifest path, manifest hash, negative crop count, source image count, review protocol version, and sampler weight
- **AND** the report states that the supervision is crop-level rather than whole-image negative evidence

#### Scenario: Deterministic promotion evidence includes background crops
- **WHEN** deterministic promotion evidence includes background or negative crop categories
- **THEN** the report distinguishes mask-derived background crops from curated negative crops from unmasked source images
- **AND** it records coverage counts for each source of background evidence separately
