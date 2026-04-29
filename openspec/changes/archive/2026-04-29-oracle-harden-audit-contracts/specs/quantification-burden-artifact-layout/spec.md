## ADDED Requirements

### Requirement: Quantification artifacts expose preprocessing, threshold, and ROI statuses
Quantification output artifacts SHALL preserve enough provenance to audit preprocessing, thresholding, and ROI inclusion decisions.

#### Scenario: Quantification review artifacts are written
- **WHEN** `eq run-config --config configs/endotheliosis_quantification.yaml` completes
- **THEN** quantification review artifacts SHALL include the segmentation inference preprocessing contract
- **AND** they SHALL include any segmentation threshold value used to generate masks or probabilities consumed by quantification
- **AND** they SHALL include counts by ROI status, including rows excluded because of size mismatch or below-threshold components

#### Scenario: ROI crop is not written
- **WHEN** a row has no valid ROI crop because of a fail-closed ROI status
- **THEN** artifact manifests and review tables SHALL record the non-written crop status
- **AND** downstream model inputs SHALL exclude that row unless a future explicit non-ROI contract supports it

### Requirement: Historical or compatibility artifacts are labeled
Quantification and segmentation review artifacts SHALL distinguish supported runtime artifacts from compatibility or historical artifacts.

#### Scenario: Artifact lacks supported provenance
- **WHEN** a quantification or segmentation report references an artifact that lacks current supported provenance
- **THEN** the report SHALL label the artifact as historical, compatibility, or non-supported
- **AND** the artifact SHALL NOT be used as evidence for scientific promotion
