## ADDED Requirements

### Requirement: Reviewed label overrides are stable scored-cohort inputs
Reviewed label overrides SHALL be treated as explicit human-reviewed scored-cohort inputs with fail-closed validation and provenance. They MUST NOT be treated as transient outputs from a prior modeling run.

#### Scenario: Override rows are validated
- **WHEN** reviewed label overrides are supplied
- **THEN** the system validates required columns, unique `subject_image_id` rows, accepted score values, numeric score parsing, and row matches against the scored examples before applying any override

#### Scenario: Unmatched override row blocks modeling
- **WHEN** an override row does not match an eligible scored example
- **THEN** quantification fails closed before downstream modeling and records the unmatched `subject_image_id`

#### Scenario: Duplicate override row blocks modeling
- **WHEN** the override file contains duplicate `subject_image_id` rows
- **THEN** quantification fails closed before downstream modeling and records the duplicate identifiers

#### Scenario: Override provenance is written
- **WHEN** reviewed label overrides are applied
- **THEN** `scored_examples/score_label_overrides_audit.csv` and `scored_examples/score_label_overrides_summary.json` record the override path, override content hash, base scored input hash, annotation/mapping hashes when file-backed, segmentation artifact metadata reference, matched row count, unmatched row count, duplicate status, accepted score set, grouping identity, and effective target-definition version

#### Scenario: No overrides is explicit
- **WHEN** no reviewed label override file is supplied
- **THEN** the score contract records `label_overrides: none` rather than omitting the target-definition field

### Requirement: Scored cohort target definition includes grouping identity
The canonical scored-cohort contract SHALL include enough grouping identity to support subject-heldout modeling, subject-aware atlas stability, and leakage checks.

#### Scenario: Grouping identity is resolved
- **WHEN** the quantification input contract is resolved
- **THEN** it records `subject_id`, row identity, `subject_image_id` uniqueness status, grouping-key derivation, row count, and subject count
- **AND** missing or ambiguous grouping identity fails before label-dependent modeling

#### Scenario: Target-defining content drift is detectable
- **WHEN** the scored input table, manifest, annotation source, mapping file, reviewed override file, or segmentation artifact metadata changes content
- **THEN** the resolved target-definition provenance changes hash or target-definition version
- **AND** downstream metrics are not presented as directly comparable without that context
