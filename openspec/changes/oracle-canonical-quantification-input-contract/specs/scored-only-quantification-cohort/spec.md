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
- **THEN** `scored_examples/score_label_overrides_audit.csv` and `scored_examples/score_label_overrides_summary.json` record the override path, content hash, matched row count, unmatched row count, duplicate status, accepted score set, and effective target-definition version

#### Scenario: No overrides is explicit
- **WHEN** no reviewed label override file is supplied
- **THEN** the score contract records `label_overrides: none` rather than omitting the target-definition field
