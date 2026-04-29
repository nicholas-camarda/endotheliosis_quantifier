## ADDED Requirements

### Requirement: Label Studio current-score recovery fails closed
The Label Studio score loader SHALL treat the latest annotation source as authoritative and SHALL NOT automatically backfill missing current grades from git history or historical exports.

#### Scenario: Latest annotation lacks a grade
- **WHEN** Label Studio score recovery reads a latest annotation for an image and that annotation has no supported grade choice
- **THEN** the score loader SHALL report the image as missing a current score or fail according to the caller's missing-score policy
- **AND** it SHALL NOT search `git` history or older annotation exports for a replacement grade

#### Scenario: Historical source is requested explicitly
- **WHEN** a caller passes an explicit historical `annotation_source` such as `git:REV:path`
- **THEN** the score loader MAY read that named source
- **AND** the resulting score artifact SHALL record the explicit annotation source
- **AND** automatic historical source discovery SHALL remain disabled

### Requirement: Manifest roots stay in manifest quantification mode
Quantification runs started from a `raw_data/cohorts` manifest root SHALL use the manifest contract and SHALL NOT fall through to raw-project Label Studio inventory behavior.

#### Scenario: Manifest root receives raw Label Studio options
- **WHEN** `run_endotheliosis_quantification` receives a project root containing `manifest.csv` and also receives `score_source=labelstudio` or `annotation_source`
- **THEN** the run SHALL fail before raw-project inventory or scoring begins
- **AND** the error SHALL state that manifest roots require manifest-mode scoring unless a future explicit manifest override contract is implemented

#### Scenario: Manifest root is processed
- **WHEN** a manifest-root quantification run starts
- **THEN** it SHALL NOT write `raw_inventory.csv`
- **AND** every scored row SHALL come from the manifest-admitted scored mask-paired contract

### Requirement: Label Studio grade extraction uses one shared rule
All Label Studio grade extraction in quantification cohort building and score recovery SHALL use one shared function and SHALL reject ambiguous multi-choice grades unless a documented rubric resolves them.

#### Scenario: Multi-choice grade is encountered
- **WHEN** a Label Studio result contains more than one supported grade choice
- **THEN** cohort manifest building and score recovery SHALL produce the same outcome
- **AND** the shared extractor SHALL either reject the result with `LabelStudioScoreError` or apply one documented rule

#### Scenario: Single grade is encountered
- **WHEN** a Label Studio result contains exactly one supported grade choice
- **THEN** cohort manifest building and score recovery SHALL extract the same numeric grade
