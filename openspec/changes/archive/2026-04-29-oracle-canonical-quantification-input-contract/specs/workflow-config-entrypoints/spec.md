## ADDED Requirements

### Requirement: Quantification entrypoints resolve one canonical input contract
All user-facing quantification entrypoints SHALL resolve scored cohort data, score source, annotation source, mapping file, reviewed label overrides, segmentation artifact, and output root through one canonical quantification input contract before loading labels or fitting quantification models.

#### Scenario: YAML workflow uses canonical contract
- **WHEN** `eq run-config --config configs/endotheliosis_quantification.yaml` starts the `endotheliosis_quantification` workflow
- **THEN** the workflow resolves its inputs through the canonical quantification input contract before score loading, ROI extraction, learned ROI modeling, source-aware modeling, severe-aware modeling, or P3 grade-model fitting

#### Scenario: Direct quantification command uses same contract
- **WHEN** `eq quant-endo` is invoked with scored cohort inputs and reviewed label overrides
- **THEN** the command resolves the same canonical quantification input contract as the YAML workflow and records the same effective target-definition fields

#### Scenario: Direct contract-preparation command uses same contract
- **WHEN** `eq prepare-quant-contract` prepares label-dependent quantification artifacts
- **THEN** the command resolves the same reviewed-label and score-source contract as `eq run-config --config configs/endotheliosis_quantification.yaml`

#### Scenario: Direct command cannot satisfy contract
- **WHEN** a direct quantification command lacks required reviewed-label or score-source contract fields needed for the current workflow
- **THEN** the command fails before label loading or modeling and reports the missing fields plus the supported YAML workflow command

### Requirement: Quantification config references stable input roots
Committed quantification configs SHALL reference reviewed label override inputs from stable runtime-derived input locations, not from previous quantification output trees.

#### Scenario: Config points to reviewed input contract
- **WHEN** `configs/endotheliosis_quantification.yaml` declares `inputs.label_overrides`
- **THEN** the path is runtime-root relative under `derived_data/quantification_inputs/reviewed_label_overrides/endotheliosis_grade_model/`

#### Scenario: Prior output-tree override is rejected
- **WHEN** a committed quantification config points `inputs.label_overrides` under `output/quantification_results/`
- **THEN** validation fails because generated model outputs are not supported as required modeling inputs

#### Scenario: Override file is missing
- **WHEN** the resolved reviewed label override path does not exist
- **THEN** quantification fails closed before model fitting and records the missing input path in run diagnostics
