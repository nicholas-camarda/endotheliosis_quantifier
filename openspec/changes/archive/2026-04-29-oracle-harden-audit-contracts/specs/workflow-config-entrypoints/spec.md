## ADDED Requirements

### Requirement: Supported workflow handoffs use exact artifact references
Workflow configs that consume supported upstream model or prediction artifacts SHALL name exact artifact paths and SHALL NOT resolve artifacts by latest-glob selection.

#### Scenario: Base artifact path is a glob
- **WHEN** a workflow config references a supported base artifact through a glob or latest-artifact selector
- **THEN** `eq run-config` SHALL fail during config validation
- **AND** the error SHALL request an exact `artifact_path`

#### Scenario: Multiple candidate artifacts match an old selector
- **WHEN** more than one artifact could satisfy an old latest-glob pattern
- **THEN** the workflow SHALL treat the handoff as ambiguous
- **AND** it SHALL NOT select the lexicographically last or most recent file

#### Scenario: Exact artifact is provided
- **WHEN** a workflow config names an exact supported artifact path
- **THEN** `eq run-config` SHALL validate existence, loadability, current namespace, and required metadata before launching downstream work

### Requirement: Direct glomeruli training has no auto-base discovery
Direct glomeruli training SHALL require either an explicit base model for transfer training or an explicit no-base training mode.

#### Scenario: Transfer training omits base model
- **WHEN** direct glomeruli transfer training is requested without `--base-model`
- **THEN** the command SHALL fail before training
- **AND** it SHALL NOT auto-discover a local mitochondria model

#### Scenario: No-base comparator is requested
- **WHEN** no-base comparator training is requested
- **THEN** the command SHALL require the explicit no-base mode
- **AND** it SHALL record that the encoder initialization is the ImageNet-pretrained no-mitochondria-base comparator
