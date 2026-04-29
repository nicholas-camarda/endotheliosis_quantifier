## ADDED Requirements

### Requirement: Workflow docs point to supported entrypoints
Operational documentation for workflow entrypoints SHALL point users to supported current commands and configs, especially `eq run-config`, and SHALL NOT direct users to missing historical inference modules or fallback integration paths.

#### Scenario: Quantification command is documented
- **WHEN** active docs describe the endotheliosis quantification workflow
- **THEN** the primary command is `eq run-config --config configs/endotheliosis_quantification.yaml`

#### Scenario: Segmentation command is documented
- **WHEN** active docs describe segmentation training or validation
- **THEN** they reference current YAML configs, current-namespace supported artifacts, and fail-closed loading behavior

#### Scenario: Historical command is present outside archive
- **WHEN** active docs outside `docs/archive/` include a command that executes `historical_glomeruli_inference.py`
- **THEN** validation fails because historical fallback execution is not a supported workflow entrypoint
