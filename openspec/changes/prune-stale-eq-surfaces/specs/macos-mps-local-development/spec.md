## MODIFIED Requirements

### Requirement: Mac certification uses current active entrypoints
Mac local-development certification SHALL use active current-namespace entrypoints and SHALL NOT rely on retired production, no-op, or historical-compatibility surfaces.

#### Scenario: CLI help is inspected
- **WHEN** `eq --help` is displayed
- **THEN** it lists only active workflows and does not show retired commands such as `production`, `data-load`, `extract-features`, `quantify`, `process-data`, or `audit-derived`

#### Scenario: Historical FastAI pickle artifact is encountered
- **WHEN** a model artifact requires removed project modules, old FastAI transform namespaces, incompatible NumPy pickle namespaces, or `__main__` patching to load
- **THEN** active model loading treats the artifact as unsupported and fails closed unless a separate compatibility change explicitly supports it
