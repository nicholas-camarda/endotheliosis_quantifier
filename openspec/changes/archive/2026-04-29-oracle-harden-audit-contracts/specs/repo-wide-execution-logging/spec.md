## ADDED Requirements

### Requirement: CLI setup failures are visible
The `eq` CLI SHALL fail visibly when mode setup or runtime directory setup fails for commands that require those resources.

#### Scenario: Runtime directory cannot be created
- **WHEN** an `eq` command requires a runtime directory and directory creation fails
- **THEN** the CLI SHALL exit nonzero
- **AND** the error SHALL identify the failed path

#### Scenario: Mode parsing is invalid
- **WHEN** CLI mode parsing receives an invalid mode
- **THEN** the CLI SHALL exit nonzero
- **AND** it SHALL NOT default to `AUTO`

#### Scenario: Import-time environment setup fails
- **WHEN** optional import-time environment setup cannot complete
- **THEN** the package SHALL NOT silently suppress the failure for commands that require that setup

### Requirement: CLI startup does not globally mutate MPS fallback
The `eq` CLI SHALL NOT set `PYTORCH_ENABLE_MPS_FALLBACK=1` globally for all Darwin commands.

#### Scenario: Non-training command runs on Darwin
- **WHEN** a non-training `eq` command starts on Darwin
- **THEN** CLI startup SHALL NOT set `PYTORCH_ENABLE_MPS_FALLBACK=1`
- **AND** MPS fallback remains an explicit training or validation command concern
