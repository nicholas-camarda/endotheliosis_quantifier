## ADDED Requirements

### Requirement: MPS fallback scope is explicit
Mac MPS fallback SHALL be scoped to explicit segmentation training or validation commands that require it and SHALL NOT be treated as evidence of scientific validity.

#### Scenario: Real Mac training command requires fallback
- **WHEN** a real Mac segmentation training or validation command requires unsupported MPS operations
- **THEN** the command MAY be run with `PYTORCH_ENABLE_MPS_FALLBACK=1`
- **AND** the run metadata or operator notes SHALL preserve the requested training intent and base artifact identity

#### Scenario: Generic CLI command runs on macOS
- **WHEN** a generic `eq` utility, config inspection, or quantification command runs on macOS
- **THEN** the command SHALL NOT rely on global MPS fallback mutation at CLI startup

#### Scenario: MPS execution succeeds
- **WHEN** an MPS-backed command completes
- **THEN** completion SHALL be treated as compatibility evidence only
- **AND** it SHALL NOT promote a segmentation model without the required validation, background/negative coverage, and non-degenerate prediction review evidence
