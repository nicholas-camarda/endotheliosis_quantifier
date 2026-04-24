## MODIFIED Requirements

### Requirement: Glomeruli promotion uses a concrete candidate-comparison workflow
Promoting a glomeruli segmentation model SHALL require a concrete comparison of supported candidate artifacts rather than evaluating a single newly trained artifact in isolation.

#### Scenario: Promotion workflow is executed
- **WHEN** glomeruli promotion is attempted under the supported training contract
- **THEN** the workflow compares at least two supported candidate families under a shared deterministic validation manifest
- **AND** the compared families include mitochondria transfer learning and no-mitochondria-base ImageNet-initialized training unless one family is explicitly unavailable and reported as such

#### Scenario: Promotion workflow control surface is evaluated
- **WHEN** glomeruli candidate comparison is defined or documented
- **THEN** the supported top-level control surface is a dedicated candidate-comparison workflow config executed through `eq run-config`
- **AND** stale mixed workflow names such as `segmentation_fixedloader_full_retrain` and `fixedloader_full` are retired if they conflict with the supported training contract
- **AND** the underlying training-module commands remain recorded in provenance rather than serving as competing orchestration contracts

#### Scenario: Scratch glomeruli candidate requests larger crop context
- **WHEN** the canonical glomeruli training path is run with `--from-scratch`, `--image-size 256`, and `--crop-size 512`
- **THEN** the scratch training path SHALL preserve the requested `512` crop size through batch-size sizing, dynamic patching, and exported provenance
- **AND** it SHALL NOT silently replace the requested crop size with `256`
- **AND** the exported provenance SHALL identify the candidate as the no-mitochondria-base ImageNet-pretrained ResNet34 baseline rather than a literal all-random initialization baseline

#### Scenario: Promotion decision is recorded
- **WHEN** candidate comparison completes
- **THEN** the resulting promotion report records whether one candidate is promoted, no candidate is promoted, or the evidence is insufficient
- **AND** runtime-compatible but non-promoted artifacts remain labeled as non-promoted in provenance and documentation

#### Scenario: Candidate comparison validation is declared complete
- **WHEN** this change is treated as implementation-complete
- **THEN** completion evidence SHALL include an unsandboxed `eq-mac` candidate-comparison validation run where both transfer and scratch execute successfully under the supported runtime contract
- **AND** a report generated only from structured candidate-family failures SHALL NOT be treated as final completion evidence
