## ADDED Requirements

### Requirement: Candidate comparison config controls negative/background supervision
The `glomeruli_candidate_comparison` workflow config SHALL expose explicit negative/background supervision controls.

#### Scenario: Config enables mask-derived background supervision
- **WHEN** `configs/glomeruli_candidate_comparison.yaml` sets `negative_background_supervision.mask_derived_background.enabled=true`
- **THEN** the workflow generates or resolves a mask-derived background manifest before candidate training
- **AND** the manifest is passed to both transfer and scratch/no-base candidate training unless a candidate explicitly opts out

#### Scenario: Config enables curated negative manifest
- **WHEN** `negative_background_supervision.curated_negative_manifest.enabled=true`
- **THEN** `manifest_path` is required
- **AND** the workflow fails before training if the manifest is missing or invalid

#### Scenario: Config declares augmentation audit
- **WHEN** `augmentation_audit` is present in the workflow config
- **THEN** dry-run output and training provenance identify the selected augmentation variant
- **AND** unsupported augmentation names are rejected rather than ignored
