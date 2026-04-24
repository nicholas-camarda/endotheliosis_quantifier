## ADDED Requirements

### Requirement: Candidate comparison uses a canonical workflow-config control surface
The repository SHALL treat a dedicated candidate-comparison workflow config run through `eq run-config` as the authoritative top-level control surface for glomeruli promotion provenance. The underlying training and comparison commands SHALL remain worker surfaces whose exact invocations are recorded in the workflow provenance, but they SHALL NOT compete as separate orchestration contracts.

#### Scenario: Candidate comparison is configured
- **WHEN** transfer and no-mitochondria-base candidate runs are defined for promotion comparison
- **THEN** the authoritative recipe is expressed through a dedicated workflow config with `workflow: segmentation_candidate_comparison`
- **AND** the workflow provenance records the exact underlying training and comparison commands that were launched

#### Scenario: Candidate comparison output location is not supplied
- **WHEN** the candidate-comparison workflow is executed without an explicit output override
- **THEN** it SHALL write promotion reports, deterministic manifests, metrics, and review assets under the active runtime output root's `output/segmentation_evaluation/glomeruli_candidate_comparison/` subtree
- **AND** it MAY still accept an explicit caller-supplied override path when the user intentionally wants a different destination

#### Scenario: Candidate comparison trains model artifacts
- **WHEN** the candidate-comparison workflow trains transfer or no-base candidates
- **THEN** trained candidate model artifacts SHALL be written under the configured model root's glomeruli segmentation subtrees
- **AND** the comparison output tree SHALL reference those artifacts rather than duplicating them under the evaluation report directory

## REMOVED Requirements

### Requirement: Candidate comparison uses a canonical CLI-first control surface
**Reason**: The repository-level reproducible contract is moving to dedicated stage-specific workflow configs under `eq run-config` so users do not have to reconstruct multi-step promotion runs manually from worker commands.
**Migration**: Run candidate comparison through the dedicated workflow config surface instead of treating the training-module CLI alone as the top-level orchestration contract.
