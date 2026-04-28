## ADDED Requirements

### Requirement: Workflow configs use the repo-wide execution logging contract
Workflow config execution through `eq run-config` SHALL capture the same function-level logging events available to direct execution, while preserving the `logs/run_config` durable log contract.

#### Scenario: Run-config captures workflow events
- **WHEN** `eq run-config` executes any supported workflow config
- **THEN** it writes a durable parent log under `$EQ_RUNTIME_ROOT/logs/run_config/<run_id>/<timestamp>.log`
- **AND** the parent log captures logger events emitted by the workflow runner and supported high-level functions
- **AND** workflow-specific manual log-handle plumbing is not maintained as a second durable logging system

#### Scenario: Run-config dry-run records planned execution
- **WHEN** `eq run-config --config <config> --dry-run` executes a supported workflow config, including `configs/mito_pretraining_config.yaml`, `configs/glomeruli_finetuning_config.yaml`, `configs/glomeruli_candidate_comparison.yaml`, `configs/glomeruli_transport_audit.yaml`, `configs/highres_glomeruli_concordance.yaml`, or `configs/endotheliosis_quantification.yaml`
- **THEN** the log records the workflow ID, run ID, config path, runtime root, dry-run status, planned commands or direct workflow action, and log path
- **AND** it does not require private data, model artifacts, or long-running training to validate the logging contract

#### Scenario: Run-config subprocess output is tee-captured
- **WHEN** a run-config workflow launches subprocess worker commands
- **THEN** worker stdout and stderr are tee-captured into the run-config log by the shared execution logging helper
- **AND** the workflow still fails closed on nonzero worker return codes

## MODIFIED Requirements

### Requirement: Retired mixed workflow names are rejected
The repository SHALL retire mixed-purpose workflow names that obscure stage boundaries. Stale workflow identifiers, config names, and runner-module names SHALL NOT remain as supported public orchestration surfaces once the split is implemented.

#### Scenario: Retired mixed workflow identifier is used
- **WHEN** a caller attempts to run a config with the retired mixed workflow identifier `segmentation_fixedloader_full_retrain`
- **THEN** `eq run-config` fails with an error that identifies the supported split workflow families instead of dispatching the old mixed behavior

#### Scenario: Committed workflow configs are inspected
- **WHEN** the committed repository workflow configs are inspected after this change
- **THEN** they use stage-specific names that describe mitochondria pretraining, glomeruli transfer fine-tuning, candidate comparison, transport audit, high-resolution concordance, or quantification directly
- **AND** the repository does not present `fixedloader_full` as a supported current workflow name
- **AND** the committed exact workflow IDs are `segmentation_mitochondria_pretraining`, `segmentation_glomeruli_transfer`, `glomeruli_candidate_comparison`, `glomeruli_transport_audit`, `highres_glomeruli_concordance`, and `endotheliosis_quantification`
