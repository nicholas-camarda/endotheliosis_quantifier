# workflow-config-entrypoints Specification

## Purpose
TBD - created by archiving change p0-negative-background-glomeruli-supervision. Update Purpose after archive.
## Requirements
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

### Requirement: Workflow-config families are stage-specific
The repository SHALL expose separate workflow-config families for glomeruli candidate comparison, standard external-cohort segmentation transport audit, high-resolution concordance on large-field microscope images, and downstream endotheliosis quantification. Each workflow config SHALL correspond to exactly one stage with one scientific meaning, and SHALL NOT silently execute a different stage as part of the same run.

#### Scenario: Candidate-comparison workflow is executed
- **WHEN** `eq run-config` executes a config with `workflow: glomeruli_candidate_comparison`
- **THEN** the run refreshes only the prerequisite manifest or training state needed for candidate comparison, trains or loads the candidate families needed for promotion evidence, and writes promotion artifacts under the segmentation-evaluation and model roots
- **AND** it SHALL NOT launch external-cohort transport audit, MR concordance grading, or downstream endotheliosis quantification as part of that same workflow

#### Scenario: Transport-audit workflow is executed
- **WHEN** `eq run-config` executes a config with `workflow: glomeruli_transport_audit`
- **THEN** the run evaluates an explicit supported segmentation artifact on the specified cohort or manifest slice, writes evaluation artifacts under `output/segmentation_evaluation/`, and writes any reusable prediction assets under `output/predictions/`
- **AND** it SHALL NOT train candidate-comparison artifacts, run the high-resolution concordance workflow, or launch ordinal quantification as part of that same workflow

#### Scenario: High-resolution concordance workflow is executed
- **WHEN** `eq run-config` executes a config with a dedicated high-resolution concordance workflow `workflow: highres_glomeruli_concordance`
- **THEN** the run accepts explicit large-field image inputs, tiling or preprocessing parameters, and a supported segmentation artifact reference
- **AND** it writes concordance and transport-evaluation artifacts under `output/segmentation_evaluation/`
- **AND** it SHALL NOT be treated as a standard transport-audit workflow or as downstream quantification

#### Scenario: Quantification workflow is executed
- **WHEN** `eq run-config` executes a config with `workflow: endotheliosis_quantification`
- **THEN** the run executes the contract-first quantification pipeline against explicit accepted ROI or segmentation inputs and writes outputs under `output/quantification_results/`
- **AND** it SHALL NOT retrain segmentation candidates, run cohort-transport audit, or run the high-resolution concordance workflow as an implicit side effect

### Requirement: Workflow handoffs are explicit and fail closed
Downstream workflow configs SHALL consume explicit upstream artifact references and SHALL fail closed when those references are missing, unsupported, or scientifically blocked. The repository SHALL NOT auto-discover a latest artifact, retrain a missing segmentation model, or silently skip a required gate in order to complete a downstream run.

#### Scenario: Transport audit is missing a segmentation artifact
- **WHEN** a transport-audit workflow is started without an explicit supported segmentation artifact reference
- **THEN** the run fails before inference begins
- **AND** it SHALL NOT train or promote a replacement segmentation model during that transport-audit run

#### Scenario: Quantification is missing accepted upstream inputs
- **WHEN** a quantification workflow is started without the explicit segmentation artifact or accepted predicted-ROI input surface required by that config
- **THEN** the run fails before ROI extraction or ordinal-model execution begins
- **AND** it SHALL NOT launch segmentation candidate comparison or transport audit to fill the gap automatically

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

