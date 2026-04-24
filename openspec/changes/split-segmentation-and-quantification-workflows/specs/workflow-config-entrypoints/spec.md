## ADDED Requirements

### Requirement: Workflow-config families are stage-specific
The repository SHALL expose separate workflow-config families for glomeruli candidate comparison, external-cohort segmentation transport audit, and downstream endotheliosis quantification. Each workflow config SHALL correspond to exactly one stage with one scientific meaning, and SHALL NOT silently execute a different stage as part of the same run.

#### Scenario: Candidate-comparison workflow is executed
- **WHEN** `eq run-config` executes a config with `workflow: segmentation_candidate_comparison`
- **THEN** the run refreshes only the prerequisite manifest or training state needed for candidate comparison, trains or loads the candidate families needed for promotion evidence, and writes promotion artifacts under the segmentation-evaluation and model roots
- **AND** it SHALL NOT launch external-cohort transport audit, MR concordance grading, or downstream endotheliosis quantification as part of that same workflow

#### Scenario: Transport-audit workflow is executed
- **WHEN** `eq run-config` executes a config with `workflow: segmentation_transport_audit`
- **THEN** the run evaluates an explicit supported segmentation artifact on the specified cohort or manifest slice, writes evaluation artifacts under `output/segmentation_evaluation/`, and writes any reusable prediction assets under `output/predictions/`
- **AND** it SHALL NOT train candidate-comparison artifacts or launch ordinal quantification as part of that same workflow

#### Scenario: Quantification workflow is executed
- **WHEN** `eq run-config` executes a config with `workflow: endotheliosis_quantification`
- **THEN** the run executes the contract-first quantification pipeline against explicit accepted ROI or segmentation inputs and writes outputs under `output/quantification_results/`
- **AND** it SHALL NOT retrain segmentation candidates or run cohort-transport audit as an implicit side effect

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
- **THEN** they use stage-specific names that describe candidate comparison, transport audit, or quantification directly
- **AND** the repository does not present `fixedloader_full` as a supported current workflow name
