# ordinal-quantification-stability Specification

## Purpose
Define the canonical ordinal quantification implementation, numerical-stability gate, and provenance requirements for frozen-embedding image-level endotheliosis prediction.

## Requirements
### Requirement: Quantification uses one canonical ordinal estimator path
The contract-first quantification workflow SHALL use one canonical ordinal estimator implementation for image-level grouped prediction, and duplicate ordinal-model logic SHALL NOT remain as a second supported execution path.

#### Scenario: Quantification pipeline trains the ordinal stage
- **WHEN** the contract-first pipeline reaches the ordinal modeling step
- **THEN** it calls the canonical ordinal estimator implementation in `src/eq/quantification/ordinal.py`
- **AND** no second duplicated ordinal model path remains as an alternative supported runtime behavior

#### Scenario: Shared ordinal evaluation helper is invoked
- **WHEN** repository code performs grouped ordinal evaluation outside the full pipeline
- **THEN** it uses the same canonical estimator path, target encoding, and probability semantics as the pipeline

### Requirement: Grouped ordinal evaluation is numerically stable on the supported cohort shape
The canonical ordinal estimator SHALL satisfy a numerical-stability contract on the supported grouped evaluation workflow for the current embedding-table problem shape.

#### Scenario: Grouped cross-validation runs on the supported embedding table
- **WHEN** grouped subject-level ordinal evaluation runs on the supported embedding table or a faithful regression fixture with the same class-sparsity shape
- **THEN** the estimator completes without overflow, divide-by-zero, or invalid-value warnings from the model-fitting path

#### Scenario: Current threshold family cannot satisfy the stability contract
- **WHEN** the current estimator family cannot meet the grouped numerical-stability requirement under the supported cohort shape
- **THEN** the implementation replaces it with a numerically stable supported estimator rather than suppressing the warnings
- **AND** a strongly regularized penalized multiclass logistic estimator is an approved replacement family for this change

#### Scenario: Available cohort does not represent the full seven-bin target
- **WHEN** grouped evaluation is numerically stable but the available cohort omits one or more target score bins
- **THEN** the pipeline still writes the expected ordinal artifacts
- **AND** the outputs report that target support is incomplete for the current seven-bin contract
- **AND** the workflow does not silently collapse classes inside this change

### Requirement: Ordinal outputs remain schema-stable and provenance-aware
Stabilizing the ordinal estimator SHALL preserve the current quantification output contract unless a separate spec explicitly changes it.

#### Scenario: Ordinal artifacts are written
- **WHEN** the stabilized ordinal workflow writes predictions, metrics, confusion matrices, serialized model artifacts, and review outputs
- **THEN** the existing artifact names and table/report schemas remain available to downstream consumers
- **AND** the saved metadata identifies the canonical estimator used for the run

#### Scenario: Stability-related metadata is inspected
- **WHEN** a developer inspects the stabilized ordinal run outputs
- **THEN** the outputs include enough information to determine the estimator class, grouped split configuration, and any remaining cohort-shape limitations relevant to interpretation
