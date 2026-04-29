# fastai-training-hygiene Specification

## Purpose
TBD - created by archiving change p1-align-fastai-practices-and-archive-history. Update Purpose after archive.
## Requirements
### Requirement: FastAI training modules use explicit imports
Active FastAI training modules SHALL import the FastAI symbols they use explicitly instead of relying on wildcard imports.

#### Scenario: Glomeruli training imports are lint-visible
- **WHEN** `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m ruff check src/eq/training/train_glomeruli.py` is run
- **THEN** the command passes without `F405` undefined-name findings for `Learner`, `unet_learner`, `resnet34`, `Dice`, `JaccardCoeff`, or `load_learner`

#### Scenario: Mitochondria training imports are lint-visible
- **WHEN** `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m ruff check src/eq/training/train_mitochondria.py` is run
- **THEN** the command passes without `F405` undefined-name findings for FastAI learner, architecture, metric, callback, or serialization symbols

#### Scenario: Transfer-learning imports are lint-visible
- **WHEN** `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m ruff check src/eq/training/transfer_learning.py` is run
- **THEN** the command passes without `F405` undefined-name findings for FastAI learner, architecture, metric, mixed-precision, or serialization symbols

### Requirement: Required training evidence artifacts fail hard
Training and model-export helpers SHALL treat required evidence artifacts as part of the supported runtime contract rather than optional logging.

#### Scenario: Split manifest write fails
- **WHEN** a supported training run reaches `save_splits(...)` and the split manifest cannot be written
- **THEN** the training run fails with an exception that identifies the failed split-manifest path
- **AND** the run is not reported as a complete supported artifact

#### Scenario: Training history write fails
- **WHEN** a supported training run reaches `save_training_history(...)` and the training history cannot be written
- **THEN** the training run fails with an exception that identifies the failed history path
- **AND** the exported artifact is not reported as provenance-complete

#### Scenario: Final model export metadata write fails
- **WHEN** `export_final_model(...)` or `save_run_metadata(...)` cannot write the required model or metadata artifact
- **THEN** the training run fails before returning a supported model path

#### Scenario: Optional visualization fails
- **WHEN** an optional plot or review-panel image fails after required training evidence artifacts are already written
- **THEN** the failure is recorded as a warning with the artifact path and exception
- **AND** the warning does not alter training semantics, metrics, model weights, or artifact promotion status

### Requirement: Trusted current-namespace FastAI learner loading is explicit
FastAI learner loading SHALL be limited to trusted, current-namespace supported artifacts unless a separate compatibility change explicitly adds another policy.

#### Scenario: Current supported artifact is loaded
- **WHEN** `load_model_safely(...)`, `load_glomeruli_model(...)`, `load_mitochondria_model(...)`, or transfer-learning base loading receives a current-namespace artifact that exists and loads under `eq-mac`
- **THEN** the loader returns the FastAI learner and records runtime compatibility only
- **AND** the result is not treated as scientific model promotion

#### Scenario: Legacy artifact requires removed namespace support
- **WHEN** a FastAI pickle artifact requires removed project modules, old FastAI transform namespaces, incompatible NumPy pickle namespaces, or legacy namespace shims
- **THEN** the active loader fails closed or records the artifact as unavailable compatibility evidence
- **AND** the implementation does not add imports or monkey patches to rescue that artifact

#### Scenario: Active code contains no historical workarounds
- **WHEN** active source under `src/eq/` is inspected after implementation
- **THEN** supported workflow code does not contain historical FastAI helper imports, legacy namespace shims, silent fallback branches, or workaround paths for unsupported artifacts
- **AND** any retained historical explanation lives in docs archive/reference material rather than active runtime code

#### Scenario: Trusted pickle limitation is documented
- **WHEN** current docs describe FastAI `load_learner`
- **THEN** they state that learner pickle loading is for trusted current-namespace repository artifacts
- **AND** they direct unsupported legacy artifacts to historical/reference documentation rather than current operational commands

### Requirement: FastAI hygiene validation is a green gate
The FastAI hygiene cleanup SHALL NOT be considered complete until lint, tests, CLI import, and OpenSpec validation are green in the `eq-mac` runtime.

#### Scenario: Final validation runs
- **WHEN** implementation for this change is complete
- **THEN** validation includes `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m ruff check .`
- **AND** validation includes `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest -q`
- **AND** validation includes `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m eq --help`
- **AND** validation includes `OPENSPEC_TELEMETRY=0 openspec validate p1-align-fastai-practices-and-archive-history --strict`

