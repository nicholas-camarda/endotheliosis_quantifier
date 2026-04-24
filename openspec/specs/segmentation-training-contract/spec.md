# segmentation-training-contract Specification

## Purpose
Define the supported full-image dynamic-patching training contract, artifact provenance requirements, and promotion-related training constraints for segmentation models in this repository.
## Requirements
### Requirement: Segmentation training uses full-image data roots
Supported segmentation training SHALL use data roots that contain full-image `images/` and corresponding `masks/` directories.

#### Scenario: Mitochondria training receives a full-image root
- **WHEN** mitochondria training is started with a data root containing `images/` and `masks/`
- **THEN** training builds dataloaders from the full-image pairs using dynamic patching

#### Scenario: Glomeruli training receives a full-image root
- **WHEN** glomeruli training is started with a data root containing `images/` and `masks/`
- **THEN** training builds dataloaders from the full-image pairs using dynamic patching

#### Scenario: Required full-image directories are missing
- **WHEN** supported segmentation training is started with a data root that does not contain both `images/` and `masks/`
- **THEN** training fails before model construction with an error that identifies the missing full-image training contract

### Requirement: Curated trainable image-mask roots use the raw-data contract
When segmentation training consumes paired image/mask files directly or through the unified admitted-mask cohort registry, those trainable roots SHALL be treated as `raw_data` assets rather than `derived_data` artifacts.

#### Scenario: Glomeruli all-data training root is defined
- **WHEN** glomeruli training should use all currently admitted masked rows
- **THEN** the canonical root is the manifest-backed `raw_data/cohorts` registry root
- **AND** the loader enumerates admitted `manual_mask` and `masked_external` rows with runtime-local image and mask paths

#### Scenario: Glomeruli project-only paired root is defined
- **WHEN** glomeruli full-image pairs are curated for a project-only training run
- **THEN** the supported root lives under `raw_data/...` and contains direct paired `images/` and `masks/`, such as `raw_data/preeclampsia_project/data`
- **AND** that root contains only trainable paired images and masks

#### Scenario: Raw backup source tree contains unpaired files
- **WHEN** a raw backup tree such as `clean_backup` contains images without matching masks
- **THEN** it is treated as source material to curate or localize from, not as the canonical supported training root

#### Scenario: Derived data contract is inspected
- **WHEN** manifests, audits, caches, metrics, or evaluation artifacts are produced from segmentation training data
- **THEN** those generated outputs live under `derived_data/...`
- **AND** they are not presented as the canonical trainable glomeruli image/mask root

### Requirement: Physical installed splits remain distinct from dynamic train/validation splits
Supported segmentation training SHALL preserve existing physical full-image `training/` and `testing/` layouts, while dynamic patching creates the internal train/validation split from the selected training root.

#### Scenario: Mitochondria installed layout is preserved
- **WHEN** `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/derived_data/mitochondria_data` is inspected after static patch retirement
- **THEN** `training/images`, `training/masks`, `testing/images`, and `testing/masks` remain active full-image directories
- **AND** those directories are not merged, renamed, or flattened by this change

#### Scenario: Mitochondria dynamic training uses the training root only
- **WHEN** mitochondria training is started with `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/derived_data/mitochondria_data/training` as the data root
- **THEN** dynamic patching creates the internal train/validation split from that selected `training/` root
- **AND** the sibling `testing/` root is not silently included in training or validation

#### Scenario: Mitochondria held-out testing is deliberate
- **WHEN** mitochondria held-out evaluation is run
- **THEN** it uses `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/derived_data/mitochondria_data/testing` only through an explicit evaluation path
- **AND** held-out testing examples do not affect training-time validation metrics

#### Scenario: Glomeruli curated paired root splits internally
- **WHEN** glomeruli training uses a curated paired full-image root under `raw_data`
- **THEN** dynamic patching may create an internal train/validation split from that selected root
- **AND** this does not imply that retired static `training/`, `testing/`, or `prediction/` patch directories are active model-training inputs

#### Scenario: Mitochondria installed runtime remains an exception
- **WHEN** the current Lucchi-derived mitochondria runtime is inspected
- **THEN** its installed full-image `training/` and `testing/` roots may remain under `derived_data/mitochondria_data`
- **AND** that installed runtime exception does not redefine the generic raw-data contract for curated glomeruli training pairs

### Requirement: Static patch roots are not supported training inputs
Supported segmentation training SHALL NOT accept pre-generated `image_patches/` and `mask_patches/` directories as training inputs.

#### Scenario: Static patch root is supplied to mitochondria training
- **WHEN** mitochondria training is started with a data root whose active training inputs are `image_patches/` and `mask_patches/`
- **THEN** training fails before model construction and instructs the user to use the full-image `images/` and `masks/` root

#### Scenario: Static patch root is supplied to glomeruli training
- **WHEN** glomeruli training is started with a data root whose active training inputs are `image_patches/` and `mask_patches/`
- **THEN** training fails before model construction and instructs the user to use the full-image `images/` and `masks/` root

#### Scenario: Static patch utilities are used outside supported training
- **WHEN** a legacy audit, conversion, or historical-inspection workflow uses static patch utilities
- **THEN** that workflow is explicitly labeled legacy or audit-only and is not presented as a supported model-training path

### Requirement: Dynamic patching is the only supported segmentation training mode
The segmentation training CLIs and configs SHALL NOT expose a supported option to disable dynamic patching.

#### Scenario: Training CLI help is displayed
- **WHEN** the mitochondria or glomeruli training CLI help is displayed
- **THEN** the help does not present `--no-dynamic-patching` or an equivalent supported static-patch training option

#### Scenario: Config examples are inspected
- **WHEN** segmentation training config files are inspected
- **THEN** they point to full-image roots and do not configure `image_patches/` or `mask_patches/` as model-training inputs

#### Scenario: Transfer learning creates glomeruli dataloaders
- **WHEN** glomeruli transfer learning builds target dataloaders
- **THEN** it uses full-image dynamic patching and does not silently fall back to static patch dataloaders

### Requirement: Powerful Apple Silicon MPS uses higher starting batch defaults
On powerful Apple Silicon systems using MPS, local segmentation training SHALL use machine-aware starting batch defaults while keeping explicit overrides available.

#### Scenario: Mitochondria local training starts on a powerful Apple Silicon MPS machine
- **WHEN** mitochondria training starts with `256x256` crops on the certified powerful Apple Silicon MPS machine class
- **THEN** the default starting batch size is `24`
- **AND** CLI or config overrides may still change the batch size

#### Scenario: Glomeruli local training starts on a powerful Apple Silicon MPS machine
- **WHEN** glomeruli training starts with `512x512` crops on the certified powerful Apple Silicon MPS machine class
- **THEN** the default starting batch size is `12`
- **AND** CLI or config overrides may still change the batch size

#### Scenario: Throughput or stability requires a different batch
- **WHEN** MPS throughput, memory pressure, or stability indicates the machine-aware starting batch should change
- **THEN** the operator may override the batch size without changing the underlying training contract

### Requirement: Static patch datasets are retired from active runtime training locations
Active ProjectsRuntime segmentation training directories SHALL NOT contain static patch directories for supported training.

#### Scenario: Glomeruli runtime data is inspected
- **WHEN** `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/derived_data/glomeruli_data` is inspected after this change
- **THEN** active `training/`, `testing/`, and `prediction/` directories do not contain `image_patches/` or `mask_patches/`

#### Scenario: Mitochondria runtime data is inspected
- **WHEN** `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/derived_data/mitochondria_data` is inspected after this change
- **THEN** active `training/` and `testing/` directories do not contain `image_patches/`, `mask_patches/`, `image_patch_validation/`, or `mask_patch_validation/`

#### Scenario: Retired static patch data is preserved
- **WHEN** static patch directories are retired from active runtime locations
- **THEN** they are moved to a dated directory under `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/_retired/` and the implementation records moved paths, size, and file count

### Requirement: Supported segmentation artifacts record training provenance
Supported segmentation model artifacts SHALL record enough provenance to determine whether they were trained with the current full-image dynamic-patching contract.

#### Scenario: Supported model artifact is exported
- **WHEN** a supported segmentation model artifact is exported
- **THEN** its sidecar metadata records the training command, code version, package versions, data root, training mode, and output artifact paths

#### Scenario: Static-patch-trained artifact is encountered
- **WHEN** a segmentation artifact provenance indicates static patch training or lacks training-mode provenance
- **THEN** it is treated as historical or compatibility-only until retrained or re-exported through the supported full-image dynamic-patching contract

#### Scenario: Legacy FastAI pickle artifact is encountered
- **WHEN** a segmentation artifact requires removed project modules, old FastAI transform namespaces, incompatible NumPy pickle namespaces, or legacy namespace shims to load
- **THEN** it is treated as a historical artifact unless a separate compatibility change explicitly supports it

### Requirement: Scientific promotion remains separate from runtime support
Runtime support SHALL NOT be treated as scientific model promotion.

#### Scenario: Current-namespace artifact loads and runs
- **WHEN** a current-namespace segmentation artifact loads and completes a pipeline or training smoke check
- **THEN** that result establishes runtime compatibility only, not scientific promotion

#### Scenario: Glomeruli model is proposed for promotion
- **WHEN** a glomeruli segmentation artifact is proposed as a promoted scientific model
- **THEN** the promotion evidence includes training-data foreground/background coverage, validation metrics against trivial all-foreground and all-background baselines, and non-degenerate prediction review artifacts

#### Scenario: Dynamic training completes
- **WHEN** segmentation training completes with dynamic patching
- **THEN** completion alone does not promote the artifact without the required model-quality evidence

### Requirement: Glomeruli training-data audit is required for promotion
Glomeruli model promotion SHALL include an audit of the training and validation data used to produce the artifact.

#### Scenario: Glomeruli training data is audited
- **WHEN** a glomeruli segmentation artifact is proposed for scientific promotion
- **THEN** the audit reports foreground fraction distribution, background-only crop rate, full-foreground crop rate, and subject or image split coverage for the training and validation data

#### Scenario: Positive-only static patch data is detected
- **WHEN** a glomeruli training or validation set contains no background-only examples and rewards all-foreground predictions
- **THEN** the resulting artifact is blocked from promotion regardless of training completion or current-namespace loadability

#### Scenario: Runtime evidence from the 2026-04-22 static patch artifact is reviewed
- **WHEN** the static-patch-trained glomeruli compatibility artifact is used as comparison evidence
- **THEN** its metrics are compared against the all-foreground baseline and treated as evidence of the retired static patch failure mode, not as promotion evidence

### Requirement: Glomeruli promotion validation uses deterministic examples
Glomeruli model promotion SHALL use deterministic validation examples rather than relying only on stochastic dynamic validation crops.

#### Scenario: Fixed validation manifest is created
- **WHEN** a glomeruli model is evaluated for promotion
- **THEN** validation uses a fixed manifest or equivalent deterministic selection of positive, boundary, and background examples

#### Scenario: Promotion validation is rerun
- **WHEN** the same promoted-candidate artifact is evaluated multiple times against the fixed validation manifest
- **THEN** the selected validation examples and trivial-baseline comparisons remain stable across runs

#### Scenario: Dynamic training uses random crops
- **WHEN** dynamic patching uses stochastic crops during training
- **THEN** the training sampler may remain stochastic, but promotion validation still uses fixed validation examples

### Requirement: Glomeruli promotion compares against trivial baselines
Glomeruli model promotion SHALL compare candidate predictions against trivial all-background and all-foreground baselines.

#### Scenario: Candidate metrics are computed
- **WHEN** glomeruli promotion metrics are reported
- **THEN** Dice and Jaccard are reported for the candidate model, an all-background baseline, an all-foreground baseline, and the current compatibility artifact when available

#### Scenario: Candidate fails to beat trivial baselines
- **WHEN** a candidate model does not beat the relevant trivial baselines on the fixed validation examples
- **THEN** the candidate is blocked from scientific promotion

#### Scenario: Candidate only matches all-foreground behavior
- **WHEN** candidate Dice or Jaccard is close to the all-foreground baseline and prediction review shows broad foreground masks
- **THEN** the candidate is treated as degenerate and blocked from promotion

### Requirement: Glomeruli promotion includes prediction review artifacts
Glomeruli model promotion SHALL include quantitative and visual prediction-review artifacts.

#### Scenario: Prediction review is generated
- **WHEN** a glomeruli model is evaluated for promotion
- **THEN** review artifacts include sampled validation panels, prediction foreground fraction summaries, and evidence that predictions are not all foreground or all background

#### Scenario: Degenerate prediction review is detected
- **WHEN** prediction review shows all-foreground or all-background behavior on representative validation examples
- **THEN** the candidate is blocked from scientific promotion even if aggregate metrics appear acceptable

### Requirement: The updated contract must run end to end
This change SHALL NOT be considered complete until the repository can run the intended pipeline from the beginning under the updated contract and complete end to end.

#### Scenario: End-to-end validation is executed
- **WHEN** final validation for this change is performed
- **THEN** the operator runs the whole pipeline from the beginning using the updated path contract, dynamic training contract, and current local hardware defaults
- **AND** the run covers the real entrypoints needed to prove the updated workflow is executable rather than only isolated smoke tests

#### Scenario: End-to-end run finds a blocker
- **WHEN** the full pipeline fails at any step during final validation
- **THEN** the blocker is treated as a required issue to fix for this change rather than a deferred note
- **AND** validation is rerun from the beginning after the fix until the pipeline completes end to end

#### Scenario: End-to-end iteration is documented
- **WHEN** the full pipeline has been iterated to completion
- **THEN** the implementation records the executed path, every issue discovered during iteration, the fix applied for each issue, and any remaining residual limitations that do not block completion
- **AND** that record lives inside this OpenSpec change's artifacts rather than in separate implementation notes, side documents, or external logs

### Requirement: Glomeruli promotion uses a concrete candidate-comparison workflow
Promoting a glomeruli segmentation model SHALL require a concrete comparison of supported candidate artifacts rather than evaluating a single newly trained artifact in isolation.

#### Scenario: Promotion workflow is executed
- **WHEN** glomeruli promotion is attempted under the supported training contract
- **THEN** the workflow compares at least two supported candidate families under a shared deterministic validation manifest
- **AND** the compared families include mitochondria transfer learning and no-mitochondria-base ImageNet-initialized training unless one family is explicitly unavailable and reported as such

#### Scenario: Promotion workflow control surface is evaluated
- **WHEN** glomeruli candidate comparison is defined or documented
- **THEN** the dedicated training-module CLI is treated as the canonical control surface
- **AND** stale config-first or patch-era surfaces are updated or retired if they conflict with the supported training contract

#### Scenario: Scratch glomeruli candidate requests larger crop context
- **WHEN** the canonical glomeruli training CLI is run with `--from-scratch`, `--image-size 256`, and `--crop-size 512`
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
