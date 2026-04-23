# macos-mps-local-development Specification

## Purpose
TBD - created by archiving change macos-mps-local-dev. Update Purpose after archive.
## Requirements
### Requirement: Apple Silicon environment definition
The repository SHALL provide a macOS Apple Silicon environment definition for local development without changing the existing CUDA-oriented environment definition.

#### Scenario: Mac environment file is available
- **WHEN** a developer prepares local development on macOS arm64
- **THEN** the repository provides `environment-macos.yml` with the verified macOS PyTorch, fastai, and scientific Python stack

#### Scenario: CUDA environment remains available
- **WHEN** a developer prepares the WSL/CUDA environment
- **THEN** the existing `environment.yml` remains the CUDA-oriented setup path

### Requirement: Environment certification precedes backend changes
Implementation SHALL certify the `eq-mac` environment before changing direct inference backend behavior.

#### Scenario: Environment package set is inspected
- **WHEN** implementation begins
- **THEN** the active `eq-mac` Python, PyTorch, torchvision, fastai, scientific Python, and image-processing package versions are inspected and compared with the intended macOS environment definition

#### Scenario: Environment smoke checks pass
- **WHEN** implementation begins
- **THEN** `eq-mac` passes import checks, `python -m eq --help`, focused CLI checks, and the existing focused unit tests before backend code changes are treated as valid

#### Scenario: Codex cannot validate real MPS
- **WHEN** Codex reports MPS unavailable or cannot validate Metal access from the sandbox
- **THEN** implementation stops only for that Metal validation point and asks the user to run the exact normal-terminal command needed to verify MPS behavior

### Requirement: Data and runtime layout audit
Implementation SHALL audit the original cloud source, cloud working copy, repository symlinks, and `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier` before treating Mac local development as certified.

#### Scenario: Data lineage is inventoried
- **WHEN** implementation begins runtime certification
- **THEN** the audit records the apparent original OneDrive source, OneDrive SideProjects working copy, repository symlink targets, ProjectsRuntime working tree, approximate sizes, and their intended roles

#### Scenario: Directory status report is presented
- **WHEN** the data/runtime audit is complete
- **THEN** implementation reports each relevant directory's path, role, size, notable contents, symlink status, code/config/test references, recommended action, and risk if changed

#### Scenario: Original source data is protected
- **WHEN** the audit identifies original cloud data such as the Lauren PreEclampsia source directory
- **THEN** it treats that source as read-only provenance and does not recommend mutation as part of Mac local-development setup

#### Scenario: Runtime data roles are clarified
- **WHEN** the audit encounters `data/`, `raw_data/`, and `derived_data/` under working-copy or runtime roots
- **THEN** it determines whether each `data/` subtree is legacy, an alias-like staging area, or a distinct supported runtime contract before recommending path changes

#### Scenario: Empty nested directories are classified
- **WHEN** the audit finds empty nested directories under `output/`, `models/`, `derived_data/`, or `data/`
- **THEN** each directory is classified as keep, remove candidate, migrate candidate, or unknown based on code/config/test references and artifact ownership

#### Scenario: Runtime cleanup requires approval
- **WHEN** an optimization would delete, move, rename, or rewrite a runtime directory outside the Git checkout
- **THEN** implementation presents the exact proposed operation and waits for explicit user approval before mutating the runtime filesystem

#### Scenario: Canonical data/runtime contract is documented in code-facing terms
- **WHEN** the audit is complete
- **THEN** the implementation identifies the intended roles of original source data, raw working data, derived data, models, logs, and output roots using the repository's path helper terminology

#### Scenario: Contract answers storage ownership questions
- **WHEN** the data/runtime contract is drafted
- **THEN** it answers where immutable original data lives, where curated raw working projects live, where derived patch datasets live, where active training outputs and checkpoints live, which model artifacts are promoted or backed up, which outputs are disposable versus publishable, and whether repo symlinks should point to ProjectsRuntime, OneDrive, or be removed

#### Scenario: Contract approval gates path changes
- **WHEN** implementation proposes cleanup, symlink changes, path-helper default changes, or runtime directory moves
- **THEN** those changes wait until the data/runtime contract has been presented and approved

#### Scenario: Directory contract approval gates implementation
- **WHEN** implementation has completed environment inspection and non-mutating validation
- **THEN** it presents the directory status report and proposed data/runtime contract to the user
- **AND** waits for user approval before implementing environment files, backend code changes, pipeline tests, cleanup, symlink changes, path-helper default changes, or runtime directory moves that depend on the contract

### Requirement: MPS-aware hardware reporting
The CLI SHALL report MPS as the primary backend when the active Python environment has PyTorch built with MPS and MPS is available, and CUDA is unavailable.

#### Scenario: MPS is available on Apple Silicon
- **WHEN** `eq capabilities` runs in a macOS arm64 environment where PyTorch reports MPS available and CUDA unavailable
- **THEN** the backend availability section reports MPS available and the primary backend is MPS

#### Scenario: MPS is not available
- **WHEN** `eq capabilities` runs where PyTorch reports MPS unavailable and CUDA unavailable
- **THEN** the primary backend is CPU

### Requirement: Direct inference auto-selects available backends
Direct glomeruli inference SHALL select the best available backend in priority order CUDA, then MPS, then CPU when initialized with `device='auto'`.

#### Scenario: CUDA is available
- **WHEN** direct inference is initialized with `device='auto'` and CUDA is available
- **THEN** the selected device is CUDA

#### Scenario: MPS is available without CUDA
- **WHEN** direct inference is initialized with `device='auto'`, CUDA is unavailable, and MPS is available
- **THEN** the selected device is MPS

#### Scenario: No accelerator is available
- **WHEN** direct inference is initialized with `device='auto'` and neither CUDA nor MPS is available
- **THEN** the selected device is CPU

### Requirement: Backend-specific runtime calls
Inference benchmarking and logging SHALL only call CUDA runtime APIs when the selected backend is CUDA.

#### Scenario: Benchmarking on MPS
- **WHEN** inference benchmarking runs with the selected device set to MPS
- **THEN** the benchmark completes without calling CUDA synchronization, CUDA device-name, CUDA device-properties, or CUDA memory APIs

#### Scenario: Benchmarking on CPU
- **WHEN** inference benchmarking runs with the selected device set to CPU
- **THEN** the benchmark completes without calling CUDA synchronization, CUDA device-name, CUDA device-properties, or CUDA memory APIs

### Requirement: MPS training smoke validation
Mac local-development certification SHALL include a repository-owned bounded training smoke test that exercises model construction, data loading, forward/backward execution, and one optimizer step on the selected Mac backend.

#### Scenario: Synthetic MPS training smoke passes
- **WHEN** MPS is available from a normal macOS terminal
- **THEN** a minimal segmentation training smoke using synthetic image/mask fixtures completes on MPS without requiring private data, large artifacts, or a long training run

#### Scenario: Synthetic CPU training smoke passes when MPS is unavailable
- **WHEN** MPS is unavailable in the execution environment
- **THEN** the same bounded training smoke completes on CPU and reports that true MPS validation still requires a normal-terminal check

### Requirement: Contract-first pipeline integration test
Mac local-development certification SHALL include a repository-owned local-runtime integration test that exercises the supported contract-first quantification pipeline using the existing ProjectsRuntime preeclampsia data and a current-namespace ProjectsRuntime glomeruli segmentation model artifact.

#### Scenario: Existing runtime assets are present
- **WHEN** the local-runtime pipeline test begins
- **THEN** it verifies that the existing raw project, Label Studio annotation export, and current glomeruli segmentation model exist at the configured ProjectsRuntime paths

#### Scenario: Existing contract baseline is validated
- **WHEN** the local-runtime pipeline test runs the contract stage
- **THEN** it verifies that the existing runtime assets produce nonzero raw inventory rows, joined Label Studio scores, and scored examples, including the current 88 joined scored examples unless the audit intentionally updates that baseline

#### Scenario: Existing full quantification pipeline passes
- **WHEN** the local-runtime integration test runs with the existing raw project, annotation export, and glomeruli model
- **THEN** `run_contract_first_quantification(..., stop_after='model')` completes and writes contract, score, ROI, embedding, ordinal prediction, metrics, and review artifacts under a temporary output directory

#### Scenario: Pipeline test owns its artifacts
- **WHEN** the integration test runs
- **THEN** generated output artifacts are created under temporary test directories and no repository-tracked files or existing runtime input/model artifacts are changed

#### Scenario: Legacy production runner is not the certification target
- **WHEN** choosing the Mac certification pipeline path
- **THEN** implementation does not use the older `eq production` runner as the primary gate unless that runner is first made current by a separate production-pipeline cleanup

### Requirement: Supported model artifacts use the current namespace
Mac local-development certification SHALL distinguish legacy FastAI pickle artifacts from supported current-namespace segmentation artifacts.

#### Scenario: Current model artifact is required for certification
- **WHEN** the local-runtime pipeline integration test requires a segmentation model artifact
- **THEN** it uses a model exported from the current `src/eq` namespace that loads in the certified environment without legacy module shims

#### Scenario: Legacy FastAI pickle artifact is encountered
- **WHEN** a model artifact references removed project modules such as `eq.segmentation...`, old FastAI transform namespaces, or incompatible NumPy pickle namespaces
- **THEN** the artifact is treated as historical unless a separate compatibility change explicitly supports and tests it

#### Scenario: Model artifact provenance is required
- **WHEN** a segmentation model artifact is proposed as a supported runtime dependency
- **THEN** its provenance records the training command, code version, package versions, data root, and training mode

### Requirement: Runtime compatibility does not imply model promotion
Mac local-development certification SHALL distinguish runtime compatibility evidence from scientific model-promotion evidence.

#### Scenario: Compatibility artifact is used for pipeline certification
- **WHEN** a current-namespace glomeruli model is used to prove that the Mac local-runtime pipeline can load a model and complete through `stop_after='model'`
- **THEN** the artifact is treated as a runtime compatibility artifact unless a separate model-promotion gate passes

#### Scenario: Glomeruli model promotion requires data and prediction evidence
- **WHEN** a glomeruli segmentation model is proposed for scientific use or as the promoted default artifact
- **THEN** implementation documents training-data foreground and background coverage, validation metrics, and prediction-review evidence showing masks are not all foreground or all background

#### Scenario: Degenerate glomeruli validation blocks promotion
- **WHEN** glomeruli validation metrics are flat or validation prediction review shows all-foreground or all-background masks
- **THEN** the model is not promoted, even if it trains, loads, and passes the contract-first pipeline integration test

### Requirement: Research artifacts and contracts remain unchanged
Mac local-development support SHALL NOT change data contracts, path resolution, generated output schemas, or scientific claim semantics.

#### Scenario: Quantification contracts are unaffected
- **WHEN** Mac local-development support is implemented
- **THEN** Label Studio score ingestion, image/mask pairing, union ROI semantics, encoder embedding outputs, ordinal prediction outputs, and review artifacts retain their current schemas and meanings

#### Scenario: Artifact storage boundaries are unaffected
- **WHEN** Mac local-development support is implemented
- **THEN** raw data, derived data, models, logs, notebooks, and output artifacts remain outside Git-tracked source files

### Requirement: Local-runtime quantification certification requires a numerically stable ordinal stage
Mac local-development certification SHALL NOT treat the contract-first quantification pipeline as fully healthy while the ordinal modeling step still emits unresolved numerical-instability warnings on the supported local runtime cohort.

#### Scenario: Local runtime quantification regression is executed
- **WHEN** the supported local-runtime quantification regression runs through the ordinal modeling step
- **THEN** it completes without unresolved overflow, divide-by-zero, or invalid-value warnings from the canonical ordinal estimator path

#### Scenario: Pipeline still completes with ordinal instability
- **WHEN** the local-runtime quantification regression writes all expected artifacts but the ordinal stage still emits unresolved numerical-instability warnings
- **THEN** Mac local-development certification remains incomplete for ordinal quantification health
- **AND** the result is treated as a pipeline-execution success with an unresolved quantification-model defect
