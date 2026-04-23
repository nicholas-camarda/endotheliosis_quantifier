## Context

The repository currently has one tracked environment file, `environment.yml`, configured for WSL/CUDA with `pytorch-cuda=12.1`. The code already contains MPS-aware hardware detection in `src/eq/utils/hardware_detection.py`, mode selection in `src/eq/utils/mode_manager.py`, and MPS fallback setup in CLI/model-loading paths. The user's `eq-mac` environment initially verified that PyTorch `2.0.0` could allocate an MPS tensor from a normal macOS terminal on macOS 26.4.1 arm64, but real FastAI DynamicUNet training exposed a hard MPS runtime failure on that stack. The certified local-development stack is now PyTorch `2.10.0`, torchvision `0.26.0`, FastAI `2.8.7`, and NumPy `2.2.6`, which completed FastAI DynamicUNet forward/backward optimization on MPS. The Codex sandbox can still report MPS unavailable and should not be treated as authoritative for Metal validation.

The first implementation gate is to make `eq-mac` itself correct and reproducible. Backend code changes should wait until the active environment has a documented package set, passes import/CLI/unit smoke checks, and has MPS tensor allocation verified from a normal macOS terminal when Codex cannot validate Metal directly.

The main code gap is narrow: `src/eq/inference/gpu_inference.py` still treats automatic GPU inference as CUDA-or-CPU and calls CUDA-only APIs for warmup, synchronization, memory, and logging. The validation gap is broader: CLI/import tests do not prove that MPS can run a real training step or that an existing trained segmentation artifact can drive the supported quantification pipeline. The local-runtime pipeline test should use existing ProjectsRuntime assets rather than manufacturing substitute raw images, masks, annotations, or model files.

The Mac local-development target also depends on distinguishing original data from working copies and runtime artifacts. Initial inspection found:

- Apparent original cloud source: `/Users/ncamarda/Library/CloudStorage/OneDrive-Personal/phd/projects/Lauren PreEclampsia/Lauren_PreEclampsia_Data` at about 3.2 GB, with raw TIF images, JPG training/testing images, and the master grading spreadsheet.
- Cloud SideProjects working copy: `/Users/ncamarda/Library/CloudStorage/OneDrive-Personal/SideProjects/endotheliosis_quantifier` at about 6.1 GB, with project raw/processed data, models, and outputs.
- Local runtime working tree: `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier` at about 6.6 GB, with separate `raw_data/`, `derived_data/`, `models/`, `logs/`, `output/`, and `data/` subtrees.
- Repository symlinks for `raw_data`, `derived_data`, `models`, and `output` currently point to the OneDrive SideProjects working copy, while `logs` points to ProjectsRuntime.

Several nested ProjectsRuntime directories are empty, including `output/mitochondria/{models,results,plots}`, `output/derived/{models,results,plots}`, `output/mitochondria_training/mito_base_model`, `output/glomeruli_finetuning/plots`, `models/regression`, `derived_data/glomeruli_data/prediction/mask_patches`, and `data/preeclampsia_data/train`. The audit must clarify whether `data/` is legacy, an alias, or a distinct runtime contract before any code or filesystem cleanup assumes a canonical layout. The audit must also distinguish immutable source data from working copies and generated artifacts before recommending cleanup.

The data/runtime contract is a required implementation deliverable, not a side note. It must answer where immutable original data lives, where curated raw working projects live, where derived patch datasets live, where active training outputs/checkpoints live, which model artifacts are promoted/backed up, which outputs are disposable versus publishable, and whether repo symlinks should point to ProjectsRuntime, OneDrive, or be removed. The directory status report and proposed contract must be presented to the user before proceeding beyond environment inspection and non-mutating validation. No implementation that depends on those path decisions should proceed until the contract is approved.

The existing runtime assets support a real pipeline gate:

- Raw project root: `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/raw_data/preeclampsia_project/clean_backup`
- Label Studio export: `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/raw_data/preeclampsia_project/annotations/annotations.json`
- Glomeruli model: `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/models/segmentation/glomeruli/transfer/glom_dynamic_unet_mps-transfer_loss-dice_s1lr1e-3_s2lr_lrfind_e30_b16_lr1e-3_sz256/glom_dynamic_unet_mps-transfer_loss-dice_s1lr1e-3_s2lr_lrfind_e30_b16_lr1e-3_sz256.pkl`
- Current contract check: 240 inventory rows, 88 joined Label Studio image/mask pairs, and 88 scored examples with `join_status='ok'`

## Goals / Non-Goals

**Goals:**

- Add a reproducible macOS Apple Silicon environment file for local development.
- Certify that `eq-mac` is clean enough for local development before changing backend behavior.
- Preserve the existing CUDA/WSL environment file and CUDA production path.
- Make direct inference device selection support CUDA, MPS, and CPU through existing backend detection.
- Ensure MPS and CPU execution paths do not call CUDA-only APIs.
- Add a bounded Mac training smoke that exercises a real FastAI/PyTorch optimization step on MPS when MPS access is available.
- Add a local-runtime integration test that uses the existing runtime raw project, annotation export, and glomeruli model, then runs `run_contract_first_quantification(..., stop_after='model')` with output isolated to a temporary directory.
- Audit the current ProjectsRuntime layout and define the intended runtime directory contract for Mac local development.
- Trace the relationship between the OneDrive original source, OneDrive SideProjects working copy, repo symlinks, and ProjectsRuntime working artifacts.
- Produce an explicit directory status report and data/runtime contract, then wait for approval before proceeding beyond environment inspection and non-mutating validation.
- Identify empty, duplicated, legacy, or ambiguous runtime directories and propose an optimization plan before any move/delete operation.
- Add tests that lock the observable backend-selection behavior.

**Non-Goals:**

- Do not replace CUDA production training with MPS.
- Do not make the stale legacy `eq production` runner the Mac certification target unless it is separately brought current.
- Do not create synthetic stand-ins for the raw images, binary masks, Label Studio annotations, or segmentation model when the existing runtime assets are available.
- Do not delete, move, or rewrite runtime artifacts as part of the audit without explicit user approval at implementation time.
- Do not mutate the apparent original OneDrive source data; treat it as read-only unless a separate, explicit data-management change says otherwise.
- Do not change model architecture, segmentation labels, mask semantics, ROI geometry, embeddings, ordinal modeling, or review artifact schemas.
- Do not add silent CPU fallbacks to rescue failed MPS operations.
- Do not modify existing local raw data, derived data, promoted model artifacts, logs, notebooks, or output artifacts. New local training exports may be created under the approved ProjectsRuntime model root for Mac certification.

## Decisions

- Add `environment-macos.yml` rather than modifying `environment.yml`.
  - Rationale: CUDA and Apple Silicon dependency constraints differ, and changing the existing file would risk the current WSL/CUDA setup.
  - Alternative considered: one parameterized environment file; rejected because conda CUDA and macOS arm64 PyTorch resolution are materially different.

- Pin the macOS PyTorch stack to the verified MPS-compatible versions for this machine.
  - Rationale: `torch 2.0.0` was tensor-capable on MPS but could not reliably train FastAI DynamicUNet on the Mac. The verified local-development stack is PyTorch `2.10.0`, torchvision `0.26.0`, FastAI `2.8.7`, and NumPy `2.2.6`.
  - Alternative considered: keep the old PyTorch stack because basic MPS allocation worked; rejected because the certification target must cover real model training, not only tensor allocation.

- Make environment validation an implementation prerequisite.
  - Rationale: dependency conflicts or a partially repaired conda/pip environment can produce misleading backend failures.
  - Alternative considered: implement code changes first and repair the environment later; rejected because it would make failures hard to attribute.

- Centralize auto-device behavior around the existing hardware detection utilities.
  - Rationale: `eq capabilities`, `eq mode`, and tests already model CUDA/MPS/CPU availability.
  - Alternative considered: add inference-local detection; rejected because it would duplicate backend policy and increase drift.

- Keep failures visible when the selected backend cannot execute an operation.
  - Rationale: this is a research pipeline where silent device fallback could change runtime behavior, performance, memory pressure, or numerical execution without being noticed.
  - Alternative considered: catch MPS errors and retry on CPU; rejected as patchwork behavior.

- Validate training and the supported pipeline with existing runtime data and model artifacts.
  - Rationale: it catches the class of MPS/training, model loading, raw data, annotation, ROI, embedding, and ordinal output failures that synthetic fixtures could miss.
  - Alternative considered: create temporary synthetic raw images, masks, annotations, and model artifacts; rejected because the runtime already contains representative project assets and the user prefers those to be the certification source.

- Treat data/runtime layout cleanup as an audit-first step.
  - Rationale: original cloud data, cloud working copies, repo symlinks, and ProjectsRuntime artifacts currently overlap, and deleting empty-looking nested directories without path tracing could break scripts or symlink expectations.
  - Alternative considered: immediately remove empty directories; rejected because runtime cleanup is destructive outside the repo and needs confirmed path ownership.

- Make the directory report and contract approval an early implementation gate.
  - Rationale: storage decisions affect environment assumptions, symlink targets, pipeline-test inputs, output paths, provenance, reproducibility, backups, and disposable runtime artifacts.
  - Alternative considered: continue with environment/code/test implementation while resolving paths opportunistically; rejected because incorrect assumptions could invalidate the Mac certification work or move work toward the wrong data source.

## Risks / Trade-offs

- [Risk] PyTorch/FastAI compatibility on macOS can differ between tensor allocation, synthetic training, and real DynamicUNet training. -> Mitigation: certify the Mac environment against a real FastAI DynamicUNet MPS optimization step and the local-runtime pipeline test, not only package imports.
- [Risk] MPS support can differ between Codex sandbox and the user's terminal. -> Mitigation: keep automated tests focused on selection logic and use normal-terminal MPS execution for real training validation.
- [Risk] Codex may be blocked from validating real MPS or mutating the conda environment. -> Mitigation: stop and ask the user to run the exact terminal command needed, then record the observed result before proceeding.
- [Risk] FastAI or PyTorch operations used in training may still expose MPS-specific unsupported operations. -> Mitigation: set `PYTORCH_ENABLE_MPS_FALLBACK=1` before model execution, but do not hide hard failures.
- [Risk] The older `eq production` runner has pre-existing references to legacy symbols. -> Mitigation: keep this change focused on Mac environment setup and direct backend selection; handle production-runner cleanup separately if needed.
- [Risk] A local-runtime integration test is machine-specific and may not be CI-portable. -> Mitigation: mark it as a Mac/local-runtime certification test, keep outputs in temporary directories, and require it to run non-skipped as part of this Mac certification on this machine.
- [Risk] Runtime directories that appear empty may still encode expected output roots for scripts or previous runs. -> Mitigation: inventory path usage in source/config/tests and produce an audit with proposed changes before any filesystem mutation.
- [Risk] Confusing original source data with a working copy could corrupt provenance. -> Mitigation: explicitly classify original source, working copy, derived data, model, log, and output roots, and treat original source data as read-only.

## Migration Plan

- Audit the active `eq-mac` package set, import behavior, CLI smoke checks, and unit tests.
- Audit original cloud data, cloud working copies, repo symlinks, ProjectsRuntime layout, sizes, empty directories, and code/config references before treating Mac local development as ready.
- Produce the directory status report and data/runtime contract, then obtain approval before proceeding beyond environment inspection and non-mutating validation.
- Ask the user to run only those normal-terminal MPS commands that Codex cannot validate directly because of sandbox Metal access.
- Add the macOS environment file only after the target package set is known and justified.
- Add bounded MPS training and full supported contract-first pipeline integration tests using existing runtime assets before treating the environment as certified.
- Apply the backend-selection code changes and tests.
- Validate CLI and focused unit tests in `eq-mac`.
- Validate true MPS availability from the user's normal terminal because sandbox MPS detection is not reliable.
