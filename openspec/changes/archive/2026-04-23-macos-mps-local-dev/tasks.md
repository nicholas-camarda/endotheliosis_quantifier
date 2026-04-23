## 1. Environment Certification

- [x] 1.1 Inspect the active `eq-mac` package set for Python, PyTorch, torchvision, fastai, NumPy, SciPy, pandas, albumentations, scikit-image, Pillow, OpenCV, and project editable install state.
- [x] 1.2 Run import, CLI, and focused unit-test smoke checks in `eq-mac` before changing backend code.
- [x] 1.3 Audit the apparent original cloud source, OneDrive SideProjects working copy, repository symlink targets, and `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier` top-level runtime directories, sizes, and empty nested directories.
- [x] 1.4 Trace source/config/test references to `data/`, `raw_data/`, `derived_data/`, `models/`, `logs/`, and `output/` before recommending any data or runtime layout cleanup.
- [x] 1.5 Classify each original source, working-copy, derived-data, model, log, output, empty, or ambiguous directory as keep, remove candidate, migrate candidate, or unknown, and document why.
- [x] 1.6 Define the intended Mac data/runtime contract for original source data, raw working data, derived data, models, logs, and output roots using the existing path-helper terminology.
- [x] 1.7 Ensure the contract answers where immutable original data lives, where curated raw working projects live, where derived patch datasets live, where active training outputs/checkpoints live, which model artifacts are promoted/backed up, which outputs are disposable versus publishable, and whether repo symlinks should point to ProjectsRuntime, OneDrive, or be removed.
- [x] 1.8 Present a directory status report covering path, role, size, notable contents, symlink status, code/config/test references, recommended action, and risk if changed.
- [x] 1.9 Present the proposed data/runtime contract for approval before proceeding beyond environment inspection and non-mutating validation.
- [x] 1.10 Do not implement environment files, backend code changes, pipeline tests, cleanup, symlink changes, path-helper default changes, or runtime directory moves that depend on the contract until approval is received.
- [x] 1.11 Ask the user to run exact normal-terminal commands for any MPS validation Codex cannot perform from the sandbox.
- [x] 1.12 Resolve or document any `eq-mac` dependency conflict before proceeding to backend code changes.
- [x] 1.13 Add `environment-macos.yml` for Apple Silicon local development, keeping `environment.yml` unchanged for WSL/CUDA.
- [x] 1.14 Pin the macOS PyTorch stack to the verified MPS-compatible versions and align core scientific/image packages with the certified `eq-mac` imports.
- [x] 1.15 Validate the environment file syntax without creating tracked artifacts.

## 2. Backend Selection

- [x] 2.1 Update direct glomeruli inference auto-device selection to use the existing CUDA > MPS > CPU backend policy.
- [x] 2.2 Ensure explicit MPS selection is accepted only when MPS is available.
- [x] 2.3 Restrict CUDA device-name, device-properties, synchronization, and memory calls to CUDA-only execution paths.
- [x] 2.4 Preserve visible failures for selected-backend runtime errors; do not add silent CPU retry behavior.

## 3. Tests

- [x] 3.1 Extend hardware/device-selection tests to cover MPS availability without CUDA.
- [x] 3.2 Add focused tests for direct inference auto-selection across CUDA, MPS, and CPU.
- [x] 3.3 Add focused tests proving MPS and CPU benchmark paths do not call CUDA runtime APIs.
- [x] 3.4 Add or adapt a bounded synthetic segmentation training smoke that runs one real optimization step on the selected backend.
- [x] 3.5 Add a repository-owned local-runtime contract-first pipeline integration test that uses `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/raw_data/preeclampsia_project/clean_backup`, `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/raw_data/preeclampsia_project/annotations/annotations.json`, and a current-namespace ProjectsRuntime glomeruli segmentation `.pkl` model.
- [x] 3.6 Make the integration test verify the existing contract baseline: nonzero raw inventory, joined Label Studio scores, scored examples, and the current 88 joined scored examples unless the runtime audit intentionally updates that baseline.
- [x] 3.7 Make the integration test run `run_contract_first_quantification(..., stop_after='model')` and assert contract, scored examples, ROI, embeddings, ordinal predictions, metrics, and review artifacts exist.
- [x] 3.8 Ensure the integration test writes outputs only under pytest temporary directories and does not mutate existing runtime input data or model artifacts.

## 4. Validation

- [x] 4.1 Run `env OPENSPEC_TELEMETRY=0 openspec validate macos-mps-local-dev --strict`.
- [x] 4.2 Run `mamba run -n eq-mac python -m eq --help`.
- [x] 4.3 Run `mamba run -n eq-mac python -m eq capabilities` and interpret sandbox MPS results cautiously.
- [x] 4.4 Run `mamba run -n eq-mac python -m eq mode --show`.
- [x] 4.5 Run `mamba run -n eq-mac python -m pytest -q tests/unit/test_imports.py tests/unit/test_hardware_detection.py tests/unit/test_config_paths.py tests/unit/test_quantification_pipeline.py`.
- [x] 4.6 From a normal macOS terminal, verify true MPS availability with a PyTorch MPS tensor allocation.
- [x] 4.7 From a normal macOS terminal when MPS is available, run the bounded MPS training smoke and record whether the model, data batch, backward pass, and optimizer step complete on MPS.
- [x] 4.8 Run the repository-owned contract-first pipeline integration test in `eq-mac`.
- [x] 4.9 Include the data/runtime layout audit findings in the implementation summary, and do not mutate original cloud data, working copies, or runtime directories without explicit approval.
- [x] 4.10 Encode the distinction between Mac runtime compatibility artifacts and promoted scientific segmentation models so flat or degenerate glomeruli validation cannot be mistaken for model readiness.
- [x] 4.11 Encode the distinction between legacy FastAI pickle artifacts and supported current-namespace segmentation artifacts so old model files cannot be mistaken for current runtime dependencies.
- [x] 4.12 Retire the old glomeruli static patch datasets out of the active ProjectsRuntime derived-data tree so they cannot be accidentally used as glomeruli training inputs.
