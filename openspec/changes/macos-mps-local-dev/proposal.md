## Why

Apple Silicon local development should be a supported project path because the current environment file is CUDA/WSL-oriented while the code already has MPS-aware hardware detection. The immediate need is to make the new `eq-mac` environment reproducible and remove CUDA-only assumptions from direct inference paths without changing the research data contracts or production CUDA workflow.

## What Changes

- Add a macOS Apple Silicon environment definition for local development with the PyTorch, fastai, and scientific package versions verified in `eq-mac`.
- Treat `eq-mac` environment correctness as the first implementation gate before backend code changes.
- Report the status of all relevant data/runtime directories and get approval for the data/runtime contract before proceeding beyond environment inspection and non-mutating validation.
- Keep the existing CUDA-oriented `environment.yml` as the WSL/CUDA environment.
- Update direct inference backend selection so `auto` can select MPS when CUDA is unavailable and MPS is available.
- Ensure backend-specific logging and benchmark metadata do not call CUDA-only APIs while running on MPS or CPU.
- Add a real local-runtime integration test that uses the existing ProjectsRuntime preeclampsia raw project, Label Studio annotation export, and a current-namespace ProjectsRuntime glomeruli segmentation model, then runs the supported contract-first quantification path through embeddings/model outputs.
- Add bounded MPS training validation so Mac certification covers more than imports and CLI startup.
- Produce and get approval for a data/runtime contract that answers ownership and storage questions across the OneDrive original source, OneDrive SideProjects working copy, and `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier`, including empty nested output directories and the relationship between `data/`, `raw_data/`, and `derived_data/`.
- Add tests for CUDA, MPS, and CPU device-selection behavior.
- Do not change raw data, derived data, model artifact, log, or output storage boundaries.
- Do not change Label Studio score contracts, union ROI semantics, embedding schemas, ordinal output schemas, or scientific interpretation claims.

## Capabilities

### New Capabilities

- `macos-mps-local-development`: Defines reproducible Apple Silicon local-development setup and observable MPS-aware CLI/inference behavior.

### Modified Capabilities

None.

## Impact

- Affected config: new `environment-macos.yml`.
- Affected runtime/data layout: audit of OneDrive original data, OneDrive SideProjects working copies, and `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier` `raw_data/`, `derived_data/`, `models/`, `logs/`, `output/`, and `data/`.
- Affected modules: `src/eq/utils/hardware_detection.py` if central device-selection behavior needs tightening, and `src/eq/inference/gpu_inference.py` for MPS-aware direct inference.
- Affected CLI checks: `eq capabilities`, `eq mode --show`, `python -m eq --help`, and model/embedding commands that depend on backend selection.
- Affected tests: `tests/unit/test_hardware_detection.py`, focused inference-device tests, and a new local-runtime integration test for the supported contract-first pipeline.
- Compatibility: existing CUDA environments, local data directories, model files, quantification outputs, review artifacts, and path resolution behavior remain unchanged.
