# macOS MPS Local Development Gate Report

Date: 2026-04-22

This report began as the non-mutating implementation gate for
`macos-mps-local-dev`. The initial audit did not change raw data, derived data,
model artifacts, logs, output artifacts, symlinks, or path-helper defaults.
Later training validation created new model artifacts under the approved local
ProjectsRuntime model root; those artifacts are documented below.

## Initial Environment Inspection

This initial snapshot is retained to explain the original blocker. It is
superseded by the final environment certification below.

Active environment:

- Conda environment: `/Users/ncamarda/mambaforge/envs/eq-mac`
- Python: `3.10.20`
- Project package: `endotheliosis-quantifier 0.1.0`
- PyTorch import: `torch 2.0.0`
- Torchvision metadata: `0.15.1`
- FastAI metadata: `2.7.13`
- SciPy import: `1.10.1`
- Pandas import: `2.0.3`
- scikit-image import: `0.21.0`
- Pillow import: `9.5.0`
- OpenCV import: `4.13.0`

Observed accelerator state from Codex sandbox:

- `torch.backends.mps.is_built()`: `True`
- `torch.backends.mps.is_available()`: `False`
- `torch.cuda.is_available()`: `False`

The user's normal macOS terminal previously verified true MPS availability with
`torch 2.0.0`, `mps built True`, `mps available True`, and an allocated
`tensor([1.], device='mps:0')`. Treat Codex sandbox Metal availability as
non-authoritative.

Smoke checks completed:

- `mamba run -n eq-mac python -m eq --help`: passed
- `mamba run -n eq-mac python -m pytest -q tests/unit/test_imports.py tests/unit/test_hardware_detection.py tests/unit/test_config_paths.py tests/unit/test_quantification_pipeline.py`: passed, `18 passed`

Environment issues to resolve or intentionally accept before backend work:

- `pip check` reports: `torch 2.0.0 is not supported on this platform`.
- NumPy import and metadata disagree:
  - Imported NumPy: `1.24.3` from `/Users/ncamarda/mambaforge/envs/eq-mac/lib/python3.10/site-packages/numpy/__init__.py`
  - Package metadata: `numpy 2.2.6` from `/Users/ncamarda/mambaforge/envs/eq-mac/lib/python3.10/site-packages/numpy-2.2.6.dist-info`
- `conda list` reports `numpy 2.2.6` even though importing NumPy reports `1.24.3`.
- `conda list` emitted stale `conda-meta` cleanup warnings for packages including
  numpy, scipy, pandas, pillow, scikit-learn, setuptools, tifffile, and
  matplotlib-base.

Assessment: the environment can run current smoke checks, but it is not yet a
clean certified environment. The macOS environment file should pin a coherent
stack rather than copying the inconsistent metadata state.

Final environment certification update:

- The old `torch 2.0.0.post2` / `fastai 2.7.13` stack could allocate MPS
  tensors, but FastAI DynamicUNet training crashed during real MPS execution.
- `eq-mac` was updated from `environment-macos.yml` to the verified
  train-capable Apple Silicon stack.
- Final import and metadata checks now agree:
  - NumPy: `2.2.6`
  - SciPy: `1.15.2`
  - Pandas: `2.3.3`
  - Pillow: `12.0.0`
  - scikit-image: `0.25.2`
  - OpenCV: `4.12.0`
  - PyTorch: `2.10.0`
  - torchvision: `0.26.0`
  - FastAI: `2.8.7`
- `mamba run -n eq-mac python -m pip check` reports no broken requirements.
- `mamba run -n eq-mac python -m eq --help` passes.
- `mamba run -n eq-mac env PYTORCH_ENABLE_MPS_FALLBACK=1 MPLCONFIGDIR=/tmp python -m pytest -q tests/test_datablock_loader.py tests/unit/test_training_backend_smoke.py tests/unit/test_gpu_inference.py tests/unit/test_hardware_detection.py` passes with `20 passed`.
- `mamba run -n eq-mac ruff check src/eq/data_management/standard_getters.py src/eq/data_management/datablock_loader.py src/eq/utils/run_io.py tests/test_datablock_loader.py` passes.

## Directory Status Report

| Path | Role | Size | Symlink status | Notable contents | References | Recommendation | Risk if changed |
| --- | --- | ---: | --- | --- | --- | --- | --- |
| `/Users/ncamarda/Library/CloudStorage/OneDrive-Personal/phd/projects/Lauren PreEclampsia/Lauren_PreEclampsia_Data` | Apparent immutable original source | 3.2G | Real cloud directory | Raw TIF subject folders, JPG training/testing images, master grading spreadsheet | Mentioned only in OpenSpec audit artifacts | Keep as read-only provenance | Mutating it would compromise source-data provenance |
| `/Users/ncamarda/Library/CloudStorage/OneDrive-Personal/SideProjects/endotheliosis_quantifier` | Cloud working copy and backup-like project artifact root | 6.1G | Real cloud directory | `data/`, `models/`, `output/` | Repo symlinks point here for `raw_data`, `derived_data`, `models`, `output` | Keep for now; decide whether it remains the symlink target | Changing it could break current repo symlinks and cloud-backed model/output access |
| `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier` | Local runtime working tree | 6.6G | Real local directory | `raw_data/`, `derived_data/`, `models/`, `logs/`, `output/`, `data/` | Spec integration-test paths; `logs` symlink points here | Keep as preferred active runtime root, pending approval | Changing it could break local runtime certification inputs and existing artifacts |
| `/Users/ncamarda/Projects/endotheliosis_quantifier/raw_data` | Repo root convenience symlink | n/a | Symlink to OneDrive SideProjects `data/raw/raw_data` | Preeclampsia raw project through OneDrive | README examples use `data/raw_data`; configs use `data/raw_data`; root symlink itself is not the path-helper default | Unknown; likely remove or retarget only after contract approval | Retargeting can change where ad hoc commands read raw projects |
| `/Users/ncamarda/Projects/endotheliosis_quantifier/derived_data` | Repo root convenience symlink | n/a | Symlink to OneDrive SideProjects `data/processed/derived_data` | Glomeruli and mitochondria derived datasets through OneDrive | Root symlink is not the path-helper default | Unknown; likely remove or retarget only after contract approval | Retargeting can change ad hoc derived-data audits |
| `/Users/ncamarda/Projects/endotheliosis_quantifier/models` | Model artifact symlink | n/a | Symlink to OneDrive SideProjects `models` | Segmentation models and model evaluations | Path helper default `models`; configs and CLI examples use `models/...` | Keep until promoted-model policy is approved | Breaking this path can break model loading commands |
| `/Users/ncamarda/Projects/endotheliosis_quantifier/output` | Output artifact symlink | n/a | Symlink to OneDrive SideProjects `output` | Prediction and model-evaluation outputs | CLI defaults and examples use `output/...` | Unknown; likely retarget to local runtime or replace with explicit env override after approval | Breaking this path can hide prior outputs or send new outputs to the wrong storage |
| `/Users/ncamarda/Projects/endotheliosis_quantifier/logs` | Log artifact symlink | n/a | Symlink to ProjectsRuntime `logs` | Local logs | Path helper default `logs` | Keep as local runtime log target | Changing it can scatter logs across cloud and local roots |
| `/Users/ncamarda/Projects/endotheliosis_quantifier/data/raw_data` | Path-helper raw-data default | 0 | Real empty repo directory | Empty | `get_data_path()` default, config tests, README examples | Migrate/retarget candidate; should not stay empty if it is the canonical path | Current default does not point at the existing raw project |
| `/Users/ncamarda/Projects/endotheliosis_quantifier/data/derived_data` | Path-helper derived-data default | 0 | Real empty repo directory | Empty | `get_output_path()` and `get_cache_path()` defaults, config tests, README examples | Migrate/retarget candidate; should not stay empty if it is the canonical path | Current default does not point at the existing derived datasets |
| `/Users/ncamarda/Projects/endotheliosis_quantifier/data/preeclampsia_data` | Legacy processed-data symlink | n/a | Symlink to OneDrive SideProjects `data/processed/preeclampsia_data` | `cache/`, empty `train/` | Legacy production runner defaults use `data/preeclampsia_data` | Unknown; legacy-only unless production runner is made current | Removing it could break legacy production commands |
| `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/raw_data` | Local raw working data root | 1.3G | Real local directory | `preeclampsia_project` | Spec integration test uses this root directly | Keep as active raw working root | Changing it would break current local-runtime test inputs |
| `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/raw_data/preeclampsia_project/clean_backup` | Current curated raw project for contract-first quantification | 195M | Real local directory | `images/`, `masks/`, `cache/` | Spec integration test input | Keep | Changing it can invalidate the 88-example contract baseline |
| `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/raw_data/preeclampsia_project/annotations/annotations.json` | Label Studio score export | n/a | Real local file | Current annotation export | Spec integration test input | Keep | Changing it can alter joined score counts |
| `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/raw_data/preeclampsia_project/backup_before_reorganization` | Historical raw-project backup | 1.0G | Real local directory | Subject folders, old annotations, old masks | No direct source reference found in code/config/tests | Keep until provenance policy decides backup retention | Removing it loses rollback context for raw reorganization |
| `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/derived_data` | Local derived artifact root | 4.5G | Real local directory | Glomeruli and mitochondria patch datasets and caches | Configs and path-helper terminology map here conceptually, not by default | Keep as active derived-data root candidate | Changing it could invalidate training caches and patch datasets |
| `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/models` | Local model artifact root | 739M | Real local directory | Mitochondria model, glomeruli transfer model, evaluations | Spec integration test model path | Keep as active model root candidate | Changing it could break local model loading |
| `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output` | Local generated-output root | 5.2M | Real local directory | Prediction reports, plots, test regression outputs, empty expected subdirs | Output-manager and CLI terminology map here conceptually, not by default | Keep for now; classify empty dirs below | Removing nonempty outputs may lose review artifacts |
| `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/logs` | Local log root | 2.3M | Real local directory | Runtime logs | Repo `logs` symlink target | Keep | Removing logs can reduce debuggability |
| `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/data` | Ambiguous local legacy/staging root | 16K | Real local directory | `preeclampsia_data/train` empty, `.DS_Store` | Legacy production defaults use repo `data/preeclampsia_data`, not this absolute path | Unknown; likely remove candidate after confirming no active use | Removing too early could break old ad hoc scripts |

Empty runtime directories found:

- Remove candidates after approval if no active scripts require placeholders:
  - `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/mitochondria/models`
  - `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/mitochondria/results`
  - `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/mitochondria/plots`
  - `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/derived/models`
  - `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/derived/results`
  - `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/derived/plots`
  - `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/mitochondria_training/mito_base_model`
  - `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/glomeruli_finetuning/plots`
  - `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/models/regression`
  - `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/data/preeclampsia_data/train`

Code/config/test references:

- `src/eq/utils/paths.py` defines the current code-facing defaults:
  - raw data: `data/raw_data`
  - derived output: `data/derived_data`
  - cache: `data/derived_data/cache`
  - models: `models`
  - logs: `logs`
- Config files use `data/raw_data/...`, `data/derived_data/...`, `models/...`, and `output/...`.
- The README describes `data/raw_data`, `data/derived_data`, `models`, `logs`, and `output` as local-only artifact roots.
- Legacy production code still defaults to `data/preeclampsia_data`.
- Tests assert the path-helper defaults, so any path-helper change requires explicit test updates.

## Proposed Data/Runtime Contract

1. Immutable original source data lives in OneDrive project source storage:
   `/Users/ncamarda/Library/CloudStorage/OneDrive-Personal/phd/projects/Lauren PreEclampsia/Lauren_PreEclampsia_Data`.
   Treat this as read-only provenance.

2. Curated raw working projects live in the active runtime raw root:
   `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/raw_data`.
   The current certification input is
   `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/raw_data/preeclampsia_project/clean_backup`.

3. Derived patch datasets and caches live in the active runtime derived root:
   `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/derived_data`.
   The code-facing role is `get_output_path()` / `EQ_OUTPUT_PATH`, despite the confusing name `output_path`.

4. Active model artifacts used for local certification live in:
   `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/models`.
   The glomeruli certification model is:
   `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/models/segmentation/glomeruli/transfer/glom_dynamic_unet_mps-transfer_loss-dice_s1lr1e-3_s2lr_lrfind_e30_b16_lr1e-3_sz256/glom_dynamic_unet_mps-transfer_loss-dice_s1lr1e-3_s2lr_lrfind_e30_b16_lr1e-3_sz256.pkl`.

5. Active logs live in:
   `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/logs`.
   The existing repo `logs` symlink already matches this.

6. Disposable and review outputs live in:
   `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output`
   during local development and certification. Publishable outputs should be copied
   or promoted intentionally, not produced directly by default into original source
   data or ambiguous project directories.

7. The OneDrive SideProjects root remains a cloud-backed working copy or backup
   until explicitly reclassified. It should not be the default active write target
   for new Mac local-development outputs unless that is chosen deliberately.

8. Repo symlink policy needs approval:
   - Recommended direction: make code-facing defaults resolve to `data/raw_data`,
     `data/derived_data`, `models`, `logs`, and `output`, with those paths either
     symlinked consistently to ProjectsRuntime or overridden with documented
     `EQ_*` environment variables.
   - Do not keep the mixed state where root-level `raw_data` and `derived_data`
     point to OneDrive while path-helper defaults point to empty `data/raw_data`
     and `data/derived_data`.

9. The ambiguous local `data/` subtree under ProjectsRuntime is not part of the
   proposed active contract unless a specific legacy production-runner workflow is
   revived. Its empty `preeclampsia_data/train` directory is a cleanup candidate
   after approval.

## Approval Gate

The data/runtime contract was approved on 2026-04-22 with no `Research` folder
reorganization. Keep the existing OneDrive `phd` source path as provenance and
do not move projects into a new OneDrive organization as part of this change.

Before changing backend code, adding the pipeline integration test, changing
path-helper defaults, retargeting symlinks, or cleaning runtime directories,
resolve the active `eq-mac` environment issue:

- `environment-macos.yml` has been added and verified with conda-forge native
  osx-arm64 `pytorch=2.10.0`, `torchvision=0.26.0`, and `fastai=2.8.7`.
- The existing `eq-mac` environment was rebuilt from `environment-macos.yml`.
- The rebuilt environment completed MPS tensor allocation, FastAI DynamicUNet
  forward/backward optimization, and the local-runtime pipeline integration
  test.

Approved contract points:

- the proposed active runtime root: `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier`
- the role of the OneDrive SideProjects copy
- whether repo-facing artifact paths should be symlinks to ProjectsRuntime or
  should remain path-helper defaults controlled by `EQ_*` variables
- whether empty runtime directories can be removed after path-reference checks

## Pipeline Validation Result on 2026-04-22

The repository-owned local-runtime integration test was added at
`tests/integration/test_local_runtime_quantification_pipeline.py`. It uses the
approved runtime raw project, Label Studio annotation export, and current
glomeruli segmentation `.pkl`, writes only under pytest `tmp_path`, verifies the
88 joined scored-example baseline, and asserts contract, ROI, embedding,
ordinal-model, metrics, and review artifacts.

Initial `eq-mac` result with the old promoted model:

```text
mamba run -n eq-mac python -m pytest -q tests/integration/test_local_runtime_quantification_pipeline.py
FAILED ModuleNotFoundError: No module named 'numpy._core'
```

The failure occurs while FastAI/PyTorch unpickles the existing glomeruli model,
before ROI embeddings run. Follow-up probes found:

- The current `eq-mac` stack is `torch 2.0.0.post2`, `fastai 2.7.13`, and
  `numpy 1.24.3`; this stack cannot import the model's `numpy._core.multiarray`
  pickle reference.
- A throwaway FastAI 2.8/PyTorch 2.6/NumPy 2 environment is not valid for this
  artifact because FastAI 2.8 rejects old `fastcore.transform` pickles and
  instructs users to load them with `fastai<2.8.0`.
- A throwaway FastAI 2.7.19/PyTorch 2.6/NumPy 2.2.6 environment resolves the
  NumPy and torch deserialization errors, but the model still references the
  removed project module
  `eq.segmentation.train_glomeruli_transfer_learning`.
- In that same throwaway environment, a temporary in-memory legacy namespace for
  only `eq.segmentation.train_glomeruli_transfer_learning.get_y_func` allowed the
  trusted model artifact to load as `Learner DynamicUnet cpu`.

This is not an MPS tensor-allocation failure. It is a promoted-model artifact
compatibility problem. The current single correct resolution should be one of:

1. Re-export or retrain/promote a glomeruli model artifact from the current
   package namespace and target Mac environment.
2. Explicitly approve a maintained legacy model-loading namespace as part of the
   supported model contract.

The chosen resolution was to retrain and export current-namespace model
artifacts in the certified Mac environment:

- Mitochondria pretraining output:
  `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/models/segmentation/mitochondria/mito_dynamic_unet_mps-pretrain_e50_b16_lr1e-3_sz256/mito_dynamic_unet_mps-pretrain_e50_b16_lr1e-3_sz256.pkl`
- Mitochondria final metrics from the training history:
  `valid_loss=0.020002959296107292`, `dice=0.943278765980747`,
  `jaccard=0.8926467412725055`.
- Glomeruli transfer output:
  `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/models/segmentation/glomeruli/transfer/glom_dynamic_unet_mps-transfer_loss-dice_s1lr1e-3_s2lr_lrfind_e30_b16_lr1e-3_sz256/glom_dynamic_unet_mps-transfer_loss-dice_s1lr1e-3_s2lr_lrfind_e30_b16_lr1e-3_sz256.pkl`
- Glomeruli final metrics from the training history:
  `valid_loss=0.514433741569519`, `dice=0.5560158748514132`,
  `jaccard=0.3850567780959495`.

Final `eq-mac` pipeline result:

```text
mamba run -n eq-mac env PYTORCH_ENABLE_MPS_FALLBACK=1 MPLCONFIGDIR=/tmp python -m pytest -q tests/integration/test_local_runtime_quantification_pipeline.py
1 passed, 104 warnings in 21.64s
```

The warnings include the expected FastAI trusted-pickle warning and scikit-learn
numeric warnings during ordinal-model fitting. They do not block the pipeline
gate, but they remain evidence that downstream model-quality and numeric
stability should be reviewed separately from Mac runtime compatibility.

Important model-quality note: the glomeruli model is loadable and proves that
the Mac environment can run the pipeline with a current-namespace artifact, but
its validation history is flat and sampled validation predictions were
degenerate all-positive masks. This is most consistent with training from a
static patch dataset that contains positive-only glomeruli patches. Treat this
export as a Mac compatibility artifact, not as a promoted scientific
segmentation model, until glomeruli training is rerun against a data contract
that includes background/negative examples or an approved dynamic-patching
training setup.

## Glomeruli Static Patch Dataset Retirement

On 2026-04-22, the old glomeruli static patch directories were moved out of the
active ProjectsRuntime derived-data tree after approval. The active
`/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/derived_data/glomeruli_data`
tree no longer contains `image_patches/` or `mask_patches/` under `training/`,
`testing/`, or `prediction/`.

Retired location:

```text
/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/_retired/glomeruli_static_patch_datasets_2026-04-22
```

Moved runtime directories:

- `derived_data/glomeruli_data/training/image_patches`
- `derived_data/glomeruli_data/training/mask_patches`
- `derived_data/glomeruli_data/testing/image_patches`
- `derived_data/glomeruli_data/testing/mask_patches`
- `derived_data/glomeruli_data/prediction/image_patches`
- `derived_data/glomeruli_data/prediction/mask_patches`

Verified retired copy size: `184M`, `16558` files. This is runtime-only data
outside Git tracking. Future glomeruli training should use the full-image
`images/` and `masks/` dynamic-patching contract rather than these retired
static patches.
