## Why

The FastAI segmentation stack currently passes runtime tests, but the repo still carries lint failures, broad FastAI star imports, fallback image/mask pairing behavior, warning-only artifact failures, and historical FastAI guidance that can be mistaken for current workflow documentation. This change makes the current FastAI training contract explicit, fail-closed, and lint-clean while moving historical material into archive/reference documentation that does not interfere with the present-state README and guides.

## What Changes

- Replace active training-path FastAI star imports with explicit imports in `src/eq/training/train_glomeruli.py`, `src/eq/training/train_mitochondria.py`, and `src/eq/training/transfer_learning.py` so `ruff check .` no longer reports undefined FastAI names.
- Remove fallback image/mask pairing from `src/eq/data_management/datablock_loader.py`; supported training roots must satisfy the single current `images/` plus `masks/` contract or fail before model construction.
- Reclassify training artifact write failures in `src/eq/utils/run_io.py` so required evidence artifacts fail hard while explicitly optional visualizations may warn without changing model-training semantics.
- Preserve FastAI `load_learner` usage only for trusted, current-namespace supported artifacts and ensure docs describe this as a trusted-pickle/current-namespace contract, not a legacy compatibility path.
- Move historical FastAI integration/planning content out of current operational docs and into accessible archive/reference documentation under `docs/HISTORICAL_NOTES.md` and `docs/archive/`.
- Update current docs so `README.md`, `docs/INTEGRATION_GUIDE.md`, `docs/SEGMENTATION_ENGINEERING_GUIDE.md`, `docs/ONBOARDING_GUIDE.md`, `docs/README.md`, and related active guidance describe only the supported current workflow.
- Apply the same rule code-wise: active code must not contain historical shims, workaround branches, silent fallback paths, or compatibility rescue logic for unsupported legacy artifacts.
- Streamline the repo during this change by reusing existing scripts, tests, helpers, and docs surfaces before adding new ones; do not create duplicate scripts or overlapping tooling for jobs the repo already performs.
- Add or update tests that enforce no fallback pairing in active training, explicit FastAI imports/lint cleanliness, and historical-doc quarantine.
- Validate the change with `ruff check .`, `python -m pytest -q`, `python -m eq --help`, and OpenSpec strict validation.

## Capabilities

### New Capabilities

- `fastai-training-hygiene`: Defines the active FastAI training hygiene contract: explicit imports, fail-closed pairing, required artifact handling, and trusted current-namespace learner loading.
- `historical-documentation-quarantine`: Defines how historical FastAI migration, legacy artifact, and older integration material remains accessible without appearing as current operational guidance.

### Modified Capabilities

- `segmentation-training-contract`: Tightens the supported segmentation training contract by explicitly disallowing fallback image/mask pairing and warning-only failure of required training evidence artifacts.
- `repo-wide-quality-review`: Adds concrete final validation expectations for FastAI lint/hygiene cleanup, historical-doc separation, reuse-before-new-tooling, and repo streamlining as follow-up from the repo-wide quality review.

## Explicit Decisions

- Change ID: `p1-align-fastai-practices-and-archive-history`.
- Active FastAI training modules in scope:
  - `src/eq/data_management/datablock_loader.py`
  - `src/eq/training/train_glomeruli.py`
  - `src/eq/training/train_mitochondria.py`
  - `src/eq/training/transfer_learning.py`
  - `src/eq/utils/run_io.py`
  - `src/eq/data_management/model_loading.py`
- Current workflow docs in scope:
  - `README.md`
  - `docs/README.md`
  - `docs/INTEGRATION_GUIDE.md`
  - `docs/ONBOARDING_GUIDE.md`
  - `docs/SEGMENTATION_ENGINEERING_GUIDE.md`
  - `docs/TECHNICAL_LAB_NOTEBOOK.md`
- Historical material should be retained in:
  - `docs/HISTORICAL_NOTES.md`
  - `docs/archive/fastai_legacy_integration.md`
  - `docs/archive/fastai_pipeline_integration_plan.md`
  - `docs/archive/fastai_historical_implementation_analysis.md`
- Historical current-doc content to extract before rewriting active docs:
  - `docs/INTEGRATION_GUIDE.md`
  - `docs/PIPELINE_INTEGRATION_PLAN.md`
  - `docs/HISTORICAL_IMPLEMENTATION_ANALYSIS.md`
- Supported current config/workflow entrypoints remain:
  - `eq run-config --config configs/mito_pretraining_config.yaml --dry-run`
  - `eq run-config --config configs/glomeruli_finetuning_config.yaml --dry-run`
  - `eq run-config --config configs/glomeruli_candidate_comparison.yaml --dry-run`
- Existing helper/check surfaces must be reused before new scripts are added:
  - `scripts/check_openspec_explicitness.py`
  - existing pytest tests under `tests/`
  - existing path helpers in `src/eq/utils/paths.py`
  - existing config runner in `src/eq/run_config.py`
  - existing CLI parser surfaces in `src/eq/__main__.py`
- The final validation command set is:
  - `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m ruff check .`
  - `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest -q`
  - `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m eq --help`
  - `python3 scripts/check_openspec_explicitness.py openspec/changes/p1-align-fastai-practices-and-archive-history`
  - `OPENSPEC_TELEMETRY=0 openspec validate p1-align-fastai-practices-and-archive-history --strict`

## Impact

- Affects FastAI segmentation training source under `src/eq/data_management/`, `src/eq/training/`, and `src/eq/utils/run_io.py`.
- Affects documentation organization under `docs/`, especially current README/implementation/onboarding guidance and historical FastAI integration/planning documents.
- Affects test and lint expectations by making `ruff check .` part of the required green gate for this cleanup.
- Affects repo organization by requiring a reuse-first audit before adding new scripts, checks, or overlapping docs surfaces.
- Does not add compatibility shims for legacy FastAI pickle artifacts.
- Does not change the scientific promotion status of any segmentation model artifact.
- Does not move raw data, derived data, model artifacts, logs, or runtime outputs into Git.
