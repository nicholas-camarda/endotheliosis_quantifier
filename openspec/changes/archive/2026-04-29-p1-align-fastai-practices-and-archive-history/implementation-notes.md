## Implementation Notes

Implemented on 2026-04-29.

## Reuse-First Inventory

- Reused `scripts/check_openspec_explicitness.py` for OpenSpec governance validation; no new OpenSpec checker was added.
- Reused existing pytest surfaces for contract coverage:
  - `tests/test_segmentation_training_contract.py` for full-image and manifest-backed pairing behavior.
  - `tests/test_training_entrypoint_contract.py` for transfer-base loading behavior.
  - `tests/unit/test_docs_quarantine.py` for active-doc and active-source quarantine checks.
- Added `tests/unit/test_run_io.py` because there was no focused run-I/O unit test surface for required artifact write failures and optional plot warning behavior. This is a pytest test module, not a standalone script.
- Reused `docs/HISTORICAL_NOTES.md`, `docs/README.md`, and `docs/archive/` outputs from `oracle-current-docs-quarantine`; no duplicate archive pages or docs index were created.
- Reused existing `src/eq/run_config.py`, `src/eq/__main__.py`, and `src/eq/utils/paths.py` contracts by validating `eq --help` and leaving workflow IDs/path policy unchanged.

## Code Changes

- Replaced remaining FastAI wildcard imports in:
  - `src/eq/training/train_glomeruli.py`
  - `src/eq/training/train_mitochondria.py`
  - `src/eq/training/transfer_learning.py`
- Removed the secondary mask-candidate search from `src/eq/data_management/datablock_loader.py::get_items_full_images`.
- Added manifest pair records so manifest-backed `raw_data/cohorts` items carry explicit image and mask paths through DataBlock `get_x` and `get_y`.
- Added `training_item_image_path(...)` and `training_item_mask_path(...)` helpers so comparison and negative-crop utilities can consume both direct-root `Path` items and manifest pair records.
- Changed required training artifacts in `src/eq/utils/run_io.py` to fail with path-bearing exceptions:
  - split manifest
  - training history
  - run metadata text/JSON/config copy
  - exported learner file
- Kept plot artifacts warning-only, with artifact path and exception details in the warning.
- Removed the `torch.load(model_path, ...)` transfer-base rescue path; transfer loading now requires the current-namespace FastAI learner artifact to load through `load_learner`.

## Documentation Changes

- Confirmed historical FastAI material was already quarantined and indexed by `oracle-current-docs-quarantine`.
- Updated `docs/SEGMENTATION_ENGINEERING_GUIDE.md` to state that FastAI `load_learner` is used only for trusted current-namespace repository artifacts in the certified environment, and that loadability is runtime compatibility evidence rather than scientific promotion evidence.

## Validation

- `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m ruff check .` passed.
- `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest -q` passed: `258 passed, 3 skipped, 8 warnings`.
- `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m eq --help` passed.
- `python3 scripts/check_openspec_explicitness.py p1-align-fastai-practices-and-archive-history` passed.
- `OPENSPEC_TELEMETRY=0 openspec validate p1-align-fastai-practices-and-archive-history --strict` passed.

## Archive Sync

- `openspec archive p1-align-fastai-practices-and-archive-history --yes` archived the change as `openspec/changes/archive/2026-04-29-p1-align-fastai-practices-and-archive-history/`.
- Archive sync created `openspec/specs/fastai-training-hygiene/spec.md` and `openspec/specs/historical-documentation-quarantine/spec.md`.
- Archive sync updated `openspec/specs/repo-wide-quality-review/spec.md` and `openspec/specs/segmentation-training-contract/spec.md`.
- `openspec validate --specs --strict` passed after archive sync: `21 passed, 0 failed`.
- Remaining active changes revalidated cleanly after archive sync:
  - `oracle-canonical-quantification-input-contract`
  - `oracle-harden-audit-contracts`
  - `label-free-roi-embedding-atlas`

## Residual Risk

- This change hardens runtime compatibility and artifact provenance, but it does not promote any segmentation model scientifically.
- Existing pytest warnings are unrelated deprecation/runtime warnings in dependencies and quantification tests; they did not block this P1 contract.
